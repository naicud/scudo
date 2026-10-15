"""Deduplica contro i benchmark held-out: hash esatto e near-duplicate. Solo stdlib.

Il testo si normalizza (NFKC, casefold, punteggiatura in spazi, spazi
compressi). Esatto: SHA-256 del testo normalizzato. Near-duplicate: MinHash a
una permutazione (Li et al., 2012) su shingle di 5 caratteri, 128 bin con
densificazione per rotazione (Shrivastava e Li, 2014); LSH a 25 bande da 5
righe trova i candidati, che sono duplicati se la Jaccard stimata (bin uguali /
128) è almeno 0,8. Tutto è deterministico: hash BLAKE2b, nessun seed casuale.
"""

import hashlib
import re
import unicodedata

SHINGLE = 5
BINS = 128
BANDS, ROWS = 25, 5  # 125 bin nelle bande; P(candidato | J=0,8) ≈ 0,9999
THRESHOLD = 0.8
_SPAN = 1 << 57  # valori per bin: 64 bit di hash meno i 7 dell'indice del bin
_NON_WORD = re.compile(r"[\W_]+")

Signature = tuple[int, ...]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return _NON_WORD.sub(" ", text).strip()


def digest(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def _hash(shingle: str) -> int:
    return int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")


def signature(text: str) -> Signature | None:
    """Firma MinHash del testo normalizzato; None se non resta testo."""
    norm = normalize(text)
    if not norm:
        return None
    shingles = {norm[i:i + SHINGLE] for i in range(max(1, len(norm) - SHINGLE + 1))}
    bins: list[int | None] = [None] * BINS
    for shingle in shingles:
        value = _hash(shingle)
        slot, rest = value % BINS, value // BINS
        if bins[slot] is None or rest < bins[slot]:
            bins[slot] = rest
    out = []
    for i, value in enumerate(bins):
        if value is None:  # rotazione: primo bin pieno a destra, spostato della distanza
            step = next(k for k in range(1, BINS) if bins[(i + k) % BINS] is not None)
            value = bins[(i + step) % BINS] + step * _SPAN
        out.append(value)
    return tuple(out)


def similarity(a: Signature, b: Signature) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / BINS


def _bands(sig: Signature):
    for band in range(BANDS):
        yield band, sig[band * ROWS:(band + 1) * ROWS]


class Index:
    """Testi dei benchmark; `match` dice se un testo di training ne duplica uno."""

    def __init__(self):
        self._exact: dict[str, str] = {}
        self._signatures: list[tuple[str, Signature]] = []
        self._buckets: dict[tuple, list[int]] = {}
        self.sizes: dict[str, int] = {}

    def add(self, benchmark: str, text: str) -> None:
        self.sizes[benchmark] = self.sizes.get(benchmark, 0) + 1
        self._exact.setdefault(digest(text), benchmark)
        sig = signature(text)
        if sig is None:
            return
        self._signatures.append((benchmark, sig))
        for key in _bands(sig):
            self._buckets.setdefault(key, []).append(len(self._signatures) - 1)

    def match(self, text: str) -> tuple[str, str] | None:
        """("exact" | "near", benchmark) del primo benchmark indicizzato che combacia."""
        if (benchmark := self._exact.get(digest(text))) is not None:
            return "exact", benchmark
        sig = signature(text)
        if sig is None:
            return None
        candidates = sorted({i for key in _bands(sig) for i in self._buckets.get(key, ())})
        for i in candidates:
            benchmark, other = self._signatures[i]
            if similarity(sig, other) >= THRESHOLD:
                return "near", benchmark
        return None
