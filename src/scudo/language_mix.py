"""Mix per lingua dei casi di training: quote IT/EN/misto configurabili, default 40/40/20.

Le quote si arrotondano col metodo dei resti maggiori; dentro ogni lingua i casi
si scelgono per hash seminato della chiave, quindi il mix è deterministico e
indipendente dall'ordine. Se una lingua non ha abbastanza casi il mix fallisce:
niente riempimento silenzioso con le altre lingue.

Fonti del mix: i casi del build (`scudo.ingest`) con label gold di fonti che
passano il cancello (`dataset_pool`) e il pool di un lotto italiano solo con
spot-check umano accettato (`batch_pool`): un lotto respinto non entra.
"""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, fields
from pathlib import Path

from scudo.cases import LANGUAGES, Case
from scudo.ingest import check_outside_repo
from scudo.italian_run import read_pool
from scudo.manifest import Entry, problems, training_entry
from scudo.spotcheck_it import require_accepted

DEFAULT_SHARES = {"it": 0.4, "en": 0.4, "mixed": 0.2}
_CASE_FIELDS = tuple(f.name for f in fields(Case))


def parse_shares(text: str) -> dict:
    """`it=0.4,en=0.4,mixed=0.2`: lingue di `scudo.cases`, quote >= 0 con somma 1."""
    shares = {language: 0.0 for language in LANGUAGES}
    seen = set()
    for part in text.split(","):
        language, _, value = part.partition("=")
        language = language.strip()
        if language not in shares or language in seen:
            raise ValueError(f"lingua {language!r} sconosciuta o ripetuta")
        seen.add(language)
        try:
            shares[language] = float(value)
        except ValueError:
            raise ValueError(f"quota {value!r} non numerica") from None
        if not math.isfinite(shares[language]) or shares[language] < 0:
            raise ValueError(f"quota {value!r} negativa o non finita")
    if abs(math.fsum(shares.values()) - 1) > 1e-9:
        raise ValueError("le quote devono sommare a 1")
    return shares


def quotas(n: int, shares: dict) -> dict:
    """Casi per lingua: parte intera più i resti maggiori, a parità nell'ordine delle lingue."""
    exact = {language: n * shares.get(language, 0.0) for language in LANGUAGES}
    out = {language: math.floor(value) for language, value in exact.items()}
    by_rest = sorted(LANGUAGES, key=lambda lang: (-(exact[lang] - out[lang]),
                                                  LANGUAGES.index(lang)))
    for language in by_rest[:n - sum(out.values())]:
        out[language] += 1
    return out


def _case(record: dict) -> dict:
    """Valida i campi del caso con `Case`; un record non conforme è un errore."""
    return asdict(Case(**{k: record[k] for k in _CASE_FIELDS}))


def dataset_pool(records: Iterable[dict], entries: list[Entry]) -> tuple[list[dict], dict]:
    """Casi del build con label gold di fonti di training ammesse dal cancello."""
    kept, excluded = [], Counter()
    for record in records:
        entry = training_entry(entries, record.get("source"), record.get("split"))
        if record.get("needs_relabel"):
            excluded["needs_relabel"] += 1
        elif entry is None:
            excluded["source_not_in_manifest"] += 1
        elif problems([entry]):
            excluded["source_blocked"] += 1
        else:
            case = _case(record)
            kept.append({"key": f"{case['source']}:{case['source_id']}:{case['surface']}",
                         "origin": "dataset", **case})
    return kept, dict(sorted(excluded.items()))


def batch_pool(batch: Path) -> list[dict]:
    """Pool deduplicato del lotto italiano, solo con spot-check umano accettato."""
    require_accepted(batch)
    return [{"key": r["key"], "origin": r["origin"], **_case(r), "provenance": r["provenance"]}
            for r in read_pool(batch)]


def _order(seed: int, record: dict) -> str:
    return hashlib.sha256(f"{seed}:{record['key']}".encode()).hexdigest()


def build_mix(pools: dict, n: int, shares: dict, *, seed: int) -> tuple[list[dict], dict]:
    """n casi con le quote per lingua; errore se una lingua non ha abbastanza casi."""
    want = quotas(n, shares)
    short = [f"{lang}: {len(pools.get(lang, []))} casi, ne servono {want[lang]}"
             for lang in LANGUAGES if len(pools.get(lang, [])) < want[lang]]
    if short:
        raise ValueError("pool per lingua insufficienti: " + "; ".join(short))
    mix = []
    for language in LANGUAGES:
        mix += sorted(pools.get(language, []), key=lambda r: _order(seed, r))[:want[language]]
    mix.sort(key=lambda r: _order(seed, r))
    meta = {"n": n, "seed": seed, "shares": shares, "quotas": want,
            "available": {lang: len(pools.get(lang, [])) for lang in LANGUAGES},
            "origins": dict(sorted(Counter(f"{r['origin']}/{r['language']}"
                                           for r in mix).items()))}
    return mix, meta


def pools_by_language(records: Iterable[dict]) -> dict:
    out: dict[str, list[dict]] = {language: [] for language in LANGUAGES}
    for record in records:
        out[record["language"]].append(record)
    return out


def write_mix(dest: Path, mix: list[dict], meta: dict) -> Path:
    dest = check_outside_repo(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "mix.jsonl").open("w", encoding="utf-8") as handle:
        for record in mix:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (dest / "mix-meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return dest
