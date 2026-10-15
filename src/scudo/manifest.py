"""Manifest dei dataset (`manifest.toml`) e cancello di licenza e provenienza.

Una voce è una fonte di training (`train`) o un benchmark held-out
(`benchmark`), con revisione fissata, licenza, generatori e filtri: gli stessi
campi della data card. `gate` blocca la build se una voce non è ammessa:
revisione non fissata, licenza non verificata o fuori dall'ADR 0002, campi della
data card vuoti. Per il training bloccano anche le fonti gated e i termini dei
generatori `NO`; `INCERTO` blocca finché la voce non porta una decisione umana
(chi, quando, motivazione). I benchmark servono solo alla valutazione: una
licenza NC è ammessa, i termini dei generatori non bloccano, ma servono i campi
di testo su cui deduplicare. Il registro dei benchmark è `benchmarks()`.
"""

import re
import tomllib
from dataclasses import dataclass, fields
from importlib import resources
from pathlib import Path

from scudo.cases import SURFACES

MANIFEST = resources.files("scudo") / "manifest.toml"
TRAIN_LICENSES = frozenset({"apache-2.0", "mit", "cc0-1.0", "cc-by-4.0", "odc-by"})
EVAL_ONLY_LICENSES = frozenset({"cc-by-nc-4.0"})  # mai nel training
ROLES = ("train", "benchmark")
GENERATOR_TERMS = ("SÌ", "NO", "INCERTO")
_SHA = re.compile(r"[0-9a-f]{40}")
_CARD_FIELDS = ("id", "read_on", "config", "license_source", "texts_by", "labels_by",
                "generator_terms_note", "title", "authors", "url", "changes")
_CARD_LISTS = ("splits", "languages", "surfaces", "generators", "filters", "risks")


class ManifestError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("manifest non ammesso:\n- " + "\n- ".join(problems))


@dataclass(frozen=True)
class Decision:
    """Decisione umana su termini INCERTO: la compila una persona, mai il codice."""

    by: str
    on: str
    rationale: str

    def complete(self) -> bool:
        return all(value.strip() for value in (self.by, self.on, self.rationale))


@dataclass(frozen=True)
class Entry:
    name: str  # chiave stabile nel report, es. `aegis2-train`
    id: str  # id del dataset sul Hub
    role: str
    revision: str  # commit di 40 caratteri, o NON VERIFICATO
    read_on: str
    config: str
    splits: tuple[str, ...]
    languages: tuple[str, ...]
    surfaces: tuple[str, ...]
    license: str
    license_source: str
    gated: bool
    texts_by: str
    labels_by: str
    generators: tuple[str, ...]
    generator_terms: str
    generator_terms_note: str
    filters: tuple[str, ...]
    title: str
    authors: str
    url: str
    changes: str
    risks: tuple[str, ...]
    text_fields: tuple[str, ...] = ()  # benchmark: colonne da deduplicare
    decision: Decision | None = None


def _entry(table: dict) -> Entry:
    known = {field.name for field in fields(Entry)}
    name = table.get("name", "?")
    if unknown := sorted(set(table) - known):
        raise ManifestError([f"{name}: campi sconosciuti {unknown}"])
    values = {key: tuple(value) if isinstance(value, list) else value
              for key, value in table.items()}
    if "decision" in values:
        values["decision"] = Decision(**values["decision"])
    try:
        return Entry(**values)
    except TypeError as error:
        raise ManifestError([f"{name}: {error}"]) from None


def load(path: Path | None = None) -> list[Entry]:
    source = Path(path) if path else MANIFEST
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    return [_entry(table) for table in data.get("dataset", [])]


def _problems(entry: Entry) -> list[str]:
    out = []
    if entry.role not in ROLES:
        out.append(f"ruolo {entry.role!r} non in {ROLES}")
    if not _SHA.fullmatch(entry.revision):
        out.append(f"revisione {entry.revision!r} non fissata a un commit")
    allowed = TRAIN_LICENSES | (EVAL_ONLY_LICENSES if entry.role == "benchmark" else set())
    if entry.license not in allowed:
        out.append(f"licenza {entry.license!r} non ammessa per il ruolo {entry.role}")
    empty = [key for key in _CARD_FIELDS if not str(getattr(entry, key)).strip()]
    empty += [key for key in _CARD_LISTS if not getattr(entry, key)]
    if empty:
        out.append(f"campi della data card vuoti: {empty}")
    if bad := [surface for surface in entry.surfaces if surface not in SURFACES]:
        out.append(f"superfici fuori dalla v1: {bad}")
    if entry.generator_terms not in GENERATOR_TERMS:
        out.append(f"termini dei generatori {entry.generator_terms!r} non in {GENERATOR_TERMS}")
    if entry.role == "benchmark":
        if not entry.text_fields:
            out.append("campi di testo da deduplicare non verificati")
        return out
    if entry.gated:
        out.append("fonte gated con termini propri")
    if entry.generator_terms == "NO":
        out.append("termini dei generatori restrittivi (NO)")
    if entry.generator_terms == "INCERTO" and not (entry.decision and entry.decision.complete()):
        out.append("termini dei generatori INCERTO senza decisione umana (chi, quando, perché)")
    return out


def problems(entries: list[Entry]) -> list[str]:
    """Tutti i motivi di blocco, una riga per problema con il nome della voce."""
    out = [f"{entry.name}: {problem}" for entry in entries for problem in _problems(entry)]
    names = [entry.name for entry in entries]
    out += [f"{name}: nome duplicato" for name in sorted({n for n in names if names.count(n) > 1})]
    trained = {(e.id, e.config, split) for e in entries if e.role == "train" for split in e.splits}
    out += [
        f"{entry.name}: split held-out {split!r} usato anche per il training"
        for entry in entries if entry.role == "benchmark" for split in entry.splits
        if (entry.id, entry.config, split) in trained
    ]
    return out


def gate(entries: list[Entry]) -> None:
    """Blocca la build se una voce non è ammessa."""
    if found := problems(entries):
        raise ManifestError(found)


def training(entries: list[Entry] | None = None) -> list[Entry]:
    return [entry for entry in (load() if entries is None else entries) if entry.role == "train"]


def benchmarks(entries: list[Entry] | None = None) -> list[Entry]:
    """Registro dei benchmark held-out: id, config, split, revisione, campi di testo."""
    return [e for e in (load() if entries is None else entries) if e.role == "benchmark"]
