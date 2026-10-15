"""Casi candidati al mix: uscite di `scudo.ingest build` e lotti italiani di #7.

Dalla build si leggono solo le fonti del suo `report.json`, e solo se il report
viene dal manifest attuale (stesso digest: la build è passata dal cancello su
quel manifest) e ogni fonte è ancora ammessa dal cancello. Un lotto italiano
entra solo con lo spot-check umano accettato (`batch_pool`). Ogni suo testo è
scritto da un teacher: il modello in provenienza (generatore o traduttore) deve
essere del registro `scudo.teacher.CANDIDATES` e avere la propria voce di
training, alla stessa revisione, ammessa dal cancello (`teacher_problem`). Un
caso tradotto resta della voce della sua fonte, che deve essere ammessa; un caso
generato è della voce del teacher che lo ha scritto. Origine sconosciuta, fonte
diversa dalla voce o caso invalido sono errori, mai righe saltate.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from scudo import generate_it, spotcheck_it
from scudo.artifacts import file_sha256
from scudo.cases import Case
from scudo.ingest import manifest_digest
from scudo.italian_run import POOL, read_pool
from scudo.manifest import Entry, load, problems, training
from scudo.teacher import CANDIDATES

_CASE_FIELDS = tuple(f.name for f in fields(Case))


class PoolError(ValueError):
    """Ingresso del mix non ammesso: nessun caso entra."""


@dataclass(frozen=True)
class PoolCase:
    case: Case
    entry: str  # voce del manifest: della fonte, o del teacher per i casi generati
    needs_relabel: bool

    @property
    def key(self) -> str:
        """Chiave degli esiti dei teacher, come nel campione del pilot."""
        return f"{self.entry}:{self.case.source_id}:{self.case.surface}"


def _cases(path: Path, entry: str, sources: set[str]) -> list[PoolCase]:
    out, keys = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            where = f"{path.name}:{number}"
            if not isinstance(record.get("needs_relabel"), bool):
                raise PoolError(f"{where}: needs_relabel mancante o non booleano")
            if missing := [name for name in _CASE_FIELDS if name not in record]:
                raise PoolError(f"{where}: campi del caso mancanti {missing}")
            try:
                case = Case(**{name: record[name] for name in _CASE_FIELDS})
            except ValueError as error:
                raise PoolError(f"{where}: {error}") from None
            if case.source not in sources:
                raise PoolError(f"{where}: fonte {case.source!r} fuori da {sorted(sources)}")
            pooled = PoolCase(case, entry, record["needs_relabel"])
            if pooled.key in keys:
                raise PoolError(f"{where}: chiave ripetuta {pooled.key}")
            keys.add(pooled.key)
            out.append(pooled)
    return out


def _admitted(entries: list[Entry], name: str) -> Entry:
    by_name = {entry.name: entry for entry in entries}
    if name not in by_name or by_name[name].role != "train":
        raise PoolError(f"{name}: non è una voce di training del manifest")
    if found := problems([by_name[name]]):
        raise PoolError(f"{name}: bloccata dal cancello: {found}")
    return by_name[name]


# Provider i cui termini lasciano usare gli output per addestrare e pubblicare un
# modello (docs/research/teacher-providers-2026-10-15.md); gli altri non sono verificati.
VERIFIED_PROVIDERS = frozenset({"deepinfra", "together", "scaleway"})


def serving_provider(provider: str, provider_model: str = "") -> str:
    """Chi ha servito il modello: il suffisso `:provider` del router HF, altrimenti `provider`."""
    if ":" in provider_model:
        return provider_model.rsplit(":", 1)[1].lower()
    return provider.lower()


def provider_problem(writer: dict) -> str | None:
    provider = serving_provider(str(writer.get("provider") or ""),
                                str(writer.get("provider_model") or ""))
    return None if provider in VERIFIED_PROVIDERS else "provider_not_verified"


def teacher_problem(entries: list[Entry], model_id) -> str | None:
    """Perché i testi di `model_id` non entrano, o None: teacher del registro con voce ammessa."""
    if not isinstance(model_id, str) or model_id not in CANDIDATES:
        return "generator_not_teacher"
    found = [entry for entry in training(entries) if entry.id == model_id]
    if len(found) != 1:
        return "teacher_not_in_manifest"
    if found[0].revision != CANDIDATES[model_id].revision:
        return "teacher_revision_differs"
    return "teacher_blocked" if problems(found) else None


def _teacher(entries: list[Entry], record: dict) -> Entry:
    """Voce del teacher che ha scritto il testo del caso: generatore o traduttore."""
    writer = record["provenance"].get(
        "translator" if record["origin"] == "translated" else "generator")
    model_id = writer.get("model_id") if isinstance(writer, dict) else None
    problem = teacher_problem(entries, model_id) or provider_problem(
        writer if isinstance(writer, dict) else {})
    if problem:
        raise PoolError(f"{record['key']}: testo di {model_id!r}: {problem}")
    return next(entry for entry in training(entries) if entry.id == model_id)


def batch_pool(batch: Path) -> list[dict]:
    """Pool deduplicato del lotto italiano, solo con spot-check umano accettato."""
    spotcheck_it.require_accepted(batch)
    return [{"key": r["key"], "origin": r["origin"],
             **asdict(Case(**{name: r[name] for name in _CASE_FIELDS})),
             "provenance": r["provenance"]} for r in read_pool(batch)]


def load_ingest(out: Path, *, entries: list[Entry] | None = None,
                manifest_sha256: str | None = None) -> tuple[list[PoolCase], dict]:
    """Casi della build e provenienza: digest del report, del manifest e di ogni JSONL."""
    if entries is None:
        entries, manifest_sha256 = load(), manifest_digest()
    report_path = Path(out) / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") is None or report["manifest_sha256"] != manifest_sha256:
        raise PoolError(f"report della build dal manifest {report.get('manifest_sha256')}, "
                        f"attuale {manifest_sha256}: rifare la build")
    cases, sources = [], {}
    for name, stats in report["sources"].items():
        entry = _admitted(entries, name)
        path = Path(out) / f"{name}.jsonl"
        found = _cases(path, name, {entry.id})
        if len(found) != stats["kept"]:
            raise PoolError(f"{name}: {len(found)} casi, il report ne dichiara {stats['kept']}")
        cases += found
        sources[name] = {"cases": len(found), "sha256": file_sha256(path)}
    return cases, {"manifest_sha256": manifest_sha256,
                   "report_sha256": file_sha256(report_path), "sources": sources}


def load_italian(batch: Path, *, entries: list[Entry] | None = None
                 ) -> tuple[list[PoolCase], dict]:
    """Pool di un lotto italiano con spot-check accettato; restituisce casi e provenienza."""
    entries = load() if entries is None else entries
    records = batch_pool(batch)  # SpotCheckError senza esito accettato per questo pool
    out, keys = [], set()
    for record in records:
        case = Case(**{name: record[name] for name in _CASE_FIELDS})
        if record["origin"] == "translated":
            entry = _admitted(entries, record["provenance"]["entry"])
            if case.source != entry.id:
                raise PoolError(f"{record['key']}: fonte {case.source!r} diversa da {entry.id!r}")
            _teacher(entries, record)
        elif record["origin"] == "generated" and case.source == generate_it.SOURCE:
            entry = _teacher(entries, record)
        else:
            raise PoolError(f"{record['key']}: origine {record['origin']!r} sconosciuta")
        pooled = PoolCase(case, entry.name, False)
        if pooled.key in keys:
            raise PoolError(f"{record['key']}: chiave ripetuta {pooled.key}")
        keys.add(pooled.key)
        out.append(pooled)
    accepted = spotcheck_it.require_accepted(batch)
    return out, {"batch": Path(batch).name, "pool_sha256": file_sha256(Path(batch) / POOL),
                 "spotcheck_pool_sha256": accepted["pool_sha256"], "cases": len(out),
                 "by_origin_language": dict(sorted(Counter(
                     f"{r['origin']}/{r['language']}" for r in records).items()))}
