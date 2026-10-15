"""Righe di training dell'Accurate: ragionamenti tenuti solo se arrivano alla label.

Fonte: `nvidia/Nemotron-Content-Safety-Reasoning-Dataset` (CC-BY-4.0), split
`nemotron_content_safety_reasoning` = file `aegis_v2_efficient_reasoning.jsonl`,
righe di Aegis 2.0 con una traccia per generatore (DeepSeek-R1-0528,
gpt-oss-120b, Qwen3-32B). Si usano le tracce complete `safety_reasoning_*`:
quelle `efficient_reasoning_*` spesso non riportano il verdetto finale.

La traccia giudica insieme richiesta e risposta, mentre l'evidenza di un caso
`response` è la sola risposta: restano quindi solo le righe senza risposta,
superficie `prompt`. Il verdetto della traccia è la riga finale
`Prompt harm: harmful|unharmful` dopo `</think>`; una traccia illeggibile esce
con il suo motivo, una che non coincide con la label gold esce come
`disagree`. Ogni riga porta fonte, id e generatore della traccia. I file
`--extra` (ragionamenti del teacher, issue #7) hanno lo stesso formato di
`ReasoningRecord` e passano gli stessi controlli. Tutto si deduplica contro i
benchmark del manifest. `python -m scudo.reasoning_data build --out DIR`
scrive `reasoning.jsonl` e `reasoning-report.json` fuori dal repository.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

from scudo import aegis
from scudo.artifacts import file_sha256
from scudo.cases import Case
from scudo.dedup import Index
from scudo.ingest import DEDUP_PARAMS, benchmark_index, check_outside_repo, load_rows, manifest_digest
from scudo.manifest import Entry, ManifestError, benchmarks, gate, load, training
from scudo.protocol import SEMANTIC_IDS
from scudo.reasoned import REASONED_FORMAT_VERSION, check_reasoning, reasoned_training_row

DATASET = "nvidia/Nemotron-Content-Safety-Reasoning-Dataset"
REVISION = "792b0715f519c0750d63b73af2bf33ddd9ac3887"
SPLIT = "nemotron_content_safety_reasoning"
FILE = "aegis_v2_efficient_reasoning.jsonl"  # data_files dello split nel README
FILE_SHA256 = "d7e9e43ea92a42cc95a20a710bb45072fd8eec75a331c1393ce96029ddfc5d0c"  # oid LFS
GENERATORS = {  # suffisso delle colonne -> modello, dai link del README
    "deepseek_r1_0528": "deepseek-ai/DeepSeek-R1-0528",
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "qwen3_32b": "Qwen/Qwen3-32B",
}
SEED = 3407
_FOOTER = re.compile(r"\s*Prompt harm: (harmful|unharmful)\nResponse harm: (?:harmful|unharmful|none)\s*")
_HARM = {"harmful": "VIOLATION", "unharmful": "NO_VIOLATION"}


class TraceError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ReasoningRecord:
    """Caso con ragionamento e provenienza; è anche il formato dei file `--extra`."""

    source: str
    source_id: str
    language: str
    surface: str
    criterion: str
    evidence: str
    verdict: str  # gold
    reasoning: str
    trace_verdict: str  # verdetto a cui arriva il ragionamento
    generator: str  # modello che ha scritto il ragionamento

    def __post_init__(self):
        if not all(isinstance(getattr(self, f.name), str) for f in fields(self)):
            raise ValueError("tutti i campi sono stringhe")
        self.case()
        check_reasoning(self.reasoning)
        if self.trace_verdict not in SEMANTIC_IDS:
            raise ValueError(f"verdetto della traccia {self.trace_verdict!r} fuori dal tri-state")
        if not self.generator.strip():
            raise ValueError("provenienza: generatore del ragionamento mancante")

    def case(self) -> Case:
        return Case(self.source, self.source_id, self.language, self.surface,  # type: ignore[arg-type]
                    self.criterion, self.evidence, self.verdict)  # type: ignore[arg-type]


def parse_trace(text: str) -> tuple[str, str]:
    """(ragionamento, verdetto finale sul prompt) di una traccia `<think>…</think>` + esito."""
    head, sep, tail = text.lstrip().partition("</think>")
    if not sep or not head.startswith("<think>") or "<think>" in head[7:] or "</think>" in tail:
        raise TraceError("traccia senza <think>…</think>")
    match = _FOOTER.fullmatch(tail)
    if not match:
        raise TraceError("verdetto finale illeggibile")
    reasoning = head[len("<think>"):].strip()
    if not reasoning:
        raise TraceError("ragionamento vuoto")
    try:
        check_reasoning(reasoning)
    except ValueError:
        raise TraceError("ragionamento con marcatore riservato") from None
    return reasoning, _HARM[match.group(1)]


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    if reason := aegis.drop_reason(row):
        return reason
    response = row.get("response")
    if isinstance(response, str) and response not in ("", "None"):
        return "traccia congiunta su prompt e risposta"
    return None


def records_from_row(row: dict) -> tuple[list[ReasoningRecord], Counter]:
    """Un record per traccia leggibile, in disaccordo compreso; scarti contati per motivo."""
    (case,) = aegis.cases_from_row(row, source=DATASET, language="en")
    records, dropped = [], Counter()
    for key, model in GENERATORS.items():
        try:
            reasoning, verdict = parse_trace(row[f"safety_reasoning_{key}"])
        except TraceError as error:
            dropped[error.reason] += 1
            continue
        records.append(ReasoningRecord(**asdict(case), reasoning=reasoning,
                                       trace_verdict=verdict, generator=model))
    return records, dropped


def read_records(path: Path) -> Iterator[ReasoningRecord]:
    """Record di un file `--extra`; un record non valido ferma la build con file e riga."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield ReasoningRecord(**json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{Path(path).name}:{number}: {error}") from None


def load_traces(entry: Entry, *, cache_dir: Path) -> Iterator[dict]:
    """Righe del file dello split alla revisione fissata, dopo il digest (extra `train`)."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(entry.id, FILE, repo_type="dataset", revision=entry.revision,
                                cache_dir=str(cache_dir / "hub")))
    if (actual := file_sha256(path)) != FILE_SHA256:
        raise ValueError(f"{FILE}: digest {actual} diverso da {FILE_SHA256}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stats() -> dict:
    return {"disagree": 0, "removed_exact": Counter(), "removed_near": Counter(), "kept": 0,
            "by_generator": Counter(), "by_verdict": Counter()}


def _keep(records: Iterable[ReasoningRecord], index: Index, stats: dict, rng: random.Random,
          handle) -> None:
    """Accordo con la label, poi deduplica; le righe tenute vanno nel JSONL."""
    for item in records:
        if item.trace_verdict != item.verdict:
            stats["disagree"] += 1
            continue
        if hit := index.match(item.evidence):
            kind, bench = hit
            stats[f"removed_{kind}"][bench] += 1
            continue
        row = reasoned_training_row(item.criterion, item.evidence, item.verdict,  # type: ignore[arg-type]
                                    item.reasoning, rng)
        handle.write(json.dumps({**asdict(item), **row}, ensure_ascii=False) + "\n")
        stats["kept"] += 1
        stats["by_generator"][item.generator] += 1
        stats["by_verdict"][item.verdict] += 1


def _ncsr(rows: Iterable[dict], stats: dict) -> Iterator[ReasoningRecord]:
    for row in rows:
        stats["rows"] += 1
        if reason := drop_reason(row):
            stats["dropped"][reason] += 1
            continue
        records, dropped = records_from_row(row)
        stats["trace_dropped"].update(dropped)
        stats["traces"] += len(records)
        yield from records


def _plain(stats: dict) -> dict:
    return {key: dict(value) if isinstance(value, Counter) else value
            for key, value in stats.items()}


def build(out: Path, *, extra: Sequence[Path] = (), seed: int = SEED,
          entries: list[Entry] | None = None, loader=None,
          trace_loader: Callable[[Entry], Iterable[dict]] | None = None) -> dict:
    """Cancello, controllo dell'uscita e della revisione prima di ogni download."""
    digest = None if entries is not None else manifest_digest()
    entries = load() if entries is None else entries
    gate(entries)
    found = [entry for entry in training(entries) if entry.id == DATASET]
    if len(found) != 1:
        raise ValueError(f"{DATASET}: serve una sola voce di training nel manifest")
    entry = found[0]
    if entry.revision != REVISION or entry.splits != (SPLIT,):
        raise ValueError(f"{entry.name}: revisione o split diversi da quelli del file fissato")
    out = check_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    loader = loader or partial(load_rows, cache_dir=out / "hf-cache")
    trace_loader = trace_loader or partial(load_traces, cache_dir=out / "hf-cache")
    index = benchmark_index(benchmarks(entries), loader)
    rng = random.Random(seed)
    # Stessi campi di deduplica del report di ingestione: `ingest.verify_dedup` lo verifica.
    report = {"format": REASONED_FORMAT_VERSION, "seed": seed, "manifest_sha256": digest,
              "dedup": dict(DEDUP_PARAMS),
              "benchmarks": {bench.name: {"texts": index.sizes.get(bench.name, 0),
                                          "removed_exact": 0, "removed_near": 0}
                             for bench in benchmarks(entries)},
              "sources": {}}
    with (out / "reasoning.jsonl").open("w", encoding="utf-8") as handle:
        stats = {"revision": entry.revision, "file": FILE, "rows": 0, "dropped": Counter(),
                 "traces": 0, "trace_dropped": Counter(), **_stats()}
        _keep(_ncsr(trace_loader(entry), stats), index, stats, rng, handle)
        report["sources"][entry.name] = _plain(stats)
        for path in extra:
            records = list(read_records(path))
            stats = {"sha256": file_sha256(Path(path)), "records": len(records), **_stats()}
            _keep(records, index, stats, rng, handle)
            report["sources"][f"extra:{Path(path).name}"] = _plain(stats)
    for stats in report["sources"].values():
        for kind in ("exact", "near"):
            for bench, count in stats[f"removed_{kind}"].items():
                report["benchmarks"][bench][f"removed_{kind}"] += count
    (out / "reasoning-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scudo.reasoning_data",
                                     description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build")
    command.add_argument("--out", required=True, type=Path)
    command.add_argument("--extra", action="append", default=[], type=Path,
                         help="JSONL di ReasoningRecord (ragionamenti del teacher)")
    command.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        report = build(args.out, extra=args.extra, seed=args.seed)
    except (ManifestError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    for name, stats in report["sources"].items():
        print(f"{name}: {stats['kept']} righe tenute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
