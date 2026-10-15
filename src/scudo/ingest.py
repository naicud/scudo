"""Ingestione riproducibile dei dataset del manifest, fuori dal repository.

`python -m scudo.ingest check` elenca i motivi di blocco del cancello.
`python -m scudo.ingest build --out DIR` passa il cancello, scarica alle
revisioni fissate (cache in `DIR/hf-cache`), indicizza i testi dei benchmark
(campi di testo ed evidenze dei casi, se il benchmark ha un convertitore),
converte le fonti con un convertitore, conta gli scarti per motivo, marca le
label non gold (`needs_relabel`), rimuove i duplicati esatti e near-duplicate
dei benchmark e scrive in DIR i casi JSONL, `report.json` e le data card. DIR
deve stare fuori dal repository e da ogni work tree git: i dati restano nel
perimetro autorizzato. `cards --out DIR` scrive solo le data card dal manifest.
`verify --report DIR/report.json` controlla che la deduplica di quella build
abbia indicizzato ogni benchmark del registro ed esce con 1 altrimenti.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path

from scudo import (
    aegis,
    aegis1,
    aya,
    canttalk,
    datacard,
    gretel,
    in_the_wild,
    jackhhao,
    llmail,
    nemotron,
    open_jev,
    procedural,
)
from scudo.cases import Case
from scudo.dedup import BANDS, BINS, ROWS, SHINGLE, THRESHOLD, Index
from scudo.manifest import MANIFEST, Entry, ManifestError, benchmarks, gate, load, training

REPO_ROOT = Path(__file__).resolve().parents[2]
# Un convertitore espone needs_relabel e drop_reason con cases_from_row, oppure
# convert_row se scarta parti di riga (domande, distrattori, risposte).
CONVERTERS = {module.DATASET: module for module in (
    aegis, nemotron, aegis1, aya, gretel, llmail, in_the_wild, jackhhao,
    open_jev, canttalk, procedural,
)}
# Fonti di training senza una mappatura fedele al tri-state: il motivo va nel report.
UNCONVERTED = {
    "google/civil_comments": (
        "punteggi di tossicità (frazione di valutatori), non label di policy: domanda ai "
        "valutatori e soglie non documentate nel README, numero di valutatori assente"),
    "Lakera/mosscap_prompt_injection": (
        "nessuna label per riga: il README avverte che molti prompt non sono injection e "
        "l'unico esito (password rivelata) viene dalle risposte ChatGPT escluse"),
}
Loader = Callable[[Entry, str], Iterable[dict]]
DEDUP_PARAMS = {"normalize": "NFKC+casefold+punteggiatura", "exact": "sha256",
                "shingle": SHINGLE, "bins": BINS, "bands": BANDS, "rows": ROWS,
                "threshold": THRESHOLD}


def manifest_digest() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def _convert(converter, row: dict) -> tuple[list[Case], Counter]:
    """Casi di una riga e scarti per motivo."""
    if hasattr(converter, "convert_row"):
        return converter.convert_row(row)
    if reason := converter.drop_reason(row):
        return [], Counter({reason: 1})
    return converter.cases_from_row(row), Counter()


def check_outside_repo(out: Path) -> Path:
    """Cartella di uscita risolta; errore se è nel repository o in un work tree git."""
    path = Path(out).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"{path}: uscita dentro il repository {REPO_ROOT}")
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            raise ValueError(f"{path}: uscita dentro il work tree git {parent}")
    return path


def load_rows(entry: Entry, split: str, *, cache_dir: Path) -> Iterable[dict]:
    """Righe di uno split alla revisione fissata (extra `train`).

    I file grezzi vanno nella cache Hub (`HF_HUB_CACHE`, letta all'import di
    huggingface_hub), le tabelle Arrow nella cache di datasets: entrambe in
    `cache_dir`, altrimenti errore.
    """
    if "huggingface_hub" not in sys.modules:
        os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    from datasets import load_dataset
    from huggingface_hub import constants

    hub = Path(constants.HF_HUB_CACHE).resolve()
    if cache_dir.resolve() not in (hub, *hub.parents):
        raise ValueError(f"cache Hub {hub} fuori dal perimetro {cache_dir}")
    return load_dataset(entry.id, entry.config, split=split, revision=entry.revision,
                        cache_dir=str(cache_dir / "datasets"))


def _index(entries: list[Entry], loader: Loader) -> Index:
    """Testi dei benchmark; con un convertitore anche le evidenze nella forma del training."""
    index = Index()
    for bench in entries:
        converter = CONVERTERS.get(bench.id)
        for split in bench.splits:
            for row in loader(bench, split):
                texts = [row[field] for field in bench.text_fields
                         if isinstance(row.get(field), str) and row[field].strip()]
                if converter is not None:
                    texts += [case.evidence for case in _convert(converter, row)[0]]
                for text in dict.fromkeys(texts):
                    index.add(bench.name, text)
    return index


def _ingest(entry: Entry, converter, loader: Loader, index: Index, path: Path) -> dict:
    stats = {"rows": 0, "dropped": Counter(), "cases": 0, "kept": 0, "needs_relabel": 0,
             "removed_exact": Counter(), "removed_near": Counter()}
    with path.open("w", encoding="utf-8") as handle:
        for split in entry.splits:
            for row in loader(entry, split):
                stats["rows"] += 1
                cases, dropped = _convert(converter, row)
                stats["dropped"].update(dropped)
                for case in cases:
                    stats["cases"] += 1
                    if hit := index.match(case.evidence):
                        kind, bench = hit
                        stats[f"removed_{kind}"][bench] += 1
                        continue
                    relabel = converter.needs_relabel(row, case)
                    stats["kept"] += 1
                    stats["needs_relabel"] += relabel
                    record = {**asdict(case), "split": split, "needs_relabel": relabel}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"revision": entry.revision,
            **{key: dict(value) if isinstance(value, Counter) else value
               for key, value in stats.items()}}


def write_cards(out: Path, entries: list[Entry], stats: dict | None = None) -> None:
    """`stats` None: schede dal solo manifest; altrimenti attribuzioni solo per le fonti con dati."""
    cards = out / "data-cards"
    cards.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        card = datacard.render(entry, (stats or {}).get(entry.name))
        (cards / f"{entry.name}.md").write_text(card, encoding="utf-8")
    used = [e for e in training(entries) if stats is None or e.name in stats]
    (cards / "ATTRIBUTIONS.md").write_text(datacard.attributions(used), encoding="utf-8")


def build(out: Path, *, entries: list[Entry] | None = None, loader: Loader | None = None) -> dict:
    """Build completa; il cancello e il controllo della cartella vengono prima di ogni download."""
    digest = None if entries is not None else manifest_digest()
    entries = load() if entries is None else entries
    gate(entries)
    out = check_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    loader = loader or partial(load_rows, cache_dir=out / "hf-cache")
    index = _index(benchmarks(entries), loader)
    report = {
        "manifest_sha256": digest,
        "dedup": dict(DEDUP_PARAMS),
        "benchmarks": {bench.name: {"texts": index.sizes.get(bench.name, 0),
                                    "removed_exact": 0, "removed_near": 0}
                       for bench in benchmarks(entries)},
        "sources": {},
        "not_ingested": {},
    }
    for entry in training(entries):
        converter = CONVERTERS.get(entry.id)
        if converter is None:
            report["not_ingested"][entry.name] = UNCONVERTED.get(entry.id, "nessun convertitore")
            continue
        stats = _ingest(entry, converter, loader, index, out / f"{entry.name}.jsonl")
        for kind in ("exact", "near"):
            for bench, count in stats[f"removed_{kind}"].items():
                report["benchmarks"][bench][f"removed_{kind}"] += count
        report["sources"][entry.name] = stats
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    write_cards(out, entries, report["sources"])
    return report


def verify_dedup(report: dict, *, entries: list[Entry] | None = None,
                 manifest_sha256: str | None = None) -> dict:
    """PASS solo se il report viene da questo manifest, con questi parametri di
    deduplica, e ha indicizzato almeno un testo per ogni benchmark del registro.

    Senza `entries` vale il manifest del pacchetto con il suo digest.
    """
    if entries is None:
        entries, manifest_sha256 = load(), manifest_digest()
    problems = []
    reported = report.get("manifest_sha256")
    if reported is None or reported != manifest_sha256:
        problems.append(f"report del manifest {reported}, atteso {manifest_sha256}")
    if report.get("dedup") != DEDUP_PARAMS:
        problems.append(f"parametri di deduplica {report.get('dedup')} diversi da {DEDUP_PARAMS}")
    indexed = {name: stats.get("texts", 0)
               for name, stats in (report.get("benchmarks") or {}).items()}
    problems += [f"{bench.name}: non indicizzato dalla deduplica"
                 for bench in benchmarks(entries) if not indexed.get(bench.name)]
    return {"status": "FAIL" if problems else "PASS", "manifest_sha256": manifest_sha256,
            "report_manifest_sha256": reported, "indexed": indexed, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scudo.ingest",
                                     description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="cancello di licenza e provenienza sul manifest")
    for name in ("build", "cards"):
        commands.add_parser(name).add_argument("--out", required=True, type=Path)
    commands.add_parser("verify", help="deduplica di una build contro tutto il registro"
                        ).add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_dedup(json.loads(args.report.read_text(encoding="utf-8")))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return int(result["status"] != "PASS")
        if args.command == "check":
            gate(load())
            print("manifest ammesso")
        elif args.command == "build":
            report = build(args.out)
            for name, stats in report["sources"].items():
                print(f"{name}: {stats['kept']} casi tenuti su {stats['cases']}")
        else:
            write_cards(args.out, load())
    except (ManifestError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
