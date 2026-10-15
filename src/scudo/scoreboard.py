"""Scoreboard versionato: metriche aggregate di ogni run di eval, per seguire il progresso.

Legge i report per benchmark di una cartella `eval/<scudo_sha>/` (una sottocartella per
modello, come li carica il notebook di eval) e aggiorna `docs/scoreboard.json`, da cui
genera `docs/scoreboard.md`. Entrano solo aggregati e identità del run (SHA, pesi con
revisione e hash, runtime, harness): mai casi, testi o risultati per caso.
"""

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOREBOARD_JSON = ROOT / "docs" / "scoreboard.json"
SCOREBOARD_MD = ROOT / "docs" / "scoreboard.md"
SKIPPED_FILES = {"results.json", "summary.json"}
IDENTITY = ("proof_type", "format", "harness", "runtime", "weights")


class ScoreboardError(ValueError):
    pass


def metrics(block: dict) -> dict:
    """Le metriche di un blocco `overall` o per lingua, nella forma dello scoreboard."""
    binary = block.get("binary") or {}
    abstention = binary.get("abstention_rate") or {}
    return {
        "cases": block["cases"],
        "errors": block["errors"],
        "unsupported": block.get("unsupported", 0),
        "accuracy": block["accuracy"],
        "binary_f1": binary.get("f1"),
        "false_positive_rate": binary.get("false_positive_rate"),
        "abstention_no_violation": abstention.get("NO_VIOLATION"),
        "ece": (block.get("calibration") or {}).get("ece"),
    }


def run_entry(model_dir: Path, on: str) -> dict:
    """Una voce per modello: identità condivisa dai report e metriche per benchmark e lingua."""
    reports = sorted(p for p in model_dir.glob("*.json") if p.name not in SKIPPED_FILES)
    if not reports:
        raise ScoreboardError(f"{model_dir}: nessun report di benchmark")
    entry, benchmarks = None, {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        identity = {key: report.get(key) for key in IDENTITY}
        head = {"scudo_sha": report["scudo_sha"], "model": report["model"], **identity,
                "dedup": (report.get("dedup_verification") or {}).get("status")}
        if entry is None:
            entry = {**head, "on": on}
        elif {k: entry[k] for k in head} != head:
            raise ScoreboardError(f"{path}: identità diversa dagli altri report di {model_dir}")
        rows = {"all": metrics(report["overall"])}
        if len(report["by_language"]) > 1:
            rows |= {lang: metrics(block) for lang, block in report["by_language"].items()}
        benchmarks[report["split"]] = rows
    return {**entry, "benchmarks": benchmarks}


def collect(eval_dir: Path, on: str) -> list[dict]:
    dirs = sorted(p for p in Path(eval_dir).iterdir() if p.is_dir())
    entries = [run_entry(p, on) for p in dirs]
    if not entries:
        raise ScoreboardError(f"{eval_dir}: nessuna cartella di modello")
    return entries


def merge(existing: list[dict], new: list[dict]) -> list[dict]:
    """Una voce per (scudo_sha, modello): un run ripetuto sostituisce il precedente."""
    by_key = {(e["scudo_sha"], e["model"]): e for e in existing}
    by_key |= {(e["scudo_sha"], e["model"]): e for e in new}
    return sorted(by_key.values(), key=lambda e: (e["on"], e["scudo_sha"], e["model"]))


def _cell(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def render(entries: list[dict]) -> str:
    lines = ["# Scoreboard", "",
             "Generato da `python -m scudo.scoreboard` a partire da `docs/scoreboard.json`:",
             "non modificarlo a mano. Metriche definite in [eval.md](eval.md); una metrica vale",
             "solo con revisione, hash dei pesi e harness, registrati nel JSON per ogni run.", ""]
    runs: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        runs.setdefault((entry["on"], entry["scudo_sha"]), []).append(entry)
    for (on, sha), group in sorted(runs.items(), reverse=True):
        dedup = sorted({str(e["dedup"]) for e in group})
        lines += [f"## {on} · scudo `{sha[:7]}`", "",
                  f"Deduplica dei benchmark nei report: {', '.join(dedup)}.", "",
                  "Per cella: F1 binario / falsi positivi sui benigni / ECE "
                  "(errori e astensioni contano contro).", ""]
        benchmarks = sorted({(b, lang) for e in group for b, rows in e["benchmarks"].items()
                             for lang in rows})
        lines += ["| benchmark | lingua | " + " | ".join(e["model"] for e in group) + " |",
                  "|---|---|" + "---|" * len(group)]
        for bench, lang in benchmarks:
            cells = []
            for e in group:
                m = e["benchmarks"].get(bench, {}).get(lang)
                cells.append("non valutato" if m is None else
                             f"{_cell(m['binary_f1'])} / {_cell(m['false_positive_rate'])} / "
                             f"{_cell(m['ece'])}")
            lines.append(f"| {bench} | {lang} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def update(eval_dir: Path, on: str, json_path: Path = SCOREBOARD_JSON,
           md_path: Path = SCOREBOARD_MD) -> list[dict]:
    existing = (json.loads(json_path.read_text(encoding="utf-8"))["runs"]
                if json_path.exists() else [])
    entries = merge(existing, collect(eval_dir, on))
    json_path.write_text(json.dumps({"runs": entries}, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    md_path.write_text(render(entries), encoding="utf-8")
    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scudo.scoreboard",
                                     description="Aggiorna lo scoreboard con un run di eval.")
    parser.add_argument("eval_dir", help="cartella eval/<scudo_sha>/ scaricata da HF")
    parser.add_argument("--on", default=date.today().isoformat(), help="data del run (ISO)")
    args = parser.parse_args(argv)
    entries = update(Path(args.eval_dir), date.fromisoformat(args.on).isoformat())
    print(f"{SCOREBOARD_JSON.relative_to(ROOT)}: {len(entries)} voci")


if __name__ == "__main__":
    main()
