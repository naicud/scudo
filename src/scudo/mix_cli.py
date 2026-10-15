"""Comandi del mix del Fast v0.1 (`python -m scudo.mix_cli ...`), cartelle fuori dal repository.

- `judge-set --ingest DIR --out JDIR`: candidati puliti (duplicati e conflitti
  esclusi) più gli INSUFFICIENT generati, nel formato del campione del pilot,
  da far giudicare ai teacher per rietichettatura e difficoltà;
- `judge --out JDIR --teachers T --ledger L --confirm-spend X`: run dei teacher in
  volume; senza la conferma umana del volume (`scudo.budget`), con un teacher
  fuori dall'insieme dell'ultima conferma o con una stima (somma sui teacher)
  oltre il residuo del volume confermato non parte nessuna chiamata;
- `build --ingest DIR --outcomes F... --out MDIR`: mix e manifest del mix;
- `upload --out MDIR --repo R`: carica mix, manifest e report di ingestione su un
  repository dataset privato e stampa revisione e digest da dare al notebook.
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from scudo import mix_clean, pilot
from scudo.artifacts import file_sha256
from scudo.budget import BudgetError, Ledger, require_volume_confirmation
from scudo.difficulty import MARGIN_THRESHOLD, read_outcomes
from scudo.ingest import REPO_ROOT, check_outside_repo
from scudo.mix import DEFAULT_CATEGORIES, DEFAULT_LANGUAGES, MixError, Targets, build
from scudo.mix_pool import PoolError, load_ingest, load_italian
from scudo.teacher import ConfigError, load_configs

MIX_FILE, MANIFEST_FILE, INGEST_REPORT = "mix.jsonl", "mix-manifest.json", "ingest-report.json"


def repo_sha(root: Path = REPO_ROOT) -> str:
    """SHA del commit; un albero con modifiche non riproduce il mix e viene rifiutato."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                              text=True).stdout.strip()

    if git("status", "--porcelain"):
        raise MixError("albero git con modifiche: il mix non sarebbe riproducibile dallo SHA")
    return git("rev-parse", "HEAD")


def candidates(ingest: Path, italian: list[Path]) -> tuple[list, dict]:
    cases, info = load_ingest(ingest)
    inputs = {"ingest": info, "italian": []}
    for batch in italian:
        extra, provenance = load_italian(batch)
        cases += extra
        inputs["italian"].append(provenance)
    return cases, inputs


def judge_set(ingest: Path, italian: list[Path], out: Path) -> dict:
    """Campione da giudicare: ogni candidato pulito e ogni INSUFFICIENT generato."""
    cases, inputs = candidates(ingest, italian)
    clean = mix_clean.dedup(cases)[0]
    pooled = clean + mix_clean.generated(clean)[0]
    out = check_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sample.jsonl").open("w", encoding="utf-8") as handle:
        for p in pooled:
            record = {"set": "volume", "key": p.key, "entry": p.entry, "role": "train",
                      "training_eligible": True, "needs_relabel": p.needs_relabel, **asdict(p.case)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {"kind": "volume", "cases": len(pooled), "inputs": inputs,
            "needs_relabel": sum(p.needs_relabel for p in pooled)}
    (out / "sample-meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return meta


def judge(out: Path, teachers: Path, ledger_path: Path, confirm: str | None) -> dict:
    """Run in volume dei teacher: mai come pilot; con la conferma, `pilot.run` applica
    `budget.check_volume` (teacher e residuo dell'ultima conferma, tetto)."""
    ledger = Ledger(ledger_path)
    require_volume_confirmation(ledger)
    return pilot.run(out, load_configs(teachers), ledger, confirm=confirm)


def build_mix(ingest: Path, italian: list[Path], outcomes: list[Path], out: Path, *,
              targets: Targets, seed: int, override_reason: str | None, sha: str,
              margin_threshold: float = MARGIN_THRESHOLD) -> dict:
    """Scrive mix e manifest del mix; un override dei target senza motivo è rifiutato."""
    if targets.overrides() and not (override_reason or "").strip():
        raise MixError(f"target diversi dai default {sorted(targets.overrides())}: "
                       "serve --override-reason")
    cases, inputs = candidates(ingest, italian)
    rows, stats = build(cases, outcomes=read_outcomes(outcomes), targets=targets, seed=seed,
                        margin_threshold=margin_threshold)
    out = check_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / MIX_FILE).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copyfile(Path(ingest) / "report.json", out / INGEST_REPORT)
    inputs["outcomes"] = [{"name": Path(p).name, "sha256": file_sha256(Path(p))} for p in outcomes]
    manifest = {"repo_sha": sha, **stats, "override_reason": override_reason, "inputs": inputs,
                "output": {"filename": MIX_FILE, "rows": len(rows),
                           "sha256": file_sha256(out / MIX_FILE)},
                "ingest_report": {"filename": INGEST_REPORT,
                                  "sha256": file_sha256(out / INGEST_REPORT)}}
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    return manifest


def upload(out: Path, repo: str) -> dict:
    """Carica sul repository dataset privato; rifiuta un repository pubblico o file alterati."""
    from huggingface_hub import HfApi

    out = check_outside_repo(out)
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    for name, expected in ((MIX_FILE, manifest["output"]["sha256"]),
                           (INGEST_REPORT, manifest["ingest_report"]["sha256"])):
        if file_sha256(out / name) != expected:
            raise MixError(f"{name}: digest diverso dal manifest del mix")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise MixError(f"{repo} non è privato: i dati non escono dal perimetro")
    commit = api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="dataset",
                               allow_patterns=[MIX_FILE, MANIFEST_FILE, INGEST_REPORT],
                               commit_message=f"mix Fast v0.1 da scudo {manifest['repo_sha']}")
    return {"repo": repo, "revision": commit.oid,
            "manifest_sha256": file_sha256(out / MANIFEST_FILE)}


def _weights(text: str | None, default: dict) -> dict:
    if text is None:
        return dict(default)
    pairs = [item.split("=", 1) for item in text.split(",") if item.strip()]
    return {key.strip(): float(value) for key, value in pairs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scudo.mix_cli",
                                     description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("judge-set", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--ingest", required=True, type=Path)
        sub.add_argument("--italian", action="append", default=[], type=Path,
                         help="lotto di #7 con spot-check accettato")
        sub.add_argument("--out", required=True, type=Path)
    run = commands.add_parser("judge")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--teachers", required=True, type=Path)
    run.add_argument("--ledger", required=True, type=Path)
    run.add_argument("--confirm-spend", default=None)
    mix = commands.choices["build"]
    mix.add_argument("--outcomes", action="append", required=True, type=Path)
    mix.add_argument("--seed", type=int, default=3407)
    mix.add_argument("--total", type=int, default=Targets().total)
    mix.add_argument("--categories",
                     help="es. prompt_response=45,injection=20,custom=15,over_refusal=0")
    mix.add_argument("--languages", help="es. it=0,en=1,mixed=0")
    mix.add_argument("--insufficient-share", type=float, default=Targets().insufficient_share)
    mix.add_argument("--method-cap", type=float, default=Targets().method_cap)
    mix.add_argument("--override-reason", default=None)
    mix.add_argument("--margin-threshold", type=float, default=MARGIN_THRESHOLD,
                     help="margine del teacher (nat) sotto cui un caso concorde è HARD")
    up = commands.add_parser("upload")
    up.add_argument("--out", required=True, type=Path)
    up.add_argument("--repo", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "judge-set":
            result = judge_set(args.ingest, args.italian, args.out)
        elif args.command == "judge":
            result = judge(args.out, args.teachers, args.ledger, args.confirm_spend)
        elif args.command == "build":
            targets = Targets(total=args.total,
                              categories=_weights(args.categories, DEFAULT_CATEGORIES),
                              languages=_weights(args.languages, DEFAULT_LANGUAGES),
                              insufficient_share=args.insufficient_share,
                              method_cap=args.method_cap)
            manifest = build_mix(args.ingest, args.italian, args.outcomes, args.out,
                                 targets=targets, seed=args.seed,
                                 override_reason=args.override_reason, sha=repo_sha(),
                                 margin_threshold=args.margin_threshold)
            result = {"output": manifest["output"], "overrides": manifest["overrides"],
                      "composition": manifest["composition"]}
        else:
            result = upload(args.out, args.repo)
    except (BudgetError, ConfigError, MixError, PoolError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
