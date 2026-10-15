"""Dati italiani (issue #7): traduzione, generazione nativa e mista, spot-check.

Comandi (`python -m scudo.italian ...`, cartelle e file sempre fuori dal repository):

- `plan --out DIR --cases FILE...`: casi EN da tradurre e richieste di
  generazione IT e miste, deterministici; nessuna chiamata;
- `estimate`: prova a secco, stampa la stima e il valore da passare a
  `--confirm-spend`;
- `run --confirm-spend X`: solo dopo la conferma umana del volume con lo stesso
  teacher (`python -m scudo.pilot confirm-volume`), con X uguale alla stima e
  nel residuo; si ferma prima di superare X; riprende saltando gli item fatti;
- `dedup`: deduplica i casi accettati contro i benchmark (extra `train`);
- `spotcheck-export` / `spotcheck-import`: campione per il madrelingua ed esito.

Un lotto accettato entra nel mix del Fast con `python -m scudo.mix_cli --italian`
(`scudo.mix_pool`). Moduli: `scudo.translate`, `scudo.generate_it`,
`scudo.italian_run`, `scudo.spotcheck_it`. Guida: `docs/italian-data.md`.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from scudo import italian_run, spotcheck_it
from scudo.budget import BudgetError, Ledger
from scudo.ingest import check_outside_repo, load_rows
from scudo.manifest import load
from scudo.teacher import ConfigError, TeacherConfig, load_configs
from scudo.translate import BACK_TRANSLATION_THRESHOLD


def _read_cases(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        with check_outside_repo(path).open(encoding="utf-8") as handle:
            out += [json.loads(line) for line in handle if line.strip()]
    return out


def _teacher(path: Path, model_id: str) -> TeacherConfig:
    configs = {cfg.model_id: cfg for cfg in load_configs(path)}
    if model_id not in configs:
        raise ConfigError([f"teacher {model_id!r} non configurato in {path}"])
    return configs[model_id]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scudo.italian",
                                     description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="casi da tradurre e richieste di generazione")
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--cases", required=True, type=Path, nargs="+",
                      help="JSONL di casi da `scudo.ingest build`")
    for name in ("--translate", "--native", "--mixed"):
        plan.add_argument(name, required=True, type=int)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--bt-threshold", type=float, default=BACK_TRANSLATION_THRESHOLD)
    for name in ("estimate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", required=True, type=Path)
        sub.add_argument("--teachers", required=True, type=Path, help="TOML fuori dal repository")
        sub.add_argument("--teacher", required=True, help="model_id del teacher confermato")
        sub.add_argument("--ledger", required=True, type=Path, help="registro JSONL della spesa")
    commands.choices["run"].add_argument("--confirm-spend", default=None)
    commands.add_parser("dedup", help="deduplica contro i benchmark (extra train)").add_argument(
        "--out", required=True, type=Path)
    export = commands.add_parser("spotcheck-export")
    export.add_argument("--out", required=True, type=Path)
    export.add_argument("--n", type=int, default=spotcheck_it.SAMPLE_SIZE)
    export.add_argument("--seed", type=int, default=0)
    review = commands.add_parser("spotcheck-import", help="solo con il foglio compilato")
    review.add_argument("--out", required=True, type=Path)
    review.add_argument("--review", required=True, type=Path)
    review.add_argument("--by", required=True)
    review.add_argument("--threshold", type=float, default=spotcheck_it.THRESHOLD,
                        help="si alza, mai sotto 0,95")
    return parser


def _dump(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = italian_run.make_plan(
                _read_cases(args.cases), load(), translate_n=args.translate,
                native_n=args.native, mixed_n=args.mixed, seed=args.seed,
                threshold=args.bt_threshold)
            italian_run.write_plan(args.out, plan)
            _dump(plan["meta"])
        elif args.command in ("estimate", "run"):
            cfg, ledger = _teacher(args.teachers, args.teacher), Ledger(args.ledger)
            if args.command == "estimate":
                _dump(italian_run.estimate(args.out, cfg, ledger))
            else:
                _dump(italian_run.run(args.out, cfg, ledger, confirm=args.confirm_spend))
        elif args.command == "dedup":
            out = check_outside_repo(args.out)
            _dump(italian_run.dedup(out, load(), partial(load_rows, cache_dir=out / "hf-cache")))
        elif args.command == "spotcheck-export":
            print(spotcheck_it.export(args.out, n=args.n, seed=args.seed))
        else:
            outcome = spotcheck_it.import_review(args.out, args.review, by=args.by,
                                                 threshold=args.threshold)
            _dump({k: v for k, v in outcome.items() if k != "incorrect_items"})
    except (BudgetError, ConfigError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
