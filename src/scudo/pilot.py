"""Pilot dei teacher (issue #5): campione gold, stima, run a spesa autorizzata, report.

Comandi (`python -m scudo.pilot ...`, cartelle e file sempre fuori dal repository):

- `sample --out DIR`: campione deterministico di casi con label umana, stratificato
  per lingua, superficie e verdetto, più un campione di righe `llm_jury` per
  validare la rietichettatura (scarica alle revisioni del manifest, extra `train`);
- `estimate`: prova a secco, nessuna chiamata; stampa la stima e il valore esatto
  da passare a `--confirm-spend`;
- `run --confirm-spend X`: chiama i teacher solo se X coincide con la stima e la
  stima sta nel residuo del tetto v0.1 (e, dopo una conferma del volume, nel suo
  residuo e con il suo teacher: `budget.check_volume`); si ferma prima di
  superare X; riprende saltando le coppie (teacher, caso) già fatte;
- `report`: metriche, stima del volume v0.1, export per la revisione umana;
- `confirm-volume`: la persona registra i teacher scelti (`--teacher` ripetibile)
  e un solo importo in volume per tutti.

Campione in `scudo.pilot_sample`, client in `scudo.teacher_client`, tetto e
registro in `scudo.budget`, report in `scudo.pilot_report`. Guida:
`docs/teacher-pilot.md`.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path

from scudo import budget
from scudo.budget import BudgetError, Guard, Ledger
from scudo.ingest import check_outside_repo, load_rows
from scudo.manifest import MANIFEST, load
from scudo.pilot_sample import (
    DEFAULT_SOURCES,
    PILOT_CASES,
    RELABEL_CASES,
    build_sample,
    write_sample,
)
from scudo.teacher import ConfigError, TeacherConfig, load_configs, messages
from scudo.teacher_client import Client

MAX_PILOT_CASES = 1000  # oltre, è volume: serve la conferma umana


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_sample(out: Path) -> list[dict]:
    records = _jsonl(check_outside_repo(out) / "sample.jsonl")
    if not records:
        raise ValueError(f"{out}: sample.jsonl assente o vuoto, eseguire prima `sample`")
    return records


def read_meta(out: Path) -> dict:
    return json.loads((check_outside_repo(out) / "sample-meta.json").read_text(encoding="utf-8"))


def read_outcomes(out: Path) -> list[dict]:
    return _jsonl(check_outside_repo(out) / "outcomes.jsonl")


def sample_sha(out: Path) -> str:
    return hashlib.sha256((check_outside_repo(out) / "sample.jsonl").read_bytes()).hexdigest()


def _todo(out: Path, configs: list[TeacherConfig]) -> list[tuple[TeacherConfig, dict, list]]:
    done = {(o["teacher"], o["key"]) for o in read_outcomes(out)}
    return [(cfg, record, messages(record["criterion"], record["evidence"]))
            for cfg in configs for record in read_sample(out)
            if (cfg.model_id, record["key"]) not in done]


def estimate(out: Path, configs: list[TeacherConfig], ledger: Ledger) -> dict:
    """Prova a secco: caso peggiore delle chiamate mancanti, nessuna rete, nessuna chiave."""
    per_teacher = Counter()
    todo = _todo(out, configs)
    for cfg, _, chat in todo:
        per_teacher[cfg.model_id] += budget.worst_case_usd(cfg, chat)
    total = sum(per_teacher.values())
    return {"items": len(todo), "teachers": {c.model_id: per_teacher[c.model_id] for c in configs},
            "total_usd": total, "confirm": f"{budget.cents_up(total):.2f}",
            "spent_usd": ledger.spent(), "remaining_usd": ledger.remaining(),
            "method": f"token stimati come caratteri/{budget.CHARS_PER_TOKEN} sull'ingresso "
                      "più max_tokens in uscita per chiamata (caso peggiore, stima)"}


def _check_scope(out: Path, configs: list[TeacherConfig], ledger: Ledger, planned: float) -> None:
    """Con una conferma del volume la run è in volume e passa da `budget.check_volume`;
    senza, è il pilot: oltre la sua taglia, o dopo il suo report, serve la conferma."""
    if ledger.of_kind("volume_confirmation"):
        budget.check_volume(ledger, [cfg.model_id for cfg in configs], planned)
        return
    if len(read_sample(out)) > MAX_PILOT_CASES:
        raise BudgetError(f"oltre {MAX_PILOT_CASES} casi è volume: manca la conferma umana")
    reported = {r["sample_sha256"] for r in ledger.of_kind("pilot_report")}
    if reported and sample_sha(out) not in reported:
        raise BudgetError("pilot già concluso: un nuovo campione richiede la conferma umana "
                          "del volume (confirm-volume)")


def run(out: Path, configs: list[TeacherConfig], ledger: Ledger, *, confirm: str | None,
        transport=None, environ=None, sleep=None) -> dict:
    """Chiamate mancanti con stop rigido; ogni esito va in outcomes.jsonl, ogni costo nel registro."""
    todo = _todo(out, configs)
    planned = math.fsum(budget.worst_case_usd(cfg, chat) for cfg, _, chat in todo)
    _check_scope(out, configs, ledger, planned)
    options = {k: v for k, v in (("transport", transport), ("environ", environ),
                                 ("sleep", sleep)) if v is not None}
    clients = {cfg.model_id: Client(cfg, **options) for cfg in configs}  # chiavi prima di spendere
    run_id = sample_sha(out)[:16]
    authorised = budget.authorise(ledger, run_id, planned, confirm)
    guard = Guard(ledger, run_id, authorised)
    done, stopped = 0, None
    with (out / "outcomes.jsonl").open("a", encoding="utf-8") as handle:
        for cfg, record, chat in todo:
            try:
                outcome, cost, usage_missing = budget.paid_call(
                    guard, cfg, record["key"], chat, partial(clients[cfg.model_id].judge, chat))
            except BudgetError as error:  # stop rigido o chiave rifiutata
                stopped = str(error)
                break
            handle.write(json.dumps({"teacher": cfg.model_id, "key": record["key"],
                                     "set": record["set"], **asdict(outcome), "cost_usd": cost,
                                     "usage_missing": usage_missing}, ensure_ascii=False) + "\n")
            handle.flush()
            done += 1
    return {"done": done, "planned": len(todo), "authorised_usd": authorised,
            "spent_usd": guard.spent, "stopped": stopped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scudo.pilot",
                                     description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="campione gold e llm_jury (extra train)")
    sample.add_argument("--out", required=True, type=Path)
    sample.add_argument("--n", type=int, default=PILOT_CASES)
    sample.add_argument("--relabel", type=int, default=RELABEL_CASES)
    sample.add_argument("--seed", type=int, default=0)
    for name in ("estimate", "run", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", required=True, type=Path)
        sub.add_argument("--teachers", required=True, type=Path, help="TOML fuori dal repository")
        sub.add_argument("--ledger", required=True, type=Path, help="registro JSONL della spesa")
    commands.choices["run"].add_argument("--confirm-spend", default=None)
    from scudo import pilot_report

    pilot_report.add_arguments(commands.choices["report"])
    volume = commands.add_parser("confirm-volume", help="solo la persona: teacher e spesa in volume")
    volume.add_argument("--ledger", required=True, type=Path)
    volume.add_argument("--teacher", required=True, action="append",
                        help="ripetibile: tutti i teacher coperti dall'importo")
    for name in ("--by", "--rationale"):
        volume.add_argument(name, required=True)
    volume.add_argument("--amount", required=True, type=float, help="totale per tutti i teacher")
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            out = check_outside_repo(args.out)
            built = build_sample(load(), partial(load_rows, cache_dir=out / "hf-cache"),
                                 sources=DEFAULT_SOURCES, n=args.n, relabel_n=args.relabel,
                                 seed=args.seed)
            built["meta"]["manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
            write_sample(out, built)
            print(json.dumps(built["meta"], indent=2, ensure_ascii=False))
        elif args.command == "confirm-volume":
            budget.confirm_volume(Ledger(args.ledger), teachers=args.teacher,
                                  amount_usd=args.amount, by=args.by, rationale=args.rationale)
            print("conferma del volume registrata")
        else:
            configs, ledger = load_configs(args.teachers), Ledger(args.ledger)
            if args.command == "estimate":
                print(json.dumps(estimate(args.out, configs, ledger), indent=2))
            elif args.command == "run":
                print(json.dumps(run(args.out, configs, ledger, confirm=args.confirm_spend), indent=2))
            else:
                report = pilot_report.write_report(args.out, configs, ledger,
                                                   **pilot_report.options(args))
                print(f"report in {args.out}: totale volume v0.1 per teacher " + json.dumps(
                    {k: round(v["total_usd"], 2) for k, v in report["volume"]["teachers"].items()
                     if v["total_usd"] is not None}))
    except (BudgetError, ConfigError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
