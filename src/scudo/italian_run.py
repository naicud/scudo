"""Lotto di dati italiani: piano, stima, run a spesa confermata e deduplica (issue #7).

Tutto vive in una cartella fuori dal repository:

- `plan.jsonl`, `plan-meta.json`: casi EN da tradurre (scelti per hash seminato fra
  quelli ammessi da `scudo.translate.eligible`) e richieste di generazione IT e
  miste (`scudo.generate_it.tasks`), con soglia di back-translation e
  ammissibilità per fonte;
- `results.jsonl`: un esito per item (accettato o rifiutato con motivo), con
  chiamate, token e costo;
- `pool.jsonl`, `dedup-report.json`, `ATTRIBUTIONS-it.md`: i casi accettati
  dopo la deduplica contro i benchmark e fra loro.

La spesa passa da `scudo.budget`: conferma umana del volume con lo stesso
teacher, stima nel residuo del volume confermato e del tetto, `--confirm-spend`
alla lettera, registro e stop rigido. Item e chiamate sono sequenziali; un
item parte solo se il suo caso peggiore sta nell'autorizzato.
"""

import hashlib
import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

from scudo import budget, generate_it, translate
from scudo.budget import BudgetError, Guard, Ledger
from scudo.dedup import digest
from scudo.ingest import Loader, benchmark_index, check_outside_repo
from scudo.manifest import Entry, benchmarks, problems
from scudo.teacher import CANDIDATES, TeacherConfig
from scudo.teacher_client import Client

PLAN, META, RESULTS, POOL = "plan.jsonl", "plan-meta.json", "results.jsonl", "pool.jsonl"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(records: list[dict], entries: list[Entry], *, translate_n: int, native_n: int,
              mixed_n: int, seed: int, threshold: float) -> dict:
    """Piano deterministico; fallisce se i casi traducibili sono meno di quelli chiesti."""
    if not 0 < threshold <= 1:
        raise ValueError("soglia di back-translation fuori da (0, 1]")
    pool, meta = translate.eligible(records, entries)
    if len(pool) < translate_n:
        raise ValueError(f"casi traducibili {len(pool)}, ne servono {translate_n}")
    pool.sort(key=lambda r: hashlib.sha256(
        f"{seed}:{r['entry']}:{r['source_id']}:{r['surface']}".encode()).hexdigest())
    tasks = [{"key": f"tr:{r['entry']}:{r['source_id']}:{r['surface']}", "kind": "translate",
              "language": "it", "source_case": r} for r in pool[:translate_n]]
    tasks += generate_it.tasks("it", native_n, seed=seed)
    tasks += generate_it.tasks("mixed", mixed_n, seed=seed)
    return {"tasks": tasks, "meta": {
        "seed": seed, "threshold": threshold, "eligible": len(pool), "translation": meta,
        "counts": {"translate": translate_n, "native": native_n, "mixed": mixed_n},
        "prompt_versions": [translate.PROMPT_VERSION, generate_it.PROMPT_VERSION]}}


def write_plan(out: Path, plan: dict) -> Path:
    out = check_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / PLAN, plan["tasks"])
    (out / META).write_text(json.dumps(plan["meta"], indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return out


def read_plan(out: Path) -> tuple[list[dict], dict]:
    out = check_outside_repo(out)
    tasks = _jsonl(out / PLAN)
    if not tasks:
        raise ValueError(f"{out}: {PLAN} assente o vuoto, eseguire prima `plan`")
    return tasks, json.loads((out / META).read_text(encoding="utf-8"))


def read_results(out: Path) -> list[dict]:
    return _jsonl(check_outside_repo(out) / RESULTS)


def read_pool(out: Path) -> list[dict]:
    return _jsonl(check_outside_repo(out) / POOL)


def _todo(out: Path) -> list[dict]:
    done = {r["key"] for r in read_results(out)}
    return [task for task in read_plan(out)[0] if task["key"] not in done]


def _worst_chats(task: dict, cfg: TeacherConfig) -> list[list[dict]]:
    if task["kind"] == "translate":
        return translate.worst_chats(task["source_case"], cfg.max_tokens)
    return generate_it.worst_chats(task, cfg.max_tokens)


def _worst(task: dict, cfg: TeacherConfig) -> float:
    return math.fsum(budget.worst_case_usd(cfg, chat) for chat in _worst_chats(task, cfg))


def estimate(out: Path, cfg: TeacherConfig, ledger: Ledger) -> dict:
    """Prova a secco: caso peggiore degli item mancanti, nessuna rete, nessuna chiave."""
    todo = _todo(out)
    total = math.fsum(_worst(task, cfg) for task in todo)
    confirmations = ledger.of_kind("volume_confirmation")
    return {"items": len(todo), "calls": sum(len(_worst_chats(t, cfg)) for t in todo),
            "total_usd": total, "confirm": f"{budget.cents_up(total):.2f}",
            "spent_usd": ledger.spent(), "remaining_usd": ledger.remaining(),
            "volume_confirmation": confirmations[-1] if confirmations else None,
            "volume_remaining_usd": budget.volume_remaining(ledger) if confirmations else None,
            "method": f"token stimati come caratteri/{budget.CHARS_PER_TOKEN} sull'ingresso più "
                      "max_tokens in uscita per chiamata; le uscite dei passi precedenti "
                      "contano max_tokens (caso peggiore, stima)"}


def _translator(cfg: TeacherConfig) -> dict:
    return {"model_id": cfg.model_id, "provider": cfg.provider,
            "provider_model": cfg.provider_model, "served_revision": cfg.served_revision}


def _caller(client: Client, cfg: TeacherConfig, guard: Guard, key: str, calls: list):
    def call(step: str, chat: list[dict], field: str | None):
        send = partial(client.judge, chat) if field is None else partial(client.complete, chat,
                                                                          field)
        outcome, cost, usage_missing = budget.paid_call(guard, cfg, f"{key}:{step}", chat, send)
        calls.append({"step": step, "status": outcome.status, "detail": outcome.detail,
                      "prompt_tokens": outcome.prompt_tokens,
                      "completion_tokens": outcome.completion_tokens,
                      "reasoning_tokens": outcome.reasoning_tokens, "cost_usd": cost,
                      "usage_missing": usage_missing})
        return outcome
    return call


def _check_volume(cfg: TeacherConfig, ledger: Ledger, planned: float) -> None:
    budget.check_volume(ledger, [cfg.model_id], planned)
    if CANDIDATES[cfg.model_id].license not in translate.TRANSLATOR_LICENSES:
        raise BudgetError(f"licenza del teacher {CANDIDATES[cfg.model_id].license} non ammessa")


def run(out: Path, cfg: TeacherConfig, ledger: Ledger, *, confirm: str | None,
        transport=None, environ=None, sleep=None) -> dict:
    """Item mancanti con stop rigido; ogni esito in results.jsonl, ogni costo nel registro."""
    out = check_outside_repo(out)
    _, meta = read_plan(out)
    todo = _todo(out)
    planned = math.fsum(_worst(task, cfg) for task in todo)
    _check_volume(cfg, ledger, planned)
    options = {k: v for k, v in (("transport", transport), ("environ", environ),
                                 ("sleep", sleep)) if v is not None}
    client = Client(cfg, **options)  # chiave verificata prima di spendere
    run_id = "it-" + _sha(out / PLAN)[:16]
    guard = Guard(ledger, run_id, budget.authorise(ledger, run_id, planned, confirm))
    translator, done, stopped = _translator(cfg), 0, None
    with (out / RESULTS).open("a", encoding="utf-8") as handle:
        for task in todo:
            calls: list[dict] = []
            call = _caller(client, cfg, guard, task["key"], calls)
            try:
                guard.check(_worst(task, cfg))
                if task["kind"] == "translate":
                    result = translate.process(task["source_case"], call,
                                               threshold=meta["threshold"], translator=translator)
                else:
                    result = generate_it.process(task, call, generator=translator)
            except BudgetError as error:  # stop rigido o chiave rifiutata (401/403)
                stopped = str(error)
                break
            handle.write(json.dumps({"key": task["key"], "kind": task["kind"],
                                     "language": task["language"], **result, "calls": calls,
                                     "cost_usd": math.fsum(c["cost_usd"] for c in calls)},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            done += 1
    return {"run_id": run_id, "done": done, "planned": len(todo),
            "authorised_usd": guard.authorised, "spent_usd": guard.spent, "stopped": stopped,
            "summary": summary(out)}


def summary(out: Path) -> dict:
    """Accettati per origine e lingua, rifiutati per tipo e motivo, costo dei risultati."""
    accepted, rejected = Counter(), {}
    results = read_results(out)
    for r in results:
        if r["status"] == "accepted":
            accepted[f"{r['provenance']['kind']}/{r['case']['language']}"] += 1
        else:
            rejected.setdefault(r["kind"], Counter())[r["reason"]] += 1
    return {"done": len(results), "accepted": dict(sorted(accepted.items())),
            "rejected": {k: dict(sorted(v.items())) for k, v in sorted(rejected.items())},
            "cost_usd": math.fsum(r["cost_usd"] for r in results)}


def dedup(out: Path, entries: list[Entry], loader: Loader) -> dict:
    """Pool dei casi accettati senza duplicati dei benchmark (anche l'originale EN) né interni."""
    out = check_outside_repo(out)
    benches = benchmarks(entries)
    usable = [b for b in benches if not problems([b])]
    index = benchmark_index(usable, loader)
    pool, removed, seen = [], Counter(), set()
    for r in read_results(out):
        if r["status"] != "accepted":
            continue
        case, prov = r["case"], r["provenance"]
        texts = [case["evidence"]] + ([prov["original_evidence"]]
                                      if prov["kind"] == "translated" else [])
        if hit := next(filter(None, map(index.match, texts)), None):
            removed[f"{hit[0]}:{hit[1]}"] += 1
            continue
        if (key := digest(case["evidence"])) in seen:
            removed["duplicate_in_batch"] += 1
            continue
        seen.add(key)
        pool.append({"key": r["key"], "origin": prov["kind"], **case, "provenance": prov})
    _write_jsonl(out / POOL, pool)
    translated = [p["provenance"] for p in pool if p["origin"] == "translated"]
    used = {p["entry"] for p in translated}
    if translated:
        (out / "ATTRIBUTIONS-it.md").write_text(translate.attributions(
            [e for e in entries if e.name in used], translated[0]["translator"]),
            encoding="utf-8")
    report = {"results_sha256": _sha(out / RESULTS), "pool_sha256": _sha(out / POOL),
              "kept": len(pool), "removed": dict(sorted(removed.items())),
              "by_language": dict(sorted(Counter(p["language"] for p in pool).items())),
              "benchmarks": {b.name: index.sizes.get(b.name, 0) for b in usable},
              "not_indexed": {b.name: problems([b]) for b in benches if b not in usable}}
    (out / "dedup-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)
                                           + "\n", encoding="utf-8")
    return report
