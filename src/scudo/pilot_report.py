"""Report del pilot dei teacher: accordo, rifiuti, costi, ragionamenti, rietichettatura, volume.

L'accordo conta ogni esito non `ok` (rifiuto, troncamento, malformato, timeout,
errore, caso non eseguito) come disaccordo: nessuno diventa `NO_VIOLATION`. Il
campione gold è bilanciato per strato, quindi l'accordo complessivo è sulla
composizione del campione, non sulla distribuzione delle fonti.

La qualità dei ragionamenti ha solo un proxy meccanico (coerenza delle menzioni
di verdetto, al più tre frasi, sovrapposizione lessicale con l'evidenza); il
giudizio vero è la revisione umana dell'export `reasoning-review-*.csv`.

Le righe `llm_jury` di Aegis 2.0 non hanno una label umana (una sola fonte di
label per riga): la validazione confronta giuria originale e giuria di teacher,
riporta l'accordo dei teacher sulle risposte gold come ancora, ed esporta uno
spot-check cieco (`relabel-spotcheck.csv`) che una persona compila; con
`--spotcheck` il report misura giuria e teacher contro le sue label.

La stima del volume v0.1 moltiplica il costo per item misurato nel pilot per i
volumi pianificati, parametri espliciti in `Volumes`.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from scudo import budget, pilot, pilot_sample, pilot_summary
from scudo.budget import CAP_USD, Ledger
from scudo.protocol import SEMANTIC_IDS
from scudo.teacher import CANDIDATES, MAX_SENTENCES, PROMPT_VERSION, TeacherConfig

STATUSES = ("ok", "refused", "truncated", "malformed", "timeout", "error", "missing")
REVIEW_N = 30
SPOT_N = 50
SPOT_COLUMNS = ["key", "criterion", "evidence", "human_verdict", "notes"]


@dataclass(frozen=True)
class Volumes:
    fast_items: int = 200_000  # #8: esempi del Fast v0.1
    fast_calls_per_item: int = 2  # #8: giudizi per esempio, difficoltà dal disaccordo
    non_en_share: float = 0.6  # #7: IT 40% + misto 20% del mix, da tradurre o generare
    translation_calls_per_item: int = 2  # #7: traduzione più back-translation
    reasoning_traces: int = 40_000  # #9: ragionamenti tenuti (solo se arrivano alla label)


def _rate(part, whole):
    return part / whole if whole else None


def _mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


_VERDICT_TOKEN = re.compile(r"\b(?:NO_VIOLATION|INSUFFICIENT|VIOLATION)\b")
_SENTENCE = re.compile(r"[.!?]+(?:\s|$)")
_WORD = re.compile(r"[^\W\d_]{5,}")


def reasoning_proxy(verdict: str, reasoning: str, evidence: str) -> dict:
    """Proxy meccanico, non un giudizio di qualità."""
    mentioned = set(_VERDICT_TOKEN.findall(reasoning))
    sentences = len(_SENTENCE.findall(reasoning.strip())) or 1
    words = {w.casefold() for w in _WORD.findall(evidence)}
    return {"consistent": mentioned <= {verdict}, "within_limit": sentences <= MAX_SENTENCES,
            "evidence_overlap": bool(words & {w.casefold() for w in _WORD.findall(reasoning)}),
            "chars": len(reasoning)}


def _agrees(record, outcome) -> bool:
    return bool(outcome) and outcome["status"] == "ok" and outcome["verdict"] == record["verdict"]


def _by(records, outs, field: str) -> dict:
    groups: dict[str, list] = {}
    for record in records:
        groups.setdefault(record[field], []).append(record)
    return {value: {"n": len(group),
                    "agreement": _rate(sum(_agrees(r, outs.get(r["key"])) for r in group), len(group))}
            for value, group in sorted(groups.items())}


def teacher_metrics(records: list[dict], outs: dict) -> dict:
    n = len(records)
    calls = [outs[r["key"]] for r in records if r["key"] in outs]
    status = Counter(outs[r["key"]]["status"] if r["key"] in outs else "missing" for r in records)
    ok = [(r, outs[r["key"]]) for r in records if r["key"] in outs and outs[r["key"]]["status"] == "ok"]
    proxies = [reasoning_proxy(o["verdict"], o["reasoning"], r["evidence"]) for r, o in ok]
    cost = math.fsum(o["cost_usd"] for o in calls)
    return {
        "n": n, "status": {s: status[s] for s in STATUSES},
        "refusal_rate": _rate(status["refused"], n), "malformed_rate": _rate(status["malformed"], n),
        "truncated_rate": _rate(status["truncated"], n),
        "error_rate": _rate(status["timeout"] + status["error"], n),
        "agreement": _rate(sum(_agrees(r, outs.get(r["key"])) for r in records), n),
        "agreement_on_ok": _rate(sum(o["verdict"] == r["verdict"] for r, o in ok), len(ok)),
        "by_verdict": _by(records, outs, "verdict"), "by_language": _by(records, outs, "language"),
        "by_surface": _by(records, outs, "surface"),
        "mean_prompt_tokens": _mean(o["prompt_tokens"] for o in calls),
        "mean_completion_tokens": _mean(o["completion_tokens"] for o in calls),
        "mean_reasoning_tokens": _mean(o["reasoning_tokens"] for o in calls),
        "usage_missing": sum(o["usage_missing"] for o in calls),
        "cost_usd": cost, "cost_per_item_usd": _rate(cost, len(calls)),
        "reasoning_proxy": {
            "n_ok": len(ok),
            **{f"{key}_rate": _rate(sum(p[key] for p in proxies), len(proxies))
               for key in ("consistent", "within_limit", "evidence_overlap")},
            "mean_chars": _mean(p["chars"] for p in proxies)},
    }


def _teacher_jury(record, outs_by_teacher) -> str | None:
    """Verdetto della giuria di teacher: tutti ok e concordi, altrimenti irrisolto."""
    outs = [o.get(record["key"]) for o in outs_by_teacher.values()]
    verdicts = {o["verdict"] for o in outs if o and o["status"] == "ok"}
    if len(verdicts) == 1 and all(o and o["status"] == "ok" for o in outs):
        return verdicts.pop()
    return None


def _spot_category(record, outs_by_teacher) -> str:
    verdict = _teacher_jury(record, outs_by_teacher)
    if verdict is None:
        return "unresolved"
    return "agrees_jury" if verdict == record["verdict"] else "disagrees_jury"


def _seeded(seed: int, record: dict) -> str:
    return hashlib.sha256(f"{seed}:{record['key']}".encode()).hexdigest()


def _pick(records: list[dict], n: int, stratum: Callable[[dict], str], seed: int) -> list[dict]:
    pools: dict[str, list] = {}
    for record in records:
        pools.setdefault(stratum(record), []).append(record)
    quota = pilot_sample.allocate({k: len(v) for k, v in pools.items()}, min(n, len(records)))
    picked = [r for k in sorted(pools)
              for r in sorted(pools[k], key=lambda r: _seeded(seed, r))[:quota[k]]]
    return sorted(picked, key=lambda r: _seeded(seed + 1, r))


def score_spotcheck(path: Path, relabel: list[dict], outs_by_teacher: dict) -> dict:
    by_key = {r["key"]: r for r in relabel}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid, invalid = [], 0
    for row in rows:
        human = (row.get("human_verdict") or "").strip().upper()
        if row.get("key") in by_key and human in SEMANTIC_IDS:
            valid.append((by_key[row["key"]], human))
        else:
            invalid += 1
    covered = [(r, h, v) for r, h in valid if (v := _teacher_jury(r, outs_by_teacher))]
    return {
        "n": len(valid), "invalid": invalid,
        "jury_accuracy": _rate(sum(r["verdict"] == h for r, h in valid), len(valid)),
        "teacher_jury_covered": len(covered),
        "teacher_jury_accuracy": _rate(sum(v == h for _, h, v in covered), len(covered)),
        "per_teacher_accuracy": {
            t: _rate(sum(bool(o.get(r["key"])) and o[r["key"]]["status"] == "ok"
                         and o[r["key"]]["verdict"] == h for r, h in valid), len(valid))
            for t, o in outs_by_teacher.items()},
    }


def relabel_validation(relabel, outs_by_teacher, gold_metrics, spotcheck=None) -> dict:
    jury = [(r, _teacher_jury(r, outs_by_teacher)) for r in relabel]
    covered = [(r, v) for r, v in jury if v is not None]
    return {
        "n": len(relabel),
        "truth": "nessuna label umana per le righe llm_jury nello schema di Aegis 2.0",
        "agreement_with_jury": {t: teacher_metrics(relabel, o)["agreement"]
                                for t, o in outs_by_teacher.items()},
        "teacher_jury_coverage": _rate(len(covered), len(relabel)),
        "teacher_jury_agrees_with_jury": _rate(sum(r["verdict"] == v for r, v in covered), len(covered)),
        "anchor_gold_response_agreement": {
            t: m["by_surface"].get("response", {}).get("agreement") for t, m in gold_metrics.items()},
        "spot_check": (score_spotcheck(spotcheck, relabel, outs_by_teacher) if spotcheck
                       else {"status": "in attesa: compilare relabel-spotcheck.csv e rieseguire "
                                       "report con --spotcheck"}),
    }


def volume_estimate(configs, metrics, gold, volumes: Volumes, remaining: float) -> dict:
    evidence_tokens = _mean(budget.estimate_tokens(r["evidence"]) for r in gold)
    translation_items = round(volumes.fast_items * volumes.non_en_share)
    teachers = {}
    for cfg in configs:
        m = metrics[cfg.model_id]
        per_item, agreement = m["cost_per_item_usd"], m["agreement"]
        fast = reasoning = translation = None
        if per_item is not None:
            fast = volumes.fast_items * volumes.fast_calls_per_item * per_item
            reasoning = volumes.reasoning_traces / agreement * per_item if agreement else None
        if m["mean_prompt_tokens"] is not None and evidence_tokens is not None:
            out_tokens = evidence_tokens + (m["mean_reasoning_tokens"] or 0)
            translation = translation_items * volumes.translation_calls_per_item * budget.cost_usd(
                cfg, m["mean_prompt_tokens"], out_tokens)
        parts = (fast, reasoning, translation)
        total = math.fsum(parts) if None not in parts else None
        teachers[cfg.model_id] = {"fast_difficulty_usd": fast, "reasoning_usd": reasoning,
                                  "translation_usd": translation, "total_usd": total,
                                  "fits_remaining": total is not None and total <= remaining}
    return {
        "parameters": asdict(volumes), "translation_items": translation_items,
        "remaining_usd": remaining, "teachers": teachers,
        "method": ("stima. #8: esempi × giudizi per esempio × costo medio per chiamata del pilot; "
                   "#9: tracce ÷ accordo con la label (si tengono solo quelle corrette) × costo "
                   "per chiamata; #7: item IT/misti × chiamate (traduzione e back-translation) × "
                   "costo di una chiamata con ingresso = token medi di ingresso del pilot e uscita "
                   f"= evidenza media (caratteri/{budget.CHARS_PER_TOKEN}) più token di "
                   "ragionamento medi riportati dal provider"),
    }


def _review_rows(gold, outs) -> list[dict]:
    rows = []
    for record in gold:
        outcome = outs.get(record["key"])
        if outcome and outcome["status"] == "ok":
            proxy = reasoning_proxy(outcome["verdict"], outcome["reasoning"], record["evidence"])
            rows.append({**record, "teacher_verdict": outcome["verdict"],
                         "reasoning": outcome["reasoning"], **proxy})
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(out: Path, configs: list[TeacherConfig], ledger: Ledger, *,
                 volumes: Volumes = Volumes(), spotcheck: Path | None = None,
                 review_n: int = REVIEW_N, spot_n: int = SPOT_N, seed: int = 0) -> dict:
    sample, meta = pilot.read_sample(out), pilot.read_meta(out)
    out = Path(out).resolve()
    gold = [r for r in sample if r["set"] == "gold"]
    relabel = [r for r in sample if r["set"] == "relabel"]
    outs_by_teacher = {cfg.model_id: {} for cfg in configs}
    for outcome in pilot.read_outcomes(out):
        if outcome["teacher"] in outs_by_teacher:
            outs_by_teacher[outcome["teacher"]][outcome["key"]] = outcome
    gold_metrics = {t: teacher_metrics(gold, o) for t, o in outs_by_teacher.items()}
    sha = pilot.sample_sha(out)
    report = {
        "sample": {"sha256": sha, "gold": len(gold), "relabel": len(relabel), "meta": meta},
        "prompt_version": PROMPT_VERSION,
        "teachers": {cfg.model_id: {
            "registry": asdict(CANDIDATES[cfg.model_id]),
            "provider": {k: v for k, v in asdict(cfg).items() if k not in ("model_id", "api_key_env")},
            "gold": gold_metrics[cfg.model_id],
            "relabel": teacher_metrics(relabel, outs_by_teacher[cfg.model_id]),
        } for cfg in configs},
        "teacher_agreement_gold": _rate(sum(_teacher_jury(r, outs_by_teacher) is not None
                                            for r in gold), len(gold)),
        "relabel_validation": relabel_validation(relabel, outs_by_teacher, gold_metrics, spotcheck),
        "volume": volume_estimate(configs, gold_metrics, gold, volumes, ledger.remaining()),
        "spend": {"pilot_usd": ledger.spent_by_run(sha[:16]), "ledger_total_usd": ledger.spent(),
                  "remaining_usd": ledger.remaining(), "cap_usd": CAP_USD,
                  "authorisations": [r for r in ledger.of_kind("authorisation")
                                     if r["run_id"] == sha[:16]]},
    }
    for cfg in configs:
        rows = _pick(_review_rows(gold, outs_by_teacher[cfg.model_id]), review_n,
                     lambda r: "agree" if r["teacher_verdict"] == r["verdict"] else "disagree", seed)
        _write_csv(out / f"reasoning-review-{cfg.model_id.replace('/', '_')}.csv",
                   ["key", "language", "surface", "criterion", "evidence", "label",
                    "teacher_verdict", "reasoning", "consistent", "within_limit",
                    "reasoning_ok", "notes"],
                   [{**r, "label": r["verdict"]} for r in rows])
    spot = _pick(relabel, spot_n, lambda r: _spot_category(r, outs_by_teacher), seed)
    _write_csv(out / "relabel-spotcheck.csv", SPOT_COLUMNS, spot)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    (out / "summary.md").write_text(pilot_summary.markdown(report), encoding="utf-8")
    ledger.append({"kind": "pilot_report", "sample_sha256": sha,
                   "spent_usd": report["spend"]["pilot_usd"]})
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for spec in fields(Volumes):
        parser.add_argument(f"--{spec.name.replace('_', '-')}", type=type(spec.default),
                            default=spec.default)
    parser.add_argument("--review-n", type=int, default=REVIEW_N)
    parser.add_argument("--spot-n", type=int, default=SPOT_N)
    parser.add_argument("--spotcheck", type=Path, default=None, help="spot-check compilato")


def options(args: argparse.Namespace) -> dict:
    return {"volumes": Volumes(**{f.name: getattr(args, f.name) for f in fields(Volumes)}),
            "spotcheck": args.spotcheck, "review_n": args.review_n, "spot_n": args.spot_n}
