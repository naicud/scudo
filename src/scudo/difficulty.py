"""Criterio di difficoltà del Fast (ADR 0001): testo, label del primo giro, righe e metriche.

Id e testo sono copiati verbatim da open-guard 2ed1e92
(`open_guard.scoring.schema.ROUTING_CRITERIA`). Il backend nativo lo rende con lo
stesso renderer decision-token dei criteri semantici e sulla stessa evidenza
(`snapshot.text`); `cascade.escalation_reason` legge VIOLATION come HARD,
NO_VIOLATION come EASY, INSUFFICIENT o un readout fallito come non valutabile,
quindi difficile.

Label del primo giro dai giudizi dei teacher, nel formato degli esiti del pilot
(`outcomes.jsonl`: `teacher`, `key`, `status`, `verdict`, ...): HARD se i teacher
discordano fra loro o con il gold, EASY se concordano tutti con il gold. Gli esiti
non portano margine né confidenza: si usa solo il disaccordo. La giuria sono i
teacher presenti negli esiti; un caso senza l'esito di ognuno, o con un esito
diverso da `ok` (rifiuto, malformato, troncato, timeout, errore), non ha label e
si conta per motivo: un guasto del teacher non è un segnale di difficoltà.
"""

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from scudo.cases import Case
from scudo.eval_report import UNSUPPORTED, EvalResult
from scudo.protocol import SEMANTIC_IDS, training_row

ROUTING_DIFFICULTY = "routing.difficulty"
ROUTING_DIFFICULTY_REF = ROUTING_DIFFICULTY + "/1"  # open-guard: cambia con il testo
CRITERION = (
    "Is this case hard to judge reliably at a glance for a small classifier: subtle or "
    "indirect wording, mixed or unclear language, long or layered context, quotation or "
    "role-play that blurs intent, or evidence that pulls toward opposite verdicts? Answer "
    "VIOLATION when the case is hard, NO_VIOLATION when it is plainly easy, and "
    "INSUFFICIENT when the difficulty itself cannot be determined."
)
HARD, EASY = "VIOLATION", "NO_VIOLATION"  # verdetti del criterio di difficoltà

Index = dict[str, dict[str, dict]]  # key del caso -> teacher -> esito


def index_outcomes(records: Iterable[dict]) -> Index:
    """Esiti per caso e teacher; due esiti diversi per la stessa coppia sono un errore."""
    index: Index = {}
    for record in records:
        by_teacher = index.setdefault(record["key"], {})
        seen = by_teacher.get(record["teacher"])
        if seen is not None and (seen["status"], seen.get("verdict")) != (
                record["status"], record.get("verdict")):
            raise ValueError(f"esiti discordi per {record['teacher']} su {record['key']}")
        by_teacher[record["teacher"]] = record
    return index


def read_outcomes(paths: Sequence[Path]) -> Index:
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    return index_outcomes(records)


def jury(index: Index) -> tuple[str, ...]:
    """I teacher presenti negli esiti: ognuno deve aver giudicato un caso perché conti."""
    return tuple(sorted({teacher for by_teacher in index.values() for teacher in by_teacher}))


def assess(index: Index, key: str, teachers: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """("ok", verdetti) se ogni teacher ha un esito `ok`; altrimenti (motivo, ())."""
    by_teacher = index.get(key, {})
    if not teachers or any(t not in by_teacher for t in teachers):
        return "no_teacher_record", ()
    statuses = [by_teacher[t]["status"] for t in teachers]
    if bad := sorted({s for s in statuses if s != "ok"}):
        return f"teacher_{bad[0]}", ()
    return "ok", tuple(by_teacher[t]["verdict"] for t in teachers)


def label(index: Index, key: str, gold: str, teachers: Sequence[str]) -> tuple[str | None, str]:
    """(HARD | EASY | None, motivo): HARD con disaccordo fra teacher o con il gold."""
    status, verdicts = assess(index, key, teachers)
    if status != "ok":
        return None, status
    return (EASY, "agree") if set(verdicts) == {gold} else (HARD, "disagree")


def row(case: Case, verdict: str, rng: random.Random) -> dict:
    """Riga decision-token del criterio di difficoltà sulla stessa evidenza del caso."""
    if verdict not in (HARD, EASY):
        raise ValueError(f"verdetto di difficoltà {verdict!r} non in HARD/EASY")
    return {**training_row(CRITERION, case.evidence, verdict, rng),
            "criterion_ref": ROUTING_DIFFICULTY_REF,
            "difficulty": "HARD" if verdict == HARD else "EASY"}


def as_difficulty(case: Case) -> Case:
    """Lo stesso caso con il criterio di difficoltà; il readout legge criterio ed evidenza."""
    return Case(case.source, case.source_id, case.language, case.surface, CRITERION,
                case.evidence, case.verdict)


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _routing(pairs: list[tuple[EvalResult, str | None]]) -> dict:
    # Come cascade.escalation_reason: tutto ciò che non è NO_VIOLATION è difficile.
    hard = [(item, d != EASY) for item, d in pairs]
    wrong = [h for item, h in hard if item.predicted != item.gold]
    right = [h for item, h in hard if item.predicted == item.gold]
    readouts = Counter(d if d is not None else "ERROR" for _, d in pairs)
    return {
        "cases": len(pairs),
        "fast_wrong": len(wrong),
        "hard_rate": _rate(sum(h for _, h in hard), len(hard)),
        "hard_recall_on_wrong": _rate(sum(wrong), len(wrong)),
        "hard_rate_on_correct": _rate(sum(right), len(right)),
        "difficulty_readouts": {key: readouts[key] for key in (*SEMANTIC_IDS, "ERROR")},
    }


def routing_metrics(semantic: Sequence[EvalResult], difficulty: Mapping[str, str | None]) -> dict:
    """Instradamento del criterio di difficoltà contro gli errori del Fast sullo stesso caso.

    `difficulty`: case_id -> verdetto del readout di difficoltà, None se fallito.
    `hard_recall_on_wrong` è la quota dei casi sbagliati dal Fast (errori compresi)
    che il criterio manda all'Accurate; `hard_rate_on_correct` il costo sui giusti.
    """
    pairs = [(item, difficulty[item.case_id]) for item in semantic if item.error != UNSUPPORTED]
    if not pairs:
        raise ValueError("nessun caso da misurare")
    grouped = {}
    for field in ("language", "surface"):
        groups: dict[str, list] = {}
        for pair in pairs:
            groups.setdefault(getattr(pair[0], field), []).append(pair)
        grouped[f"by_{field}"] = {k: _routing(v) for k, v in sorted(groups.items())}
    return {"criterion_ref": ROUTING_DIFFICULTY_REF, "overall": _routing(pairs), **grouped}
