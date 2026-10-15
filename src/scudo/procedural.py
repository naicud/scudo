"""Casi a criterio personalizzato da `tasksource/procedural-typed-decisions` (Apache-2.0).

Ogni riga è uno stato con più domande tipizzate e risposte calcolate in modo
esatto dalle regole scritte nello stato. Mappatura:

- `noul` sì/no: la domanda è il criterio, 1 è VIOLATION e 0 è NO_VIOLATION;
  i `noul` graduati (probabilità esatte) restano fuori anche se valgono 0 o 1;
- `choice` a tre valori di `record_eligibility`, dove «sconosciuto» significa
  per definizione dello stato informazione assente o contraddittoria: diventano
  tri-state con un criterio riscritto (violato/ineleggibile è VIOLATION);
- ogni altra `choice` e ogni `score` restano fuori con il motivo.

L'evidenza è lo stato verbatim; superficie `prompt` come per Open-Jev.
"""

import json
import re
from collections import Counter
from collections.abc import Iterable

from scudo.cases import Case

DATASET = "tasksource/procedural-typed-decisions"
CONFIG = "all"
REVISION = "e4b21b7b55f6cc9e76e33563e9d604ac5fcd25ad"
LICENCE = "Apache-2.0"

# `noul` graduati: colonna float, non ClassLabel, nelle config per task (datasets-server /info).
GRADED_NOUL = frozenset({
    ("arithmetic", "random_line_bulk"),
    ("arithmetic", "random_is_deposit"),
    ("arithmetic", "random_is_long"),
    ("partial_observation_calibration", "incident_real"),
    ("policy_under_uncertainty", "access_allowed"),
})


def eligibility_criterion(record: str) -> str:
    return (
        f"Is record {record} ineligible under the numbered requirements of the user's request, "
        "that is, does it violate at least one of them? If no requirement is violated but some "
        "cannot be determined from the record descriptions, or the record is not described, "
        "that means INSUFFICIENT."
    )


def requirement_criterion(record: str, requirement: str) -> str:
    return (
        f"Does record {record} violate requirement {requirement} of the user's request? If the "
        "record descriptions do not establish whether the requirement is met or violated, or the "
        "record is not described, that means INSUFFICIENT."
    )


# (task, domanda) -> (forma della domanda nella fonte, criterio, valore -> verdetto)
TRISTATE_CHOICES = {
    ("record_eligibility", "probe_eligibility"): (
        re.compile(r"What is the eligibility status of (R\d+)\?"),
        eligibility_criterion,
        {"ineligible": "VIOLATION", "eligible": "NO_VIOLATION", "undetermined": "INSUFFICIENT"},
    ),
    ("record_eligibility", "requirement_status"): (
        re.compile(r"For (R\d+), is (C\d+) met, violated or unknown\?"),
        requirement_criterion,
        {"violated": "VIOLATION", "met": "NO_VIOLATION", "unknown": "INSUFFICIENT"},
    ),
}


def _case(row: dict, name: str, question: dict, answer: dict) -> tuple[str, str] | str:
    """(criterio, verdetto) per una domanda, oppure il motivo dello scarto."""
    kind, task = question["type"], row["task"]
    if kind == "score":
        return "score_not_tristate"
    if kind == "noul":
        value = answer["noul"]
        if (task, name) in GRADED_NOUL or value not in (0.0, 1.0):
            return "graded_noul"
        return question["instructions"], "VIOLATION" if value == 1.0 else "NO_VIOLATION"
    if kind != "choice":
        raise ValueError(f"tipo di domanda sconosciuto: {kind!r}")
    if (task, name) not in TRISTATE_CHOICES:
        return "choice_not_tristate"
    pattern, build, verdicts = TRISTATE_CHOICES[(task, name)]
    match = pattern.fullmatch(question["instructions"])
    if match is None:
        return "unparsed_question"
    if answer["choice"] not in verdicts:
        raise ValueError(f"valore tri-state sconosciuto per {name}: {answer['choice']!r}")
    if answer["probabilities"][answer["choice"]] != 1.0:
        return "soft_label"
    return build(*match.groups()), verdicts[answer["choice"]]


def cases_from_row(row: dict) -> tuple[list[Case], Counter]:
    """Un caso per domanda convertibile; ogni domanda scartata è contata con il motivo."""
    questions, answers = json.loads(row["questions"]), json.loads(row["answers"])
    cases, dropped = [], Counter()
    for name, question in questions.items():
        result = _case(row, name, question, answers[name])
        if isinstance(result, str):
            dropped[result] += 1
            continue
        criterion, verdict = result
        cases.append(Case(source=DATASET, source_id=f"{row['id']}:{name}", language="en",
                          surface="prompt", criterion=criterion, evidence=row["state"],
                          verdict=verdict))
    return cases, dropped


# Ingestione (`scudo.ingest`): casi e scarti per motivo di una riga.
convert_row = cases_from_row


def needs_relabel(row: dict, case: Case) -> bool:
    """Risposte calcolate esattamente dallo stato."""
    return False


def convert(rows: Iterable[dict]) -> tuple[list[Case], Counter]:
    cases, dropped = [], Counter()
    for row in rows:
        row_cases, row_dropped = cases_from_row(row)
        cases.extend(row_cases)
        dropped.update(row_dropped)
    return cases, dropped
