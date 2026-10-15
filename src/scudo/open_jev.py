"""Casi a criterio personalizzato da `ZefanCai/Open-Jev`, solo config redistributable.

Una riga è uno stato e una domanda tipizzata. Entrano solo le domande `noul`
sì/no con target netto: la domanda è il criterio, lo stato è l'evidenza, `yes`
è VIOLATION (criterio soddisfatto) e `no` è NO_VIOLATION. `choice` e `score`
non si riducono al tri-state senza perdita e restano fuori con il motivo.
Superficie `prompt`: lo stato è il dato in ingresso al sistema, non un output
del modello. `metadata_json` e `record_json` possono contenere label teacher
privilegiate: non si leggono mai.
"""

import json
from collections import Counter
from collections.abc import Iterable

from scudo.cases import Case

DATASET = "ZefanCai/Open-Jev"
CONFIG = "release-v2-redistributable"
REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
LICENCE = "CC0-1.0"  # record generati; codice MIT; testi upstream esclusi (README)

# Le descrizioni delle domande di customer-control-v1 vengono dalla documentazione
# TypeSafe, con licenza dichiarata non verificata nella provenienza della riga.
UNVERIFIED_SOURCES = frozenset({"customer-control-v1"})
_VERDICTS = {(1.0, 0.0): "NO_VIOLATION", (0.0, 1.0): "VIOLATION"}
_DROPPED_KINDS = {"choice": "choice_not_tristate", "score": "score_not_tristate"}


def _evidence(state_json: str) -> str:
    state = json.loads(state_json)
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def cases_from_row(row: dict) -> tuple[list[Case], Counter]:
    """Al più un caso per riga; ogni riga scartata è contata con il motivo."""
    kind = row["kind"]
    if kind in _DROPPED_KINDS:
        return [], Counter({_DROPPED_KINDS[kind]: 1})
    if kind != "noul":
        raise ValueError(f"tipo Open-Jev sconosciuto: {kind!r}")
    if row["source"] in UNVERIFIED_SOURCES:
        return [], Counter({"unverified_upstream_licence": 1})
    if list(row["options"]) != ["no", "yes"]:
        return [], Counter({"unexpected_options": 1})
    verdict = _VERDICTS.get(tuple(float(x) for x in row["target"]))
    if verdict is None:
        return [], Counter({"soft_target": 1})
    case = Case(source=DATASET, source_id=row["id"], language="en", surface="prompt",
                criterion=row["question"], evidence=_evidence(row["state_json"]), verdict=verdict)
    return [case], Counter()


# Ingestione (`scudo.ingest`): casi e scarti per motivo di una riga.
convert_row = cases_from_row


def needs_relabel(row: dict, case: Case) -> bool:
    """Target di riferimento della fonte; il teacher non dichiarato blocca il cancello."""
    return False


def convert(rows: Iterable[dict]) -> tuple[list[Case], Counter]:
    cases, dropped = [], Counter()
    for row in rows:
        row_cases, row_dropped = cases_from_row(row)
        cases.extend(row_cases)
        dropped.update(row_dropped)
    return cases, dropped
