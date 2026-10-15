"""Casi da `CohereLabs/aya_redteaming` (Apache-2.0), solo lo split `english`.

Il dataset raccoglie prompt dannosi di red-teaming scritti da annotatori
professionisti (README): ogni riga con almeno una `harm_category` è VIOLATION
sul prompt, con il criterio di Aegis 2.0. È solo positivi: i NO_VIOLATION
devono venire da altre fonti. La fonte non ha id di riga: `source_id` è un hash
del prompt.
"""

import hashlib
import json

from scudo.aegis import HARMFUL_REQUEST
from scudo.cases import Case

DATASET = "CohereLabs/aya_redteaming"
REVISION = "5a16fee03c19d681b4f3527b6a750e1ed4d434c7"
SPLIT = "english"  # aya_eng.jsonl; lo split si chiama come la lingua


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    if row["language"] != "English":
        return "lingua non inglese"
    if not row["prompt"].strip():
        return "testo vuoto"
    if not json.loads(row["harm_category"]):
        return "nessuna categoria di danno"
    return None


def cases_from_row(row: dict) -> list[Case]:
    if drop_reason(row):
        return []
    key = hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
    return [Case(source=DATASET, source_id=key, language="en", surface="prompt",
                 criterion=HARMFUL_REQUEST, evidence=row["prompt"], verdict="VIOLATION")]


def needs_relabel(row: dict, case: Case) -> bool:
    """Prompt e categorie scritti da annotatori umani."""
    return False
