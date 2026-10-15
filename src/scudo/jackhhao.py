"""Casi di injection diretta da `jackhhao/jailbreak-classification` (Apache-2.0).

Lo split `train` della config `default` è
`balanced/jailbreak_dataset_train_balanced.csv` (`prompt`, `type`). Entrano
solo le righe `jailbreak`, raccolte da `verazuo/jailbreak_llms` (MIT), come
VIOLATION; i `benign` vengono da OpenOrca e GPTeacher, generati con modelli
OpenAI, e restano fuori con il motivo. La fonte non ha id di riga:
`source_id` è un hash del prompt.
"""

import hashlib

from scudo.cases import Case
from scudo.default_criteria import INJECTION_DIRECT

DATASET = "jackhhao/jailbreak-classification"
REVISION = "2f2ceeb39658696fd3f462403562b6eea5306287"


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    if row["type"] == "benign":
        return "benigno generato con modelli OpenAI"
    if not row["prompt"].strip():
        return "testo vuoto"
    return None


def cases_from_row(row: dict) -> list[Case]:
    if drop_reason(row):
        return []
    if row["type"] != "jailbreak":
        raise ValueError(f"tipo jackhhao sconosciuto: {row['type']!r}")
    key = hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
    return [Case(source=DATASET, source_id=key, language="en", surface="injection_direct",
                 criterion=INJECTION_DIRECT.text, evidence=row["prompt"], verdict="VIOLATION")]


def needs_relabel(row: dict, case: Case) -> bool:
    """Tag jailbreak della fonte (piattaforme di jailbreak_llms)."""
    return False
