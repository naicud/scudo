"""Casi di injection diretta da `TrustAIRLab/in-the-wild-jailbreak-prompts` (MIT).

Prompt raccolti da Reddit, Discord, siti e dataset aperti (dic 2022 - dic
2023). La colonna `jailbreak` viene dai tag delle piattaforme (paper, §3):
vero nelle config `jailbreak_*`, falso nelle `regular_*`. Un jailbreak è
VIOLATION; un prompt regolare è NO_VIOLATION, ma è negativo per assenza del
tag, non per verifica (gli autori ne hanno controllato solo un campione), ed è
spesso un role-play vicino ai jailbreak: `needs_relabel` lo marca per la giuria
di teacher. La fonte non ha id di riga: `source_id` è un hash dei campi.
"""

import hashlib
import json

from scudo.cases import Case
from scudo.default_criteria import INJECTION_DIRECT

DATASET = "TrustAIRLab/in-the-wild-jailbreak-prompts"
REVISION = "a10aab8eff1c73165a442d4464dce192bd28b9c5"


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    return None if row["prompt"].strip() else "testo vuoto"


def _key(row: dict) -> str:
    fields = [row["jailbreak"], row["platform"], row["source"], row["created_at"], row["prompt"]]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()[:16]


def cases_from_row(row: dict) -> list[Case]:
    if drop_reason(row):
        return []
    if not isinstance(row["jailbreak"], bool):
        raise ValueError(f"label jailbreak non booleana: {row['jailbreak']!r}")
    return [Case(source=DATASET, source_id=_key(row), language="en", surface="injection_direct",
                 criterion=INJECTION_DIRECT.text, evidence=row["prompt"],
                 verdict="VIOLATION" if row["jailbreak"] else "NO_VIOLATION")]


def needs_relabel(row: dict, case: Case) -> bool:
    """I prompt regolari non sono negativi verificati."""
    return not row["jailbreak"]
