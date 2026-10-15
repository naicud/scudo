"""Casi da `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` (CC-BY-4.0), inglese.

Le label sono per dialogo (README): valgono per il prompt solo se il dialogo è
un solo messaggio utente. Entrano quindi solo le righe `text_type =
user_message`, superficie `prompt` con il criterio di Aegis 2.0;
`llm_response`, `combined` e `multi_turn` concatenano prompt e risposta senza
separatore e restano fuori. Ogni annotatore dà `Safe`, `Needs Caution` o una o
più categorie di rischio separate da virgola. Il verdetto chiede l'unanimità:
tutti `Safe` è NO_VIOLATION, tutti categorie di rischio è VIOLATION; zona
grigia, disaccordo, annotazioni mancanti e label fuori tassonomia restano fuori
con il motivo. Label umane: nulla da rietichettare.
"""

from scudo.aegis import HARMFUL_REQUEST
from scudo.cases import Case

DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
REVISION = "bd96d862068e47630197de64eb91f8d1481ff3e0"
# Le 13 categorie di rischio del README (Other compresa), con la grafia dei dati
# (train alla revisione fissata): «Guns and Illegal Weapons», «Sexual (minor)», ...
RISK_CATEGORIES = frozenset({
    "Hate/Identity Hate", "Sexual", "Violence", "Suicide and Self Harm", "Threat",
    "Sexual (minor)", "Guns and Illegal Weapons", "Controlled/Regulated Substances",
    "Criminal Planning/Confessions", "PII/Privacy", "Harassment", "Profanity", "Other",
})
_LABELS = tuple(f"labels_{i}" for i in range(5))


def _vote(label: str) -> str | None:
    parts = set(label.split(", "))
    if parts == {"Safe"}:
        return "NO_VIOLATION"
    if "Needs Caution" in parts:
        return "Needs Caution"
    return "VIOLATION" if parts <= RISK_CATEGORIES else None


def _votes(row: dict) -> list[str | None]:
    return [_vote(row[key]) for key in _LABELS if row.get(key) is not None]


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    if row["text_type"] != "user_message":
        return "label a livello di dialogo"
    if not row["text"].strip():
        return "testo vuoto"
    votes = _votes(row)
    if not votes or len(votes) != row["num_annotations"]:
        return "annotazioni mancanti"
    if None in votes:
        return "label fuori tassonomia"
    if "Needs Caution" in votes:
        return "Needs Caution"
    if len(set(votes)) > 1:
        return "annotatori in disaccordo"
    return None


def cases_from_row(row: dict) -> list[Case]:
    if drop_reason(row):
        return []
    return [Case(source=DATASET, source_id=row["id"], language="en", surface="prompt",
                 criterion=HARMFUL_REQUEST, evidence=row["text"].strip(),
                 verdict=_votes(row)[0])]


def needs_relabel(row: dict, case: Case) -> bool:
    """Label umane all'unanimità."""
    return False
