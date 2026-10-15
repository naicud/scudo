"""Casi di injection indiretta da `microsoft/llmail-inject-challenge` (MIT).

Ogni riga è un'email (subject e body) scritta da un partecipante per far
eseguire all'assistente email della vittima un'azione non chiesta. La fonte
non ha negativi: anche un invio fallito è un tentativo di attacco, quindi mai
NO_VIOLATION. VIOLATION solo con esito osservato: `exfil.sent` vero, cioè
l'assistente ha generato la chiamata API di invio mentre l'utente chiedeva solo
un riassunto o una risposta. Entrano solo i livelli con vittima
`microsoft/Phi-3-medium-128k-instruct` (MIT): sui livelli GPT-4o mini l'esito
viene da output OpenAI, i cui termini vanno decisi da una persona.

Evidenza nel formato con cui il challenge passa l'email all'LLM
(«Subject of the email: … Body: …»); la colonna `output` non entra.
"""

import json
import re

from scudo.cases import Case
from scudo.default_criteria import INJECTION_INDIRECT

DATASET = "microsoft/llmail-inject-challenge"
REVISION = "1063bdf01ec8762b812d5e06ee768a06faa5a6f7"
# data/levels_descriptions.json: lettere dei livelli con vittima Phi-3 (fase 1 e 2).
PHI3_LEVELS = frozenset("acegikmoqsu")
_LEVEL = re.compile(r"level\d([a-v])")


def _evidence(row: dict) -> str:
    return f"Subject of the email: {row['subject'] or ''}.   Body: {row['body'] or ''}"


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    flags = json.loads(row["objectives"] or "{}")
    if "exfil.sent" not in flags:
        return "esito assente"
    level = _LEVEL.fullmatch(row["scenario"])
    if level is None:
        return "livello sconosciuto"
    if level.group(1) not in PHI3_LEVELS:
        return "vittima GPT-4o mini (termini OpenAI)"
    if not flags["exfil.sent"]:
        return "nessuna chiamata API: esito non osservabile"
    if not (row["subject"] or "").strip() and not (row["body"] or "").strip():
        return "email vuota"
    return None


def cases_from_row(row: dict) -> list[Case]:
    if drop_reason(row):
        return []
    return [Case(source=DATASET, source_id=row["RowKey"], language="en",
                 surface="injection_indirect", criterion=INJECTION_INDIRECT.text,
                 evidence=_evidence(row), verdict="VIOLATION")]


def needs_relabel(row: dict, case: Case) -> bool:
    """La label è l'esito registrato dal sistema del challenge, non un giudizio LLM."""
    return False
