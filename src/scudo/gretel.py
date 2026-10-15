"""Casi da `gretelai/gretel-safety-alignment-en-v1` (Apache-2.0), interamente sintetico.

Per costruzione (README) `prompt` contiene una richiesta non sicura, esplicita
o sottile, `response` è la risposta iniziale e `safe_response` la sua revisione
sicura. Una riga dà:

- il prompt come VIOLATION sul criterio di Aegis 2.0 per le richieste;
- `response` come VIOLATION e `safe_response` come NO_VIOLATION sul criterio
  per le risposte, solo se la probabilità di danno di
  `ibm-granite/granite-guardian-3.0-2b` (Apache-2.0) concorda con il ruolo
  della colonna (> 0,5 dannosa, < 0,5 sicura); altrimenti la risposta resta
  fuori con il motivo.

I punteggi `judge_*` non entrano: il README non nomina il modello giudice. Ogni
label viene da modelli (progetto del generatore o Granite Guardian), non da
persone: `needs_relabel` le marca tutte per la giuria di teacher.
"""

from collections import Counter

from scudo.aegis import HARMFUL_REQUEST, HARMFUL_RESPONSE
from scudo.cases import Case

DATASET = "gretelai/gretel-safety-alignment-en-v1"
REVISION = "7f220b493d7027919d9d4489b43ca2f75fb6f718"
THRESHOLD = 0.5  # soglia della probabilità di rischio di Granite Guardian
# colonna -> (colonna della probabilità, verdetto atteso per il ruolo)
RESPONSES = {
    "response": ("response_probability_of_harm", "VIOLATION"),
    "safe_response": ("safe_response_probability_of_harm", "NO_VIOLATION"),
}


def _guardian(probability) -> str | None:
    if probability is None or probability == THRESHOLD:
        return None
    return "VIOLATION" if probability > THRESHOLD else "NO_VIOLATION"


def convert_row(row: dict) -> tuple[list[Case], Counter]:
    """Casi della riga e scarti per motivo (riga intera o singola risposta)."""
    if not row["prompt"].strip():
        return [], Counter({"testo vuoto": 1})

    def case(part: str, surface: str, criterion: str, verdict: str) -> Case:
        return Case(source=DATASET, source_id=f"{row['id']}:{part}", language="en",
                    surface=surface, criterion=criterion, evidence=row[part], verdict=verdict)

    cases = [case("prompt", "prompt", HARMFUL_REQUEST, "VIOLATION")]
    dropped = Counter()
    for column, (probability, role) in RESPONSES.items():
        if not (row.get(column) or "").strip():
            dropped[f"{column}: testo vuoto"] += 1
        elif _guardian(row.get(probability)) != role:
            dropped[f"{column}: Granite Guardian non concorde"] += 1
        else:
            cases.append(case(column, "response", HARMFUL_RESPONSE, role))
    return cases, dropped


def needs_relabel(row: dict, case: Case) -> bool:
    """Ogni label Gretel è prodotta da modelli: non è gold prima della giuria."""
    return True
