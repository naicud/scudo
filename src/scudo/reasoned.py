"""Formato reasoned-decision dell'Accurate: ragionamento limitato, poi score delle opzioni.

Il prompt è quello decision-token (stesso JSON di criterio, evidenza e opzioni,
stesso `Decision:`) seguito da `Reasoning:\\n`. Il modello genera un ragionamento
che termina con la stop sequence; il serving aggiunge `Answer:\\n` e legge i
logit di A, B, C in quella posizione con `protocol.read_options`. È il frame
`reasoning` del backend nativo di open-guard (`reasoning_prefix`,
`stop_sequence`, `answer_prefix`, `max_new_tokens`). Stop mancante, budget
esaurito o ragionamento vuoto sono errori: Scudo non emette mai un verdetto
per quei casi. Dettagli in `docs/protocol.md`.
"""

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

from scudo.protocol import OPTION_LETTERS, SEMANTIC_IDS, Verdict, render_prompt

# Cambiare con qualsiasi modifica ai byte del prompt, del frame o del completamento.
REASONED_FORMAT_VERSION = "scudo.reasoned-decision/1"
REASONING_PREFIX = "Reasoning:\n"
REASONING_STOP = "\n</reasoning>\n"
ANSWER_PREFIX = "Answer:\n"
MAX_NEW_TOKENS = 512  # budget del ragionamento, stop sequence compresa

T = TypeVar("T")


class ReasonedError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def check_reasoning(reasoning: str) -> str:
    """Ragionamento di training: non vuoto, senza spazi ai bordi né marcatori riservati."""
    if not reasoning.strip():
        raise ValueError("ragionamento vuoto")
    if reasoning != reasoning.strip():
        raise ValueError("ragionamento con spazi ai bordi")
    if "</reasoning>" in reasoning or "\x00" in reasoning:
        raise ValueError("ragionamento con marcatore riservato o NUL")
    return reasoning


def render_reasoned_prompt(criterion: str, evidence: str, semantic_ids: Sequence[str]) -> str:
    """Prompt decision-token più il prefisso del ragionamento; il gold non vi compare mai."""
    return render_prompt(criterion, evidence, semantic_ids) + REASONING_PREFIX


def reasoned_training_row(
    criterion: str, evidence: str, verdict: Verdict, reasoning: str, rng: random.Random
) -> dict:
    """Completamento: ragionamento, stop, prefisso della risposta, lettera del gold."""
    if verdict not in SEMANTIC_IDS:
        raise ValueError(f"verdetto sconosciuto: {verdict!r}")
    check_reasoning(reasoning)
    semantic_ids = list(SEMANTIC_IDS)
    rng.shuffle(semantic_ids)
    letter = OPTION_LETTERS[semantic_ids.index(verdict)]
    return {
        "format": REASONED_FORMAT_VERSION,
        "prompt": render_reasoned_prompt(criterion, evidence, semantic_ids),
        "completion": reasoning + REASONING_STOP + ANSWER_PREFIX + letter,
        "semantic_ids": semantic_ids,
    }


def tokens_through_stop(
    tokens: Sequence[T], decode: Callable[[Sequence[T]], str], *, budget: int
) -> int:
    """Token generati fino a quello che completa la stop sequence, come il bridge nativo.

    `decode` rende il testo senza token speciali. I token oltre lo stop non
    entrano nel readout. Errori: stop assente entro il budget
    (`REASONING_TRUNCATED`), fine della generazione prima dello stop
    (`REASONING_NO_STOP`), ragionamento vuoto (`REASONING_EMPTY`).
    """
    if len(tokens) > budget:
        raise ReasonedError("REASONING_OVER_BUDGET", f"{len(tokens)} token oltre {budget}")
    text = decode(tokens)
    if REASONING_STOP not in text:
        if len(tokens) == budget:
            raise ReasonedError("REASONING_TRUNCATED", f"stop assente in {budget} token")
        raise ReasonedError("REASONING_NO_STOP", "generazione finita prima dello stop")
    count = len(tokens)
    while count > 1 and REASONING_STOP in decode(tokens[:count - 1]):
        count -= 1
    if not decode(tokens[:count]).split(REASONING_STOP, 1)[0].strip():
        raise ReasonedError("REASONING_EMPTY", "nessun ragionamento prima dello stop")
    return count
