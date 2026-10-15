"""Formato decision-token condiviso da training e serving.

Il prompt riproduce byte per byte il renderer `decision-token` del backend
nativo di open-guard (`open_guard.scoring.rendering`, 2ed1e92), con il testo del
criterio libero invece che scelto da un elenco del server. La risposta è una
sola lettera, A, B o C, letta sugli score delle opzioni senza generare testo.
"""

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FORMAT_VERSION = "scudo.decision-token/1"  # Cambiare con qualsiasi modifica ai byte del prompt.

Verdict = Literal["VIOLATION", "NO_VIOLATION", "INSUFFICIENT"]
SEMANTIC_IDS: tuple[Verdict, ...] = ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT")
SEMANTIC_DESCRIPTIONS: dict[str, str] = {
    "VIOLATION": "The stated criterion is met by the supplied authorized evidence.",
    "NO_VIOLATION": "The stated criterion is not met by the supplied authorized evidence.",
    "INSUFFICIENT": (
        "Essential evidence is missing or contradictory, so the stated criterion "
        "cannot be determined."
    ),
}
OPTION_LETTERS = "ABC"
DECISION_SUFFIX = "\nDecision:\n"


def _check_permutation(semantic_ids: Sequence[str]) -> None:
    if len(semantic_ids) != 3 or set(semantic_ids) != set(SEMANTIC_IDS):
        raise ValueError("le opzioni devono essere una permutazione esatta dei tre verdetti")


def render_prompt(criterion: str, evidence: str, semantic_ids: Sequence[str]) -> str:
    """Prompt esatto letto dal serving; il gold non vi compare mai."""
    _check_permutation(semantic_ids)
    if not criterion.strip():
        raise ValueError("criterio vuoto")
    options = {
        letter: {"semantic_id": key, "description": SEMANTIC_DESCRIPTIONS[key]}
        for letter, key in zip(OPTION_LETTERS, semantic_ids, strict=True)
    }
    payload = {"criterion": criterion, "untrusted_evidence": evidence, "options": options}
    return json.dumps(payload, ensure_ascii=False) + DECISION_SUFFIX


def training_row(criterion: str, evidence: str, verdict: Verdict, rng: random.Random) -> dict:
    """Una permutazione delle opzioni per caso; il gold sta solo nella lettera di completamento."""
    if verdict not in SEMANTIC_IDS:
        raise ValueError(f"verdetto sconosciuto: {verdict!r}")
    semantic_ids = list(SEMANTIC_IDS)
    rng.shuffle(semantic_ids)
    return {
        "format": FORMAT_VERSION,
        "prompt": render_prompt(criterion, evidence, semantic_ids),
        "completion": OPTION_LETTERS[semantic_ids.index(verdict)],
        "semantic_ids": semantic_ids,
    }


@dataclass(frozen=True)
class Readout:
    """Score delle opzioni nell'ordine canonico di SEMANTIC_IDS."""

    logits: tuple[float, float, float]
    probabilities: tuple[float, float, float]
    verdict: Verdict
    margin: float  # distanza fra i due logit più alti, come `option_margin` di open-guard


def read_options(letter_logits: Sequence[float], semantic_ids: Sequence[str]) -> Readout:
    """Riporta i logit di A, B, C all'ordine canonico; un readout malformato è un errore."""
    _check_permutation(semantic_ids)
    values = tuple(float(x) for x in letter_logits)
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError("readout malformato: servono tre logit finiti")
    by_id = dict(zip(semantic_ids, values, strict=True))
    logits = tuple(by_id[key] for key in SEMANTIC_IDS)
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = math.fsum(exps)
    probabilities = tuple(x / total for x in exps)
    first, second = sorted(logits, reverse=True)[:2]
    return Readout(
        logits=logits,  # type: ignore[arg-type]
        probabilities=probabilities,  # type: ignore[arg-type]
        verdict=SEMANTIC_IDS[logits.index(top)],
        margin=first - second,
    )
