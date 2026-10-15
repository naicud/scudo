"""Caso etichettato: l'unità condivisa da conversione dei dataset, mix ed eval.

Un caso è un criterio semantico, l'evidenza da giudicare e il verdetto gold,
con fonte, lingua e superficie v1. La superficie dice da dove viene
l'evidenza e quale copertura v1 misura; `INSUFFICIENT` è un verdetto, non una
superficie.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from scudo.protocol import SEMANTIC_IDS, Verdict, training_row

Surface = Literal["prompt", "response", "injection_direct", "injection_indirect", "over_refusal"]
SURFACES: tuple[Surface, ...] = (
    "prompt", "response", "injection_direct", "injection_indirect", "over_refusal",
)
Language = Literal["it", "en", "mixed"]
LANGUAGES: tuple[Language, ...] = ("it", "en", "mixed")  # mixed: testo misto IT/EN


@dataclass(frozen=True)
class Case:
    source: str  # id del dataset, es. `nvidia/Nemotron-Safety-Guard-Dataset-v3`
    source_id: str  # id della riga nella fonte
    language: Language
    surface: Surface
    criterion: str
    evidence: str
    verdict: Verdict

    def __post_init__(self):
        if not self.source or not self.source_id:
            raise ValueError("fonte e id della riga sono obbligatori")
        if self.language not in LANGUAGES:
            raise ValueError(f"lingua {self.language!r} non in {LANGUAGES}")
        if self.surface not in SURFACES:
            raise ValueError(f"superficie {self.surface!r} non in {SURFACES}")
        if self.verdict not in SEMANTIC_IDS:
            raise ValueError(f"verdetto {self.verdict!r} fuori dal contratto tri-state")
        if not self.criterion.strip():
            raise ValueError("criterio vuoto")


def training_rows(cases: Sequence[Case], *, seed: int) -> list[dict]:
    """Righe decision-token con i metadati del caso per l'audit; il trainer usa solo prompt e completion."""
    rng = random.Random(seed)
    return [
        {
            **training_row(case.criterion, case.evidence, case.verdict, rng),
            "source": case.source,
            "source_id": case.source_id,
            "language": case.language,
            "surface": case.surface,
            "verdict": case.verdict,
        }
        for case in cases
    ]
