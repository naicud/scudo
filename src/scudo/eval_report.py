"""Report di eval per lingua, con tipo di prova e identità del modello dichiarati.

Un readout fallito resta un errore nella matrice di confusione: non diventa mai
`NO_VIOLATION` e conta contro l'accuratezza.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from scudo.artifacts import ArtifactIdentity
from scudo.protocol import FORMAT_VERSION, SEMANTIC_IDS, Readout, Verdict

ProofType = Literal["local", "live"]  # local: checkpoint in training; live: GGUF nel backend nativo
PROOF_TYPES: tuple[ProofType, ...] = ("local", "live")
ERROR = "ERROR"


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    language: str
    gold: Verdict
    readout: Readout | None
    error: str | None = None

    def __post_init__(self):
        if (self.readout is None) == (self.error is None):
            raise ValueError("un risultato ha un readout oppure un errore, mai entrambi o nessuno")

    @property
    def predicted(self) -> str:
        return self.readout.verdict if self.readout is not None else ERROR


def _metrics(results: Sequence[EvalResult]) -> dict:
    confusion = {gold: dict.fromkeys((*SEMANTIC_IDS, ERROR), 0) for gold in SEMANTIC_IDS}
    for item in results:
        confusion[item.gold][item.predicted] += 1
    correct = sum(confusion[key][key] for key in SEMANTIC_IDS)
    recall, precision = {}, {}
    for key in SEMANTIC_IDS:
        gold_total = sum(confusion[key].values())
        predicted_total = sum(confusion[gold][key] for gold in SEMANTIC_IDS)
        recall[key] = confusion[key][key] / gold_total if gold_total else None
        precision[key] = confusion[key][key] / predicted_total if predicted_total else None
    return {
        "cases": len(results),
        "errors": sum(item.readout is None for item in results),
        "accuracy": correct / len(results),
        "recall": recall,
        "precision": precision,
        "confusion": confusion,
    }


def build_report(
    results: Sequence[EvalResult],
    *,
    proof_type: ProofType,
    model: ArtifactIdentity,
    split: str,
    harness: str,
) -> dict:
    if proof_type not in PROOF_TYPES:
        raise ValueError(f"tipo di prova {proof_type!r} non in {PROOF_TYPES}")
    if proof_type == "live" and not model.filename.endswith(".gguf"):
        raise ValueError("la prova live riguarda il GGUF nel backend nativo")
    if not results:
        raise ValueError("nessun risultato da riportare")
    if any(item.gold not in SEMANTIC_IDS for item in results):
        raise ValueError("gold fuori dal contratto tri-state")
    by_language = defaultdict(list)
    for item in results:
        by_language[item.language].append(item)
    return {
        "proof_type": proof_type,
        "model": model.manifest_entry(),
        "format": FORMAT_VERSION,
        "split": split,
        "harness": harness,
        "overall": _metrics(results),
        "by_language": {lang: _metrics(items) for lang, items in sorted(by_language.items())},
    }


def compare_readouts(
    local: dict[str, Readout], live: dict[str, Readout], *, tolerance: float
) -> dict:
    """Score del checkpoint in training contro quelli del GGUF nel backend nativo.

    I logit si confrontano centrati sulla loro media: la quantizzazione e il runtime
    possono spostarli di una costante senza cambiare probabilità né margine.
    """
    missing = sorted(set(local) - set(live))
    shared = sorted(set(local) & set(live))
    diffs, agree = [], 0
    for case_id in shared:
        a, b = local[case_id].logits, live[case_id].logits
        mean_a, mean_b = sum(a) / len(SEMANTIC_IDS), sum(b) / len(SEMANTIC_IDS)
        diffs.append(max(abs((x - mean_a) - (y - mean_b)) for x, y in zip(a, b, strict=True)))
        agree += local[case_id].verdict == live[case_id].verdict
    worst = max(diffs, default=float("inf"))
    agreement = agree / len(shared) if shared else 0.0
    ok = not missing and shared and worst <= tolerance and agreement == 1.0
    return {
        "status": "PASS" if ok else "FAIL",
        "cases": len(shared),
        "missing": missing,
        "max_abs_logit_diff": worst,
        "verdict_agreement": agreement,
        "tolerance": tolerance,
    }
