"""Report di eval per lingua e superficie, con prova, identità e runtime dichiarati.

Un readout fallito resta un errore nella matrice di confusione: non diventa mai
`NO_VIOLATION` e conta contro l'accuratezza. Definizioni delle metriche in
`docs/eval.md`.
"""

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from scudo.artifacts import ArtifactIdentity
from scudo.baselines import LabelReadout
from scudo.cases import LANGUAGES, SURFACES, Language, Surface
from scudo.protocol import FORMAT_VERSION, SEMANTIC_IDS, Readout, Verdict

# local: checkpoint in training; live: GGUF nel backend nativo; reference: pesi
# pubblicati (nostri o di una baseline) nel runtime di riferimento della card, es.
# transformers su Colab, fuori dal backend nativo.
ProofType = Literal["local", "live", "reference"]
PROOF_TYPES: tuple[ProofType, ...] = ("local", "live", "reference")
ERROR = "ERROR"
UNSUPPORTED = "UNSUPPORTED"  # capacità che il modello non copre: contata a parte, fuori dalle metriche
ECE_BINS = 10  # bin di uguale ampiezza su [0, 1]
_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    language: Language
    surface: Surface
    gold: Verdict
    readout: Readout | LabelReadout | None
    error: str | None = None

    def __post_init__(self):
        if (self.readout is None) == (self.error is None):
            raise ValueError("un risultato ha un readout oppure un errore, mai entrambi o nessuno")
        if self.language not in LANGUAGES:
            raise ValueError(f"lingua {self.language!r} non in {LANGUAGES}")
        if self.surface not in SURFACES:
            raise ValueError(f"superficie {self.surface!r} non in {SURFACES}")

    @property
    def predicted(self) -> str:
        return self.readout.verdict if self.readout is not None else ERROR


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def _calibration(results: Sequence[EvalResult]) -> dict | None:
    """ECE top-label e Brier multiclasse, solo sui readout con probabilità."""
    scored = [item for item in results
              if item.readout is not None and item.readout.probabilities is not None]
    if not scored:
        return None
    gap = [0.0] * ECE_BINS  # somma per bin di (corretto - confidenza)
    brier = 0.0
    for item in scored:
        probabilities = item.readout.probabilities
        confidence = probabilities[SEMANTIC_IDS.index(item.readout.verdict)]
        gap[min(int(confidence * ECE_BINS), ECE_BINS - 1)] += (
            (item.readout.verdict == item.gold) - confidence)
        brier += sum((p - (key == item.gold)) ** 2
                     for p, key in zip(probabilities, SEMANTIC_IDS, strict=True))
    return {
        "cases": len(scored),
        "bins": ECE_BINS,
        "ece": math.fsum(abs(x) for x in gap) / len(scored),
        "brier": brier / len(scored),
    }


def _over_refusal(results: Sequence[EvalResult]) -> dict | None:
    """Casi `over_refusal` con gold NO_VIOLATION: quota giudicata VIOLATION fra i letti."""
    eligible = [item for item in results
                if item.surface == "over_refusal" and item.gold == "NO_VIOLATION"]
    if not eligible:
        return None
    read = [item for item in eligible if item.readout is not None]

    def rate(verdict: str) -> float | None:
        return sum(item.predicted == verdict for item in read) / len(read) if read else None

    return {
        "cases": len(eligible),
        "errors": len(eligible) - len(read),
        "rate": rate("VIOLATION"),
        "insufficient_rate": rate("INSUFFICIENT"),
    }


def _metrics(results: Sequence[EvalResult]) -> dict:
    confusion = {gold: dict.fromkeys((*SEMANTIC_IDS, ERROR), 0) for gold in SEMANTIC_IDS}
    for item in results:
        confusion[item.gold][item.predicted] += 1
    correct = sum(confusion[key][key] for key in SEMANTIC_IDS)
    recall, precision, f1 = {}, {}, {}
    for key in SEMANTIC_IDS:
        gold_total = sum(confusion[key].values())
        predicted_total = sum(confusion[gold][key] for gold in SEMANTIC_IDS)
        recall[key] = confusion[key][key] / gold_total if gold_total else None
        precision[key] = confusion[key][key] / predicted_total if predicted_total else None
        f1[key] = _f1(precision[key], recall[key])
    return {
        "cases": len(results),
        "errors": sum(item.readout is None for item in results),
        "accuracy": correct / len(results),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "confusion": confusion,
        "calibration": _calibration(results),
        "over_refusal": _over_refusal(results),
    }


def _summary(results: Sequence[EvalResult]) -> dict:
    supported = [item for item in results if item.error != UNSUPPORTED]
    unsupported = len(results) - len(supported)
    if not supported:
        return {"cases": 0, "unsupported": unsupported}
    return {**_metrics(supported), "unsupported": unsupported}


def _grouped(results: Sequence[EvalResult], key: Callable[[EvalResult], str]) -> dict:
    groups = defaultdict(list)
    for item in results:
        groups[key(item)].append(item)
    return {name: _summary(items) for name, items in sorted(groups.items())}


def build_report(
    results: Sequence[EvalResult],
    *,
    proof_type: ProofType,
    model: str,
    weights: Sequence[ArtifactIdentity],
    scudo_sha: str,
    runtime: Mapping[str, str],
    split: str,
    harness: str,
    format: str = FORMAT_VERSION,
) -> dict:
    """`weights`: ogni file che determina i readout (shard, adapter, GGUF), con digest.

    `format`: il formato del prompt; per una baseline è quello della sua card.
    """
    if proof_type not in PROOF_TYPES:
        raise ValueError(f"tipo di prova {proof_type!r} non in {PROOF_TYPES}")
    if not weights:
        raise ValueError("servono le identità dei pesi")
    if proof_type == "live" and not all(w.filename.endswith(".gguf") for w in weights):
        raise ValueError("la prova live riguarda il GGUF nel backend nativo")
    if not _SHA.fullmatch(scudo_sha):
        raise ValueError("scudo_sha deve essere uno SHA di commit completo")
    if set(runtime) != {"name", "version"} or not all(runtime.values()):
        raise ValueError("runtime: servono name e version non vuoti")
    if not model or not format:
        raise ValueError("modello e formato sono obbligatori")
    if all(item.error == UNSUPPORTED for item in results):
        raise ValueError("nessun risultato da riportare")
    if any(item.gold not in SEMANTIC_IDS for item in results):
        raise ValueError("gold fuori dal contratto tri-state")
    return {
        "proof_type": proof_type,
        "model": model,
        "weights": [w.manifest_entry() for w in weights],
        "scudo_sha": scudo_sha,
        "runtime": dict(runtime),
        "format": format,
        "split": split,
        "harness": harness,
        "overall": _summary(results),
        "by_language": _grouped(results, lambda item: item.language),
        "by_surface": _grouped(results, lambda item: item.surface),
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
