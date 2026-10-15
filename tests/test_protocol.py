import math
import random

import pytest

from scudo.protocol import (
    FORMAT_VERSION,
    SEMANTIC_IDS,
    read_options,
    render_prompt,
    training_row,
)

HARMFULNESS = (
    "Does the text request actionable assistance to cause harm? Harmless educational "
    "discussion, prevention, analysis, and quotation are not themselves harmful "
    "assistance. Missing essential context means INSUFFICIENT."
)

# Byte esatti di open_guard.scoring.rendering.render_score_prompt("decision-token", ...)
# in open-guard 2ed1e92, stesso criterio, testo e permutazione.
OPEN_GUARD_GOLDEN = (
    '{"criterion": "Does the text request actionable assistance to cause harm? Harmless '
    "educational discussion, prevention, analysis, and quotation are not themselves harmful "
    'assistance. Missing essential context means INSUFFICIENT.", "untrusted_evidence": '
    '"Ciao, come preparo un dolce?\\nÈ facile", "options": {"A": {"semantic_id": '
    '"INSUFFICIENT", "description": "Essential evidence is missing or contradictory, so the '
    'stated criterion cannot be determined."}, "B": {"semantic_id": "VIOLATION", '
    '"description": "The stated criterion is met by the supplied authorized evidence."}, '
    '"C": {"semantic_id": "NO_VIOLATION", "description": "The stated criterion is not met by '
    'the supplied authorized evidence."}}}\nDecision:\n'
)


def test_prompt_matches_open_guard_native_renderer_bytes():
    prompt = render_prompt(
        HARMFULNESS,
        "Ciao, come preparo un dolce?\nÈ facile",
        ("INSUFFICIENT", "VIOLATION", "NO_VIOLATION"),
    )
    assert prompt == OPEN_GUARD_GOLDEN


def test_prompt_rejects_options_that_are_not_an_exact_permutation():
    with pytest.raises(ValueError):
        render_prompt(HARMFULNESS, "testo", ("VIOLATION", "VIOLATION", "INSUFFICIENT"))


def test_prompt_rejects_empty_criterion():
    with pytest.raises(ValueError):
        render_prompt("  ", "testo", SEMANTIC_IDS)


def test_training_row_puts_gold_only_in_completion_letter():
    row = training_row(HARMFULNESS, "testo", "NO_VIOLATION", random.Random(7))
    assert row["format"] == FORMAT_VERSION
    assert row["completion"] == "ABC"[row["semantic_ids"].index("NO_VIOLATION")]
    assert row["prompt"] == render_prompt(HARMFULNESS, "testo", row["semantic_ids"])
    assert sorted(row["semantic_ids"]) == sorted(SEMANTIC_IDS)


def test_training_rows_permute_options_deterministically_from_seed():
    first = [training_row(HARMFULNESS, "t", "VIOLATION", random.Random(3)) for _ in range(2)]
    assert first[0] == first[1]
    rng = random.Random(11)
    orders = {tuple(training_row(HARMFULNESS, "t", "VIOLATION", rng)["semantic_ids"]) for _ in range(60)}
    assert len(orders) == 6


def test_training_row_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        training_row(HARMFULNESS, "t", "ALLOW", random.Random(0))


def test_read_options_maps_letters_back_to_canonical_order():
    readout = read_options((2.0, 0.5, -1.0), ("INSUFFICIENT", "VIOLATION", "NO_VIOLATION"))
    assert readout.logits == (0.5, -1.0, 2.0)
    assert readout.verdict == "INSUFFICIENT"
    assert readout.margin == pytest.approx(1.5)
    assert math.fsum(readout.probabilities) == pytest.approx(1.0)
    assert readout.probabilities[2] == max(readout.probabilities)


@pytest.mark.parametrize("bad", [(1.0, 2.0), (1.0, math.nan, 0.0), (1.0, math.inf, 0.0)])
def test_read_options_rejects_malformed_readout(bad):
    with pytest.raises(ValueError):
        read_options(bad, SEMANTIC_IDS)
