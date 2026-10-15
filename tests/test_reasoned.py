import random

import pytest

from scudo.protocol import SEMANTIC_IDS, render_prompt
from scudo.reasoned import (
    ANSWER_PREFIX,
    MAX_NEW_TOKENS,
    REASONED_FORMAT_VERSION,
    REASONING_PREFIX,
    REASONING_STOP,
    ReasonedError,
    reasoned_training_row,
    render_reasoned_prompt,
    tokens_through_stop,
)

CRITERION = "Is the user's message unsafe?"
PERMUTATION = ("INSUFFICIENT", "VIOLATION", "NO_VIOLATION")


def test_frame_constants_are_the_versioned_bytes():
    assert REASONED_FORMAT_VERSION == "scudo.reasoned-decision/1"
    assert REASONING_PREFIX == "Reasoning:\n"
    assert REASONING_STOP == "\n</reasoning>\n"
    assert ANSWER_PREFIX == "Answer:\n"
    assert MAX_NEW_TOKENS == 512
    assert len(REASONING_STOP.encode()) <= 64  # limite della stop sequence nel bridge nativo


def test_prompt_is_the_decision_token_prompt_followed_by_the_reasoning_prefix():
    prompt = render_reasoned_prompt(CRITERION, "Ciao\nÈ facile", PERMUTATION)
    assert prompt == render_prompt(CRITERION, "Ciao\nÈ facile", PERMUTATION) + "Reasoning:\n"
    assert prompt.endswith('"}}}\nDecision:\nReasoning:\n')


def test_training_row_puts_gold_only_in_the_letter_after_the_answer_prefix():
    row = reasoned_training_row(CRITERION, "testo", "VIOLATION", "It asks for a weapon.",
                                random.Random(1))
    letter = "ABC"[row["semantic_ids"].index("VIOLATION")]
    assert row["format"] == REASONED_FORMAT_VERSION
    assert row["prompt"] == render_reasoned_prompt(CRITERION, "testo", row["semantic_ids"])
    assert row["completion"] == f"It asks for a weapon.\n</reasoning>\nAnswer:\n{letter}"
    assert sorted(row["semantic_ids"]) == sorted(SEMANTIC_IDS)


def test_training_rows_are_deterministic_for_a_seed():
    def rows(seed):
        rng = random.Random(seed)
        return [reasoned_training_row(CRITERION, f"t{i}", "NO_VIOLATION", "Benign.", rng)
                for i in range(20)]

    assert rows(7) == rows(7)
    assert len({tuple(row["semantic_ids"]) for row in rows(7)}) > 1


@pytest.mark.parametrize("reasoning", [
    "", "   ", " leading space", "trailing newline\n", "a\n</reasoning>\nb", "x</reasoning>",
    "nul\x00",
])
def test_invalid_reasoning_is_rejected(reasoning):
    with pytest.raises(ValueError):
        reasoned_training_row(CRITERION, "t", "VIOLATION", reasoning, random.Random(0))


def test_unknown_verdict_is_rejected():
    with pytest.raises(ValueError):
        reasoned_training_row(CRITERION, "t", "SAFE", "ok", random.Random(0))


def decode(tokens):
    return "".join(tokens)


def test_generation_ends_on_the_token_that_completes_the_stop():
    tokens = ["It ", "is ", "benign.", "\n</", "reason", "ing>", "\n"]
    assert tokens_through_stop(tokens, decode, budget=10) == 7


def test_tokens_after_the_stop_are_cut_like_the_native_bridge():
    tokens = ["Benign.", "\n</reasoning>\n", "Answer", ":\n", "C"]
    assert tokens_through_stop(tokens, decode, budget=10) == 2


def test_stop_completed_inside_a_longer_piece_counts_that_piece():
    assert tokens_through_stop(["Ok.", "\n</reasoning>\n\n"], decode, budget=10) == 2


def test_missing_stop_at_budget_is_truncation():
    with pytest.raises(ReasonedError) as error:
        tokens_through_stop(["word "] * 4, decode, budget=4)
    assert error.value.code == "REASONING_TRUNCATED"


def test_missing_stop_before_budget_is_end_of_generation_without_stop():
    with pytest.raises(ReasonedError) as error:
        tokens_through_stop(["Benign.", "\nAnswer:\nC"], decode, budget=4)
    assert error.value.code == "REASONING_NO_STOP"


def test_empty_reasoning_before_the_stop_is_malformed():
    with pytest.raises(ReasonedError) as error:
        tokens_through_stop([" \n</reasoning>\n"], decode, budget=4)
    assert error.value.code == "REASONING_EMPTY"


def test_more_tokens_than_budget_is_malformed():
    with pytest.raises(ReasonedError) as error:
        tokens_through_stop(["a", "\n</reasoning>\n"], decode, budget=1)
    assert error.value.code == "REASONING_OVER_BUDGET"
