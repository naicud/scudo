import math

import pytest

from scudo.teacher_margin import verdict_margin

CONTENT = '{"verdict": "VIOLATION", "reasoning": "x"}'


def tok(token, logprob, *alternatives):
    return {"token": token, "logprob": logprob,
            "top_logprobs": [{"token": token, "logprob": logprob}]
            + [{"token": t, "logprob": lp} for t, lp in alternatives]}


def logprobs(*decisive, split=("VI", "OLATION")):
    """Token del contenuto; il primo token del verdetto porta le alternative `decisive`."""
    head = [tok('{"', -0.01), tok("verdict", -0.01), tok('":', -0.01), tok(' "', -0.01)]
    tail = [tok(split[1], -0.01), tok('",', -0.01), tok(' "reasoning', -0.01), tok('": "x"}', -0.1)]
    return {"content": head + [tok(split[0], -0.1, *decisive)] + tail}


def test_margin_is_the_gap_to_the_best_competing_verdict_at_the_decisive_token():
    lp = logprobs(("NO", -2.5), ("IN", -4.0), ("V", -0.9))  # "V" resta VIOLATION: non compete
    assert verdict_margin(CONTENT, "VIOLATION", lp) == pytest.approx(2.4)


def test_token_with_the_opening_quote_is_matched_on_the_text_after_it():
    head = [tok('{"verdict":', -0.01)]
    decisive = tok(' "VI', -0.3, (' "NO', -1.3), (' "', -0.2), ('"IN', -0.1))
    tail = [tok('OLATION", "reasoning": "x"}', -0.01)]
    margin = verdict_margin(CONTENT, "VIOLATION", {"content": head + [decisive] + tail})
    assert margin == pytest.approx(1.0)  # ' "' non nomina un verdetto, '"IN' non ha lo spazio


def test_without_a_competitor_in_the_top_list_the_margin_is_a_lower_bound():
    lp = logprobs(("Hello", -7.0), ("V", -3.0))
    assert verdict_margin(CONTENT, "VIOLATION", lp) == pytest.approx(6.9)
    assert verdict_margin(CONTENT, "VIOLATION", logprobs()) is None  # nessuna alternativa


def test_reasoning_tokens_before_the_content_are_skipped():
    lp = logprobs(("NO", -1.1))
    lp["content"] = [tok("thinking about it. ", -0.5)] + lp["content"]
    assert verdict_margin(CONTENT, "VIOLATION", lp) == pytest.approx(1.0)


def test_sampled_runner_up_gives_a_negative_margin():
    lp = logprobs(("NO", -0.05))
    assert verdict_margin(CONTENT, "VIOLATION", lp) == pytest.approx(-0.05)


@pytest.mark.parametrize("lp", [None, {}, {"content": None}, {"content": [tok("altro", -1.0)]},
                                {"content": [{"token": "x"}]}, "rotto"])
def test_missing_or_mismatched_logprobs_give_no_margin(lp):
    assert verdict_margin(CONTENT, "VIOLATION", lp) is None


def test_verdict_must_be_the_one_in_the_content():
    assert verdict_margin(CONTENT, "NO_VIOLATION", logprobs(("NO", -2.0))) is None
    fenced = "```json\n" + CONTENT + "\n```"
    lp = logprobs(("NO", -2.0))
    lp["content"] = [tok("```json\n", -0.01)] + lp["content"] + [tok("\n```", -0.01)]
    assert verdict_margin(fenced, "VIOLATION", lp) == pytest.approx(1.9)
    assert math.isfinite(verdict_margin(fenced, "VIOLATION", lp))
