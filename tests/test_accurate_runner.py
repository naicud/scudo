import pytest

from scudo.benchmarks import sample
from scudo.reasoned import REASONING_STOP
from scudo.reasoned_runner import accurate_reports, accurate_scorer
from scudo.runner import RunnerError, letter_ids, run
from test_runner import SHA, FakeBackend, FakeTokenizer, case, loaded_of, verify_weights

REASONING = "It asks for a weapon."
HEADER = dict(proof_type="local", model="scudo-accurate", weights=verify_weights(),
              scudo_sha=SHA, runtime={"name": "unsloth", "version": "2026.10.5"},
              harness="test", format="scudo.reasoned-decision/1")


def scorer(generated_text, *, budget=512, letter_logits=(1.0, 3.0, 0.0)):
    tokenizer = FakeTokenizer()
    ids = letter_ids(tokenizer)
    backend = FakeBackend(next_logits=dict(zip(ids.values(), letter_logits)),
                          generated_text=generated_text, tokenizer=tokenizer)
    score, usage = accurate_scorer(tokenizer, backend, seed=7, budget=budget)
    return score, usage, backend, tokenizer


def test_readout_follows_reasoning_stop_and_answer_prefix():
    text = REASONING + REASONING_STOP + "Answer:\nB"  # i token dopo lo stop si tagliano
    score, usage, backend, tokenizer = scorer(text)
    item = case()
    readout = score(item)
    generate_prompt, readout_prompt = (tokenizer.decode(p) for p in backend.prompts)
    assert generate_prompt.endswith("\nDecision:\nReasoning:\n")
    assert readout_prompt == generate_prompt + REASONING + REASONING_STOP + "Answer:\n"
    assert usage[item]["generated_tokens"] == len(REASONING + REASONING_STOP)
    assert usage[item]["reasoning"] == REASONING + REASONING_STOP
    assert usage[item]["logits"] == list(readout.logits)
    assert readout.margin == pytest.approx(2.0)


@pytest.mark.parametrize("text,budget,code", [
    ("no stop here", 512, "REASONING_NO_STOP"),
    ("abcde", 5, "REASONING_TRUNCATED"),
    (" " + REASONING_STOP, 512, "REASONING_EMPTY"),
])
def test_missing_stop_is_an_error_result_never_a_verdict(text, budget, code):
    score, usage, backend, _ = scorer(text, budget=budget)
    item = case()
    [result] = run(loaded_of(item), score)
    assert result.readout is None and result.error == code
    assert usage[item] == {"generated_tokens": len(text)}
    assert len(backend.prompts) == 1  # nessuna lettura delle opzioni


def test_reports_carry_token_cost_of_their_own_cases():
    score, usage, _, _ = scorer(REASONING + REASONING_STOP)
    cases = (case(source_id="a"), case(source_id="b", verdict="NO_VIOLATION"))
    loaded = loaded_of(*cases, machine=(1,))
    results = run(loaded, score)
    reports = accurate_reports(loaded, results, usage, budget=512, dedup={"status": "PASS"},
                               **HEADER)
    n = len(REASONING + REASONING_STOP)
    assert reports["bench"]["token_cost"]["cases"] == 2
    assert reports["bench"]["token_cost"]["mean"] == n
    assert reports["bench/human-labels"]["token_cost"]["cases"] == 1
    assert reports["bench"]["format"] == "scudo.reasoned-decision/1"
    assert reports["bench"]["dedup_verification"] == {"status": "PASS"}
    assert "token_cost" not in accurate_reports(loaded, results, {}, budget=512, dedup={},
                                                **HEADER)["bench"]


def test_answer_prefix_must_end_on_a_token_boundary():
    tokenizer = FakeTokenizer(pieces=(":\nA",))
    with pytest.raises(RunnerError) as error:
        accurate_scorer(tokenizer, FakeBackend(), seed=0)
    assert error.value.code == "TOKENIZER_UNSUPPORTED"


def test_sample_is_deterministic_and_keeps_label_provenance():
    cases = tuple(case(source_id=str(i)) for i in range(10))
    loaded = loaded_of(*cases, machine=(0, 1, 2))
    first, again = sample(loaded, 4, seed=3), sample(loaded, 4, seed=3)
    assert first == again and len(first.cases) == 4
    assert first.machine_labelled <= set(first.case_ids)
    assert first.machine_labelled == loaded.machine_labelled & set(first.case_ids)
    assert dict(zip(first.case_ids, first.cases)).items() <= dict(
        zip(loaded.case_ids, loaded.cases)).items()
    assert first.rows == loaded.rows
