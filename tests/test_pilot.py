import json
from dataclasses import replace

import pytest

from scudo import aegis, nemotron, pilot
from scudo.budget import BudgetError, Ledger
from scudo.ingest import REPO_ROOT
from scudo.pilot_sample import allocate, build_sample
from scudo.teacher import parse_configs
from test_manifest import BENCH, TRAIN
from test_teacher import CONFIG

AEGIS = replace(TRAIN, name="aegis2-train", id=aegis.DATASET, surfaces=("prompt", "response"))
NEMO_TEST = replace(BENCH, name="nemotron3-test", id=nemotron.DATASET, languages=("it", "en"),
                    surfaces=("prompt", "response"), text_fields=("prompt", "response"))
NEMO_TRAIN = replace(TRAIN, name="nemotron3-train", id=nemotron.DATASET,
                     generator_terms="INCERTO")
ENTRIES = [AEGIS, NEMO_TRAIN, NEMO_TEST]
SOURCES = {"aegis2-train": ("en",), "nemotron3-test": ("it",)}
QWEN = {**CONFIG, "model_id": "Qwen/Qwen3.5-397B-A17B", "provider_model": "qwen3.5-397b",
        "api_key_env": "SCUDO_TEST_KEY_2"}
ENV = {"SCUDO_TEST_KEY": "k1-not-real", "SCUDO_TEST_KEY_2": "k2-not-real"}


def row(id_, prompt_label, response_label=None, source=None, language=None):
    out = {"id": id_, "reconstruction_id_if_redacted": None, "prompt": f"prompt {id_}",
           "response": f"response {id_}" if response_label else None,
           "prompt_label": prompt_label, "response_label": response_label,
           "violated_categories": "", "prompt_label_source": "human",
           "response_label_source": source}
    if language:
        out["language"] = language
    return out


def rows_for(prefix, language=None, count=24):
    labels = ("safe", "unsafe")
    sources = ("human", "llm_jury", "human", "refusal_data_augmentation")
    return [row(f"{prefix}{i}", labels[i % 2], labels[(i // 2) % 2], sources[(i // 4) % 4],
                language) for i in range(count)]


ROWS = {aegis.DATASET: rows_for("a"),
        nemotron.DATASET: rows_for("n", "it") + rows_for("e", "en") + [row("x", "safe", language="de")]}


def loader(entry, split):
    return list(ROWS[entry.id])


def sample(n=16, relabel_n=4, seed=0, rows_loader=loader):
    return build_sample(ENTRIES, rows_loader, sources=SOURCES, n=n, relabel_n=relabel_n, seed=seed)


def test_allocate_is_equal_and_redistributes_shortfalls():
    assert allocate({"a": 10, "b": 10}, 6) == {"a": 3, "b": 3}
    assert allocate({"a": 1, "b": 10, "c": 10}, 9) == {"a": 1, "b": 4, "c": 4}
    assert allocate({"a": 0, "b": 5}, 5) == {"a": 0, "b": 5}
    with pytest.raises(ValueError):
        allocate({"a": 2, "b": 2}, 5)


def test_gold_is_human_labelled_and_relabel_is_llm_jury_from_training_sources():
    built = sample()
    assert len(built["gold"]) == 16 and len(built["relabel"]) == 4
    assert {r["label_source"] for r in built["gold"]} == {"human"}
    assert {(r["label_source"], r["entry"], r["surface"]) for r in built["relabel"]} == {
        ("llm_jury", "aegis2-train", "response")}
    assert {(r["entry"], r["language"]) for r in built["gold"]} == {
        ("aegis2-train", "en"), ("nemotron3-test", "it")}


def test_gold_is_stratified_by_language_surface_and_verdict():
    strata = sample()["meta"]["gold_strata"]
    assert len(strata) == 8 and set(strata.values()) == {2}


def test_eval_only_rows_are_never_training_eligible():
    built = sample()
    for record in built["gold"]:
        assert record["training_eligible"] is (record["entry"] == "aegis2-train")


def test_sample_is_deterministic_and_order_independent():
    reverse = sample(rows_loader=lambda entry, split: list(reversed(loader(entry, split))))
    assert [r["key"] for r in sample()["gold"]] == [r["key"] for r in reverse["gold"]]
    assert [r["key"] for r in sample(seed=1)["gold"]] != [r["key"] for r in sample()["gold"]]


def test_blocked_source_is_reported_and_never_read():
    read = []

    def tracking(entry, split):
        read.append(entry.name)
        return loader(entry, split)

    built = build_sample(ENTRIES, tracking, sources={**SOURCES, "nemotron3-train": ("it",)},
                         n=16, relabel_n=4, seed=0)
    assert "nemotron3-train" not in read
    assert "nemotron3-train" in built["meta"]["blocked"]
    assert any("INCERTO" in problem for problem in built["meta"]["gate"]["nemotron3-train"])
    assert built["meta"]["languages_missing"] == ["mixed"]


def test_too_few_cases_fail_instead_of_filling_half():
    with pytest.raises(ValueError):
        sample(n=1000)


def answer(payload):
    user = json.loads(payload["messages"][-1]["content"])
    verdict = "VIOLATION" if user["untrusted_evidence"].endswith(("1", "3", "5", "7", "9")) else "NO_VIOLATION"
    if payload["model"] == "qwen3.5-397b" and user["untrusted_evidence"].startswith("response"):
        return 200, json.dumps({"choices": [{"message": {"content": "I'm sorry."},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 5}}).encode()
    content = json.dumps({"verdict": verdict, "reasoning": f"The text {user['untrusted_evidence']}."})
    return 200, json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()


class Transport:
    def __init__(self):
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        return answer(json.loads(payload))


def prepared(tmp_path, n=16, relabel_n=4):
    out = tmp_path / "pilot"
    pilot.write_sample(out, sample(n=n, relabel_n=relabel_n))
    configs = parse_configs({"teacher": [CONFIG, QWEN]})
    return out, configs, Ledger(tmp_path / "ledger.jsonl")


def run(out, configs, ledger, confirm, transport=None):
    return pilot.run(out, configs, ledger, confirm=confirm, transport=transport or Transport(),
                     environ=ENV, sleep=lambda s: None)


def test_output_inside_the_repository_is_refused(tmp_path):
    with pytest.raises(ValueError):
        pilot.write_sample(REPO_ROOT / "pilot", sample())


def test_dry_run_estimates_without_calls_and_run_needs_the_exact_amount(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    estimate = pilot.estimate(out, configs, ledger)
    assert estimate["items"] == 2 * 20 and set(estimate["teachers"]) == {c.model_id for c in configs}
    transport = Transport()
    with pytest.raises(BudgetError):
        run(out, configs, ledger, None, transport)
    with pytest.raises(BudgetError):
        run(out, configs, ledger, "0.01", transport)
    assert transport.calls == 0 and ledger.spent() == 0
    result = run(out, configs, ledger, estimate["confirm"], transport)
    assert result["done"] == 40 and result["stopped"] is None and transport.calls == 40
    assert ledger.spent() == pytest.approx(result["spent_usd"]) and ledger.spent() > 0
    outcomes = pilot.read_outcomes(out)
    assert len(outcomes) == 40 and all("k1-not-real" not in json.dumps(o) for o in outcomes)


def test_missing_key_refuses_before_spending(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    estimate = pilot.estimate(out, configs, ledger)
    with pytest.raises(ValueError, match="SCUDO_TEST_KEY_2"):
        pilot.run(out, configs, ledger, confirm=estimate["confirm"], transport=Transport(),
                  environ={"SCUDO_TEST_KEY": "k"}, sleep=lambda s: None)
    assert not ledger.records()


def test_rerun_resumes_only_missing_items(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    run(out, configs, ledger, pilot.estimate(out, configs, ledger)["confirm"])
    estimate = pilot.estimate(out, configs, ledger)
    assert estimate["items"] == 0
    transport = Transport()
    assert run(out, configs, ledger, estimate["confirm"], transport)["done"] == 0
    assert transport.calls == 0


def test_estimate_over_the_remaining_budget_is_refused(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    estimate = pilot.estimate(out, configs, ledger)
    ledger.append({"kind": "call", "run_id": "other", "teacher": "t", "key": "k",
                   "cost_usd": 150.0 - estimate["total_usd"] + 0.001})
    with pytest.raises(BudgetError, match="residuo"):
        run(out, configs, ledger, pilot.estimate(out, configs, ledger)["confirm"])


class Expensive(Transport):
    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        status, raw = answer(json.loads(payload))
        data = json.loads(raw)
        data["usage"]["prompt_tokens"] = 10_000_000
        return status, json.dumps(data).encode()


def test_hard_stop_when_actual_spend_reaches_the_authorised_amount(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    estimate = pilot.estimate(out, configs, ledger)
    transport = Expensive()
    result = run(out, configs, ledger, estimate["confirm"], transport)
    assert result["stopped"] and result["done"] == 1 and transport.calls == 1
    assert len(pilot.read_outcomes(out)) == 1
    assert ledger.spent() == pytest.approx(result["spent_usd"])


def test_new_sample_after_the_pilot_needs_volume_confirmation(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    run(out, configs, ledger, pilot.estimate(out, configs, ledger)["confirm"])
    ledger.append({"kind": "pilot_report", "sample_sha256": pilot.sample_sha(out)})
    other = tmp_path / "other"
    pilot.write_sample(other, sample(seed=7))
    with pytest.raises(BudgetError, match="conferma"):
        run(other, configs, ledger, pilot.estimate(other, configs, ledger)["confirm"])


def test_volume_sized_sample_needs_volume_confirmation(tmp_path, monkeypatch):
    out, configs, ledger = prepared(tmp_path)
    monkeypatch.setattr(pilot, "MAX_PILOT_CASES", 10)
    with pytest.raises(BudgetError, match="conferma"):
        run(out, configs, ledger, pilot.estimate(out, configs, ledger)["confirm"])


class Unauthorised(Transport):
    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        return 401, b'{"error": {"message": "invalid api key"}}'


def test_401_stops_the_run_without_charging_or_recording_an_outcome(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    transport = Unauthorised()
    result = run(out, configs, ledger, pilot.estimate(out, configs, ledger)["confirm"], transport)
    assert result["stopped"] and result["done"] == 0 and transport.calls == 1
    assert ledger.spent() == 0 and pilot.read_outcomes(out) == []
    assert pilot.estimate(out, configs, ledger)["items"] == 40  # si riprende da capo


class BadRequest(Transport):
    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        return 400, b'{"error": {"message": "flagged input"}}'


def test_rejected_request_is_an_error_outcome_that_costs_nothing(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    confirm = pilot.estimate(out, configs, ledger)["confirm"]
    assert run(out, configs, ledger, confirm, BadRequest())["done"] == 40
    assert ledger.spent() == 0
    assert {(o["status"], o["verdict"]) for o in pilot.read_outcomes(out)} == {("error", None)}


def toml(tables):
    def value(v):
        return json.dumps(v) if isinstance(v, str) else repr(v)
    return "\n".join("[[teacher]]\n" + "\n".join(f"{k} = {value(v)}" for k, v in t.items())
                     for t in tables) + "\n"


def test_cli_dry_run_then_refused_run(tmp_path, capsys, monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    out, _, _ = prepared(tmp_path)
    teachers = tmp_path / "teachers.toml"
    teachers.write_text(toml([CONFIG, QWEN]), encoding="utf-8")
    common = ["--out", str(out), "--teachers", str(teachers), "--ledger", str(tmp_path / "l.jsonl")]
    assert pilot.main(["estimate", *common]) == 0
    assert json.loads(capsys.readouterr().out)["items"] == 40
    assert pilot.main(["run", *common]) == 1
    assert "--confirm-spend" in capsys.readouterr().err
    assert not (out / "outcomes.jsonl").exists()
