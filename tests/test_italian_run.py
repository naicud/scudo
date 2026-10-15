import json
from dataclasses import replace

import pytest

from scudo import aegis, budget, italian_run
from scudo.budget import BudgetError, Ledger
from scudo.ingest import REPO_ROOT
from scudo.teacher import parse_configs
from test_manifest import BENCH, TRAIN
from test_teacher import CONFIG

AEGIS = replace(TRAIN, name="aegis2-train", id=aegis.DATASET)
BENCH_IT = replace(BENCH, name="bench-it", id="org/bench-it", text_fields=("prompt",))
ENTRIES = [AEGIS, BENCH_IT]
ENV = {"SCUDO_TEST_KEY": "k-not-real"}
TEACHER = "openai/gpt-oss-120b"


TOPICS = ("baking sourdough bread at home", "renewing an expired passport abroad",
          "training a stubborn puppy to sit", "repairing a leaking kitchen faucet",
          "choosing running shoes for flat feet", "planting tomatoes on a small balcony")


def source(i, unsafe):
    word = "unsafe" if unsafe else "harmless"
    return {"source": aegis.DATASET, "source_id": f"r{i}", "language": "en", "surface": "prompt",
            "criterion": aegis.HARMFUL_REQUEST,
            "evidence": f"Question {i} about {TOPICS[i]} is {word}",
            "verdict": "VIOLATION" if unsafe else "NO_VIOLATION", "split": "train",
            "needs_relabel": False}


RECORDS = [source(i, i % 2 == 0) for i in range(6)]


def reply(content, prompt_tokens=100, completion_tokens=20):
    return 200, json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": prompt_tokens,
                                      "completion_tokens": completion_tokens}}).encode()


class Transport:
    """Teacher finto: traduce con un prefisso, genera testo col target, giudica dalle parole."""

    def __init__(self, status=200, usage=(100, 20)):
        self.calls, self.status, self.usage = 0, status, usage

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        if self.status != 200:
            return self.status, b'{"error": {"message": "no"}}'
        system, user = (m["content"] for m in json.loads(payload)["messages"])
        data = json.loads(user)
        if "to Italian" in system:
            return reply(json.dumps({"translation": "[it] " + data["text"]}), *self.usage)
        if "to English" in system:
            return reply(json.dumps({"translation": data["text"].removeprefix("[it] ")}),
                         *self.usage)
        if "evaluation data" in system:
            text = (f"Testo {data['variant']} {data['target_verdict']} {data['topic']} "
                    f"{data['language'][:7]}")
            return reply(json.dumps({"evidence": text}), *self.usage)
        evidence = data["untrusted_evidence"]
        verdict = ("VIOLATION" if "unsafe" in evidence or " VIOLATION" in evidence
                   else "NO_VIOLATION")
        if "Question 4" in evidence:  # il teacher dissente dal gold
            verdict = "NO_VIOLATION"
        return reply(json.dumps({"verdict": verdict, "reasoning": "Motivo."}), *self.usage)


def setup(tmp_path, *, translate_n=6, native_n=4, mixed_n=2, amount=10.0, confirm=True):
    out = tmp_path / "it"
    plan = italian_run.make_plan(RECORDS, ENTRIES, translate_n=translate_n, native_n=native_n,
                                 mixed_n=mixed_n, seed=0, threshold=0.5)
    italian_run.write_plan(out, plan)
    ledger = Ledger(tmp_path / "spesa.jsonl")
    if confirm:
        budget.confirm_volume(ledger, teacher=TEACHER, amount_usd=amount, by="Mario Rossi",
                              rationale="pilot riportato")
    [cfg] = parse_configs({"teacher": [CONFIG]})
    return out, cfg, ledger


def run(out, cfg, ledger, confirm, transport=None):
    return italian_run.run(out, cfg, ledger, confirm=confirm, transport=transport or Transport(),
                           environ=ENV, sleep=lambda s: None)


def test_plan_is_deterministic_and_records_eligibility(tmp_path):
    a = italian_run.make_plan(RECORDS, ENTRIES, translate_n=3, native_n=2, mixed_n=1, seed=0,
                              threshold=0.5)
    b = italian_run.make_plan(list(reversed(RECORDS)), ENTRIES, translate_n=3, native_n=2,
                              mixed_n=1, seed=0, threshold=0.5)
    assert a == b
    kinds = [(t["kind"], t["language"]) for t in a["tasks"]]
    assert kinds.count(("translate", "it")) == 3 and kinds.count(("generate", "it")) == 2
    assert kinds.count(("generate", "mixed")) == 1
    assert a["meta"]["translation"]["sources"]["aegis2-train"]["eligible"]
    assert a["meta"]["threshold"] == 0.5


def test_plan_fails_instead_of_translating_fewer_cases(tmp_path):
    with pytest.raises(ValueError, match="traducibili"):
        italian_run.make_plan(RECORDS, ENTRIES, translate_n=7, native_n=0, mixed_n=0, seed=0,
                              threshold=0.5)


def test_plan_output_inside_the_repository_is_refused():
    plan = italian_run.make_plan(RECORDS, ENTRIES, translate_n=1, native_n=0, mixed_n=0, seed=0,
                                 threshold=0.5)
    with pytest.raises(ValueError):
        italian_run.write_plan(REPO_ROOT / "it", plan)


def test_run_refuses_without_the_human_volume_confirmation(tmp_path):
    out, cfg, ledger = setup(tmp_path, confirm=False)
    transport = Transport()
    estimate = italian_run.estimate(out, cfg, ledger)
    with pytest.raises(BudgetError, match="conferma umana"):
        run(out, cfg, ledger, estimate["confirm"], transport)
    assert transport.calls == 0 and ledger.spent() == 0


def test_run_refuses_a_teacher_other_than_the_confirmed_one(tmp_path):
    out, cfg, ledger = setup(tmp_path)
    other = replace(cfg, model_id="Qwen/Qwen3.5-397B-A17B")
    with pytest.raises(BudgetError, match="confermato"):
        run(out, other, ledger, italian_run.estimate(out, other, ledger)["confirm"])


def test_run_refuses_beyond_the_confirmed_volume_amount(tmp_path):
    out, cfg, ledger = setup(tmp_path, amount=0.01)
    with pytest.raises(BudgetError, match="volume"):
        run(out, cfg, ledger, italian_run.estimate(out, cfg, ledger)["confirm"])


def test_estimate_then_exact_confirmation_runs_and_counts_rejections(tmp_path):
    out, cfg, ledger = setup(tmp_path)
    transport = Transport()
    estimate = italian_run.estimate(out, cfg, ledger)
    assert estimate["items"] == 12 and estimate["calls"] == 6 * 3 + 6 * 2
    assert transport.calls == 0
    with pytest.raises(BudgetError):
        run(out, cfg, ledger, "0.01", transport)
    assert transport.calls == 0
    result = run(out, cfg, ledger, estimate["confirm"], transport)
    assert result["done"] == 12 and result["stopped"] is None and transport.calls == 30
    summary = result["summary"]
    assert summary["accepted"] == {"translated/it": 5, "generated/it": 4, "generated/mixed": 2}
    assert summary["rejected"] == {"translate": {"label_disagrees": 1}}
    calls = ledger.of_kind("call")
    assert len(calls) == 30 and {c["run_id"] for c in calls} == {result["run_id"]}
    assert result["run_id"].startswith("it-")
    assert summary["cost_usd"] == pytest.approx(ledger.spent())
    # La ripresa salta gli item fatti, con una nuova stima.
    again = italian_run.estimate(out, cfg, ledger)
    assert again["items"] == 0
    assert run(out, cfg, ledger, again["confirm"], transport)["done"] == 0


def test_hard_stop_before_exceeding_the_authorised_amount(tmp_path):
    out, cfg, ledger = setup(tmp_path, translate_n=2, native_n=0, mixed_n=0)
    estimate = italian_run.estimate(out, cfg, ledger)
    transport = Transport(usage=(2_000_000, 1000))  # ingresso reale oltre la stima
    result = run(out, cfg, ledger, estimate["confirm"], transport)
    assert result["stopped"] and result["done"] < 2
    assert ledger.spent() > 0


def test_unauthorised_key_stops_without_recording_a_result(tmp_path):
    out, cfg, ledger = setup(tmp_path, translate_n=1, native_n=0, mixed_n=0)
    transport = Transport(status=401)
    result = run(out, cfg, ledger, italian_run.estimate(out, cfg, ledger)["confirm"], transport)
    assert result["stopped"] and result["done"] == 0 and transport.calls == 1
    assert italian_run.read_results(out) == []


def test_dedup_removes_benchmark_matches_and_batch_duplicates(tmp_path):
    out, cfg, ledger = setup(tmp_path)
    run(out, cfg, ledger, italian_run.estimate(out, cfg, ledger)["confirm"])
    accepted = [r for r in italian_run.read_results(out) if r["status"] == "accepted"]
    translated = [r for r in accepted if r["provenance"]["kind"] == "translated"]
    generated = next(r for r in accepted if r["provenance"]["kind"] == "generated")
    with (out / italian_run.RESULTS).open("a", encoding="utf-8") as handle:  # stesso testo
        handle.write(json.dumps({**generated, "key": "gen:copy"}) + "\n")
    bench_rows = {"test": [{"prompt": translated[0]["case"]["evidence"]},  # IT tradotto
                           {"prompt": translated[1]["provenance"]["original_evidence"]}]}

    def loader(entry, split):
        assert entry.name == "bench-it"
        return bench_rows[split]

    report = italian_run.dedup(out, ENTRIES, loader)
    removed = report["removed"]
    assert removed.pop("duplicate_in_batch") == 1 and removed.pop("exact:bench-it") >= 1
    assert sum(removed.values(), 1) == 2  # il resto è near-duplicate dello stesso benchmark
    assert set(removed) <= {"near:bench-it"}
    assert report["kept"] == len(accepted) - 2
    pool = italian_run.read_pool(out)
    assert len(pool) == report["kept"] and {p["origin"] for p in pool} == {"translated",
                                                                            "generated"}
    assert all(p["key"] and p["evidence"] and p["provenance"] for p in pool)
    attributions = (out / "ATTRIBUTIONS-it.md").read_text(encoding="utf-8")
    assert "Demo" in attributions and TEACHER in attributions
