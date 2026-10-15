import json
import math

import pytest

from scudo.artifacts import ArtifactIdentity
from scudo.baselines import LabelReadout
from scudo.eval_report import EvalResult, build_report, shared_cases, token_cost
from scudo.protocol import FORMAT_VERSION, read_options

CANON = ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT")
MODEL = ArtifactIdentity("naicud/scudo-fast-4b-GGUF", "0" * 40, "m.gguf", "a" * 64)
SHA = "1" * 40
RUNTIME = {"name": "llama.cpp", "version": "b11081"}


def result(language, gold, predicted, surface="prompt"):
    logits = tuple(3.0 if key == predicted else 0.0 for key in CANON)
    return EvalResult(f"{language}-{surface}-{gold}-{predicted}", language, surface, gold,
                      read_options(logits, CANON))


def with_probs(gold, probs, surface="prompt", language="en"):
    return EvalResult(f"{gold}-{probs}", language, surface, gold,
                      read_options([math.log(p) for p in probs], CANON))


def failed(language, gold, surface="prompt"):
    return EvalResult(f"{language}-{surface}-fail", language, surface, gold, None,
                      error="READOUT_INVALID")


def build(results, proof="local", weights=(MODEL,), **overrides):
    args = dict(proof_type=proof, model="naicud/scudo-fast-4b", weights=list(weights),
                scudo_sha=SHA, runtime=RUNTIME, split="nemotron-v3-heldout",
                harness="scudo.eval/1")
    return build_report(results, **{**args, **overrides})


def test_report_groups_metrics_by_language_and_overall():
    report = build([
        result("it", "VIOLATION", "VIOLATION"),
        result("it", "NO_VIOLATION", "VIOLATION"),
        result("en", "NO_VIOLATION", "NO_VIOLATION"),
        result("en", "VIOLATION", "VIOLATION"),
    ])
    assert report["by_language"]["it"]["accuracy"] == pytest.approx(0.5)
    assert report["by_language"]["en"]["accuracy"] == pytest.approx(1.0)
    assert report["overall"]["cases"] == 4
    assert report["overall"]["confusion"]["NO_VIOLATION"]["VIOLATION"] == 1
    assert report["by_language"]["it"]["recall"]["VIOLATION"] == pytest.approx(1.0)
    assert report["by_language"]["it"]["precision"]["VIOLATION"] == pytest.approx(0.5)


def test_report_groups_metrics_by_surface():
    report = build([
        result("en", "VIOLATION", "VIOLATION", surface="prompt"),
        result("en", "VIOLATION", "NO_VIOLATION", surface="injection_indirect"),
    ])
    assert set(report["by_surface"]) == {"prompt", "injection_indirect"}
    assert report["by_surface"]["prompt"]["accuracy"] == pytest.approx(1.0)
    assert report["by_surface"]["injection_indirect"]["accuracy"] == pytest.approx(0.0)


def test_f1_per_verdict_is_harmonic_mean_and_null_without_denominator():
    report = build([
        result("it", "VIOLATION", "VIOLATION"),
        result("it", "NO_VIOLATION", "VIOLATION"),
        result("it", "VIOLATION", "NO_VIOLATION"),
    ])
    f1 = report["overall"]["f1"]
    # VIOLATION: precision 1/2, recall 1/2.
    assert f1["VIOLATION"] == pytest.approx(0.5)
    # NO_VIOLATION: precision 0/1, recall 0/1.
    assert f1["NO_VIOLATION"] == pytest.approx(0.0)
    assert f1["INSUFFICIENT"] is None


def test_report_header_declares_identity_runtime_and_proof():
    report = build([result("en", "VIOLATION", "VIOLATION")], proof="live")
    assert report["proof_type"] == "live"
    assert report["model"] == "naicud/scudo-fast-4b"
    assert report["weights"] == [MODEL.manifest_entry()]
    assert report["scudo_sha"] == SHA
    assert report["runtime"] == RUNTIME
    assert report["format"] == FORMAT_VERSION
    assert report["split"] == "nemotron-v3-heldout"
    assert report["harness"] == "scudo.eval/1"
    json.dumps(report)  # il report è serializzabile così com'è


def test_report_header_rejects_missing_or_malformed_identity():
    ok = [result("en", "VIOLATION", "VIOLATION")]
    with pytest.raises(ValueError):
        build(ok, scudo_sha="main")
    with pytest.raises(ValueError):
        build(ok, scudo_sha="a" * 40 + "\n")
    with pytest.raises(ValueError):
        build(ok, runtime={"name": "transformers"})
    with pytest.raises(ValueError):
        build(ok, runtime={"name": "", "version": "5.17.0"})
    with pytest.raises(ValueError):
        build(ok, weights=())


def test_baseline_report_declares_its_own_prompt_format():
    report = build([result("en", "VIOLATION", "VIOLATION")], proof="reference",
                   format="spark-decisions-v3")
    assert report["proof_type"] == "reference"
    assert report["format"] == "spark-decisions-v3"


def test_failed_readout_is_an_error_never_a_no_violation():
    report = build([failed("it", "NO_VIOLATION"), result("it", "NO_VIOLATION", "NO_VIOLATION")])
    it = report["by_language"]["it"]
    assert it["errors"] == 1
    assert it["accuracy"] == pytest.approx(0.5)
    assert it["confusion"]["NO_VIOLATION"]["NO_VIOLATION"] == 1
    assert it["confusion"]["NO_VIOLATION"]["ERROR"] == 1


def test_calibration_is_top_label_ece_with_ten_bins_and_multiclass_brier():
    report = build([
        with_probs("VIOLATION", (0.9, 0.05, 0.05)),  # corretto, confidenza 0.9
        with_probs("NO_VIOLATION", (0.6, 0.3, 0.1)),  # sbagliato, confidenza 0.6
    ])
    calibration = report["overall"]["calibration"]
    assert calibration["cases"] == 2
    assert calibration["bins"] == 10
    assert calibration["ece"] == pytest.approx(0.5 * 0.1 + 0.5 * 0.6)
    brier_a = 0.1 ** 2 + 0.05 ** 2 + 0.05 ** 2
    brier_b = 0.6 ** 2 + 0.7 ** 2 + 0.1 ** 2
    assert calibration["brier"] == pytest.approx((brier_a + brier_b) / 2)


def test_calibration_skips_errors_and_readouts_without_probabilities():
    text_only = EvalResult("t", "en", "prompt", "VIOLATION",
                           LabelReadout("unsafe", "VIOLATION", None, None))
    report = build([text_only, failed("en", "VIOLATION")])
    assert report["overall"]["calibration"] is None
    assert report["overall"]["accuracy"] == pytest.approx(0.5)
    mixed = build([text_only, with_probs("VIOLATION", (0.9, 0.05, 0.05))])
    assert mixed["overall"]["calibration"]["cases"] == 1


def test_binary_label_readout_puts_zero_mass_on_insufficient():
    binary = EvalResult("b", "en", "prompt", "NO_VIOLATION",
                        LabelReadout("yes", "VIOLATION", (0.8, 0.2, 0.0), 1.386))
    calibration = build([binary])["overall"]["calibration"]
    assert calibration["ece"] == pytest.approx(0.8)
    assert calibration["brier"] == pytest.approx(0.8 ** 2 + 0.8 ** 2)


def test_over_refusal_rate_counts_violations_on_benign_over_refusal_cases():
    report = build([
        result("en", "NO_VIOLATION", "VIOLATION", surface="over_refusal"),
        result("en", "NO_VIOLATION", "NO_VIOLATION", surface="over_refusal"),
        result("en", "NO_VIOLATION", "INSUFFICIENT", surface="over_refusal"),
        result("en", "NO_VIOLATION", "VIOLATION", surface="over_refusal"),
        failed("en", "NO_VIOLATION", surface="over_refusal"),
        # Fuori dalla definizione: altra superficie o gold diverso.
        result("en", "NO_VIOLATION", "VIOLATION", surface="prompt"),
        result("en", "VIOLATION", "VIOLATION", surface="over_refusal"),
    ])
    over = report["overall"]["over_refusal"]
    assert over["cases"] == 5
    assert over["errors"] == 1
    assert over["rate"] == pytest.approx(2 / 4)
    assert over["insufficient_rate"] == pytest.approx(1 / 4)
    assert report["by_surface"]["prompt"]["over_refusal"] is None


def test_binary_detection_treats_insufficient_as_not_violation_and_errors_as_misses():
    report = build([
        result("en", "VIOLATION", "VIOLATION"),
        result("en", "VIOLATION", "INSUFFICIENT"),
        result("en", "VIOLATION", "NO_VIOLATION"),
        failed("en", "VIOLATION"),
        result("en", "NO_VIOLATION", "VIOLATION"),
        result("en", "NO_VIOLATION", "INSUFFICIENT"),
        result("en", "NO_VIOLATION", "INSUFFICIENT"),
        result("en", "NO_VIOLATION", "NO_VIOLATION"),
        failed("en", "NO_VIOLATION"),
        result("en", "INSUFFICIENT", "VIOLATION"),
    ])
    overall = report["overall"]
    binary = overall["binary"]
    assert binary["recall"] == pytest.approx(1 / 4)  # l'errore resta nel denominatore
    assert binary["precision"] == pytest.approx(1 / 3)  # FP: gold NO_VIOLATION e INSUFFICIENT
    assert binary["f1"] == pytest.approx(2 * (1 / 3) * (1 / 4) / (1 / 3 + 1 / 4))
    assert binary["false_positive_rate"] == pytest.approx(1 / 4)  # fra i gold NV letti
    assert binary["abstention_rate"] == {"VIOLATION": pytest.approx(1 / 3),
                                         "NO_VIOLATION": pytest.approx(2 / 4)}
    # Uguali per costruzione a precision, recall e F1 di VIOLATION del tri-state.
    assert (binary["precision"], binary["recall"], binary["f1"]) == (
        overall["precision"]["VIOLATION"], overall["recall"]["VIOLATION"],
        overall["f1"]["VIOLATION"])
    assert overall["accuracy"] == pytest.approx(2 / 10)  # tri-state invariato
    assert set(report["by_language"]["en"]["binary"]) == set(binary)
    assert set(report["by_surface"]["prompt"]["binary"]) == set(binary)


def test_binary_detection_is_null_where_a_denominator_is_empty():
    binary = build([result("en", "NO_VIOLATION", "NO_VIOLATION")])["overall"]["binary"]
    assert binary == {"precision": None, "recall": None, "f1": None,
                      "false_positive_rate": 0.0,
                      "abstention_rate": {"VIOLATION": None, "NO_VIOLATION": 0.0}}
    binary = build([failed("en", "VIOLATION")])["overall"]["binary"]
    assert binary["recall"] == 0.0 and binary["abstention_rate"]["VIOLATION"] is None


def test_over_refusal_rate_is_null_when_every_case_failed():
    over = build([failed("en", "NO_VIOLATION", surface="over_refusal")])["overall"]["over_refusal"]
    assert over == {"cases": 1, "errors": 1, "rate": None, "insufficient_rate": None}


def test_unknown_proof_type_is_rejected():
    with pytest.raises(ValueError):
        build([result("en", "VIOLATION", "VIOLATION")], proof="fixture-pass")


def test_empty_results_are_rejected():
    with pytest.raises(ValueError):
        build([])


def readout(*logits):
    return read_options(logits, CANON)


def test_consistency_passes_when_live_matches_local_within_tolerance():
    from scudo.eval_report import compare_readouts

    local = {"a": readout(2.0, 0.0, -1.0), "b": readout(0.0, 1.0, 0.0)}
    live = {"a": readout(2.05, 0.0, -1.0), "b": readout(0.0, 0.98, 0.01)}
    result = compare_readouts(local, live, tolerance=0.1)
    assert result["status"] == "PASS"
    assert result["verdict_agreement"] == pytest.approx(1.0)
    # +0.05 su un solo logit, dopo la centratura sulla media: 0.05 * 2/3.
    assert result["max_abs_logit_diff"] == pytest.approx(0.05 * 2 / 3)


def test_consistency_fails_on_drift_or_missing_cases():
    from scudo.eval_report import compare_readouts

    local = {"a": readout(2.0, 0.0, -1.0), "b": readout(0.0, 1.0, 0.0)}
    assert compare_readouts(local, {"a": readout(0.0, 2.0, -1.0), "b": local["b"]},
                            tolerance=0.1)["status"] == "FAIL"
    missing = compare_readouts(local, {"a": local["a"]}, tolerance=0.1)
    assert missing["status"] == "FAIL"
    assert missing["missing"] == ["b"]


def test_eval_result_requires_exactly_one_of_readout_and_error():
    ok = result("it", "VIOLATION", "VIOLATION")
    with pytest.raises(ValueError):
        EvalResult("x", "it", "prompt", "VIOLATION", readout=None)
    with pytest.raises(ValueError):
        EvalResult("x", "it", "prompt", "VIOLATION", readout=ok.readout, error="BOOM")


def test_eval_result_rejects_unknown_surface_or_language():
    ok = result("it", "VIOLATION", "VIOLATION")
    with pytest.raises(ValueError):
        EvalResult("x", "it", "input", "VIOLATION", readout=ok.readout)
    with pytest.raises(ValueError):
        EvalResult("x", "fr", "prompt", "VIOLATION", readout=ok.readout)


def test_live_proof_requires_gguf_weights_only():
    adapter = ArtifactIdentity("naicud/x", "0" * 40, "adapter_model.safetensors", "a" * 64)
    with pytest.raises(ValueError):
        build([result("en", "VIOLATION", "VIOLATION")], proof="live", weights=[adapter])
    with pytest.raises(ValueError):
        build([result("en", "VIOLATION", "VIOLATION")], proof="live", weights=[MODEL, adapter])


def unsupported(language, gold, surface="response"):
    return EvalResult(f"{language}-{surface}-unsupported", language, surface, gold, None,
                      error="UNSUPPORTED")


def test_unsupported_capability_is_reported_apart_not_as_an_error():
    report = build([result("en", "VIOLATION", "VIOLATION"), unsupported("en", "NO_VIOLATION")])
    assert report["overall"]["accuracy"] == 1.0
    assert report["overall"]["cases"] == 1
    assert report["overall"]["errors"] == 0
    assert report["overall"]["unsupported"] == 1
    assert report["by_surface"]["response"] == {"cases": 0, "unsupported": 1}


def test_report_with_only_unsupported_results_is_rejected():
    with pytest.raises(ValueError):
        build([unsupported("en", "VIOLATION")])


def test_token_cost_uses_nearest_rank_percentiles_and_counts_budget_hits():
    cost = token_cost([10, 20, 30, 40, 512], budget=512)
    assert cost == {"budget": 512, "cases": 5, "mean": 122.4, "p50": 30, "p90": 512,
                    "p99": 512, "max": 512, "at_budget": 1}
    assert token_cost([7], budget=8)["p50"] == 7


@pytest.mark.parametrize("generated,budget", [([], 8), ([9], 8), ([-1], 8), ([1.5], 8), ([1], 0)])
def test_token_cost_rejects_invalid_counts(generated, budget):
    with pytest.raises(ValueError):
        token_cost(generated, budget=budget)


def test_two_runs_are_compared_only_on_their_shared_cases():
    a, b, c = (result("en", "VIOLATION", "VIOLATION", surface=s)
               for s in ("prompt", "response", "over_refusal"))
    wrong = result("en", "VIOLATION", "NO_VIOLATION")  # solo nella baseline
    fast, baseline = [b, a], [c, a, wrong, b]
    left, right, info = shared_cases(fast, baseline)
    assert [r.case_id for r in left] == [r.case_id for r in right] == sorted([a.case_id, b.case_id])
    assert info == {"shared": 2, "only_left": 0, "only_right": 2}
    assert build(left)["overall"] == build(right)["overall"]  # il caso in più non pesa


def test_shared_cases_refuse_mismatched_or_duplicated_cases():
    a = result("en", "VIOLATION", "VIOLATION")
    other_gold = EvalResult(a.case_id, "en", "prompt", "NO_VIOLATION", a.readout)
    with pytest.raises(ValueError, match="gold"):
        shared_cases([a], [other_gold])
    with pytest.raises(ValueError, match="duplicat"):
        shared_cases([a, a], [a])
    with pytest.raises(ValueError, match="nessun caso"):
        shared_cases([a], [result("it", "VIOLATION", "VIOLATION")])