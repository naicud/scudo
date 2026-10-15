import pytest

from scudo.artifacts import ArtifactIdentity
from scudo.eval_report import EvalResult, build_report
from scudo.protocol import FORMAT_VERSION, read_options

CANON = ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT")
MODEL = ArtifactIdentity("naicud/scudo-fast-4b-GGUF", "0" * 40, "m.gguf", "a" * 64)


def result(language, gold, predicted):
    logits = tuple(3.0 if key == predicted else 0.0 for key in CANON)
    return EvalResult(case_id=f"{language}-{gold}-{predicted}", language=language, gold=gold,
                      readout=read_options(logits, CANON))


def failed(language, gold):
    return EvalResult(case_id=f"{language}-fail", language=language, gold=gold, readout=None,
                      error="READOUT_INVALID")


def build(results, proof="local"):
    return build_report(results, proof_type=proof, model=MODEL, split="nemotron-v3-heldout",
                        harness="scudo.eval/1")


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


def test_report_declares_proof_type_model_and_format():
    report = build([result("en", "VIOLATION", "VIOLATION")], proof="live")
    assert report["proof_type"] == "live"
    assert report["model"] == MODEL.manifest_entry()
    assert report["format"] == FORMAT_VERSION
    assert report["split"] == "nemotron-v3-heldout"


def test_failed_readout_is_an_error_never_a_no_violation():
    report = build([failed("it", "NO_VIOLATION"), result("it", "NO_VIOLATION", "NO_VIOLATION")])
    it = report["by_language"]["it"]
    assert it["errors"] == 1
    assert it["accuracy"] == pytest.approx(0.5)
    assert it["confusion"]["NO_VIOLATION"]["NO_VIOLATION"] == 1
    assert it["confusion"]["NO_VIOLATION"]["ERROR"] == 1


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
        EvalResult("x", "it", "VIOLATION", readout=None)
    with pytest.raises(ValueError):
        EvalResult("x", "it", "VIOLATION", readout=ok.readout, error="BOOM")


def test_live_proof_requires_a_gguf_identity():
    adapter = ArtifactIdentity("naicud/x", "0" * 40, "adapter_model.safetensors", "a" * 64)
    with pytest.raises(ValueError):
        build_report([result("en", "VIOLATION", "VIOLATION")], proof_type="live", model=adapter,
                     split="s", harness="h")
