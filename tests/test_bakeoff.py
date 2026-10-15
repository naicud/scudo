import json

import pytest

from scudo.artifacts import ArtifactIdentity
from scudo.bakeoff import (
    ECE_MAX_DELTA,
    GAP_MAX,
    MIN_GAIN,
    OVER_REFUSAL_MAX_DELTA,
    RESAMPLES,
    compare,
    from_rows,
    macro_f1,
    main,
    primary,
    result_rows,
)
from scudo.baselines import LabelReadout
from scudo.eval_report import EvalResult, build_report
from scudo.protocol import SEMANTIC_IDS, read_options

SHA = "0123456789abcdef0123456789abcdef01234567"


def res(case_id, lang, gold, pred, surface="prompt", conf=0.9):
    if pred is None:
        return EvalResult(case_id, lang, surface, gold, None, error="READOUT_MALFORMED")
    rest = (1 - conf) / 2
    probs = tuple(conf if key == pred else rest for key in SEMANTIC_IDS)
    return EvalResult(case_id, lang, surface, gold, LabelReadout(pred, pred, probs, None))


def bench(lang, pairs, prefix="b", surface="prompt", conf=0.9):
    return [res(f"{prefix}:{i}", lang, gold, pred, surface, conf)
            for i, (gold, pred) in enumerate(pairs)]


V, N, INS = "VIOLATION", "NO_VIOLATION", "INSUFFICIENT"


def test_macro_f1_matches_the_report_f1_when_every_class_is_predicted():
    results = bench("en", [(V, V), (V, N), (N, N), (N, N), (N, V), (INS, N), (V, None)])
    report = build_report(results, proof_type="local", model="m",
                          weights=[ArtifactIdentity("org/r", SHA, "w.safetensors", "0" * 64)],
                          scudo_sha=SHA, runtime={"name": "t", "version": "1"}, split="s",
                          harness="h")
    f1 = report["overall"]["f1"]
    assert macro_f1(results) == pytest.approx((f1[V] + f1[N]) / 2)


def test_macro_f1_uses_only_classes_with_gold_and_errors_are_misses():
    # Solo gold NO_VIOLATION (es. OR-Bench): conta F1 di NO_VIOLATION, i rifiuti sono falsi negativi.
    assert macro_f1(bench("en", [(N, N), (N, N)])) == 1.0
    assert macro_f1(bench("en", [(N, N), (N, V)])) == pytest.approx(2 / 3)
    # Un errore non diventa NO_VIOLATION: abbassa il recall come un verdetto sbagliato.
    assert macro_f1(bench("en", [(N, N), (N, None)])) == pytest.approx(2 / 3)
    assert macro_f1(bench("en", [(INS, N)])) is None


def test_primary_averages_benchmarks_per_language_then_the_two_languages():
    runs = {
        "it-bench": bench("it", [(V, V), (N, N)]),  # IT 1.0
        "en-good": bench("en", [(V, V), (N, N)]),  # EN 1.0
        "en-bad": bench("en", [(V, N), (N, V)]),  # EN 0.0
        "mixed": bench("mixed", [(V, N)]),  # fuori dalla regola IT/EN
    }
    out = primary(runs)
    assert out == {"it": 1.0, "en": 0.5, "score": 0.75, "gap": 0.5}
    with pytest.raises(ValueError, match="it"):
        primary({"en": bench("en", [(V, V)])})


def control_runs(n=200):
    pairs = [(V, V), (N, N), (V, N), (N, N)] * (n // 4)
    refusals = [(N, N)] * 40 + [(N, V)] * 10
    return {"it-bench": bench("it", pairs, "it"), "en-bench": bench("en", pairs, "en"),
            "xstest-it": bench("it", refusals, "xsit", "over_refusal"),
            "xstest": bench("en", refusals, "xs", "over_refusal")}


def test_identical_runs_have_zero_delta_and_do_not_replace_the_control():
    out = compare(control_runs(), control_runs(), resamples=200, seed=1)
    assert out["delta"]["primary"] == 0.0 and out["ci95"] == [0.0, 0.0]
    assert out["shared"] == 500 and out["resamples"] == 200
    assert out["verdict"] == "NON SUPERA IL CONTROLLO"


def test_clear_gain_with_ci_above_zero_replaces_the_control_pending_a_second_seed():
    better = control_runs()
    for name in ("it-bench", "en-bench"):
        better[name] = [res(r.case_id, r.language, r.gold, r.gold) for r in better[name]]
    out = compare(control_runs(), better, resamples=300, seed=1)
    assert out["delta"]["primary"] >= MIN_GAIN and out["ci95"][0] > 0
    assert out["verdict"].startswith("SOSTITUISCE IL CONTROLLO")
    assert out == compare(control_runs(), better, resamples=300, seed=1)  # seme fissato


def test_secondary_constraints_and_readout_errors_exclude_the_candidate():
    worse = control_runs()
    worse["xstest"] = bench("en", [(N, N)] * 30 + [(N, V)] * 20, "xs", "over_refusal")
    out = compare(control_runs(), worse, resamples=50, seed=1)
    assert out["delta"]["over_refusal"] == pytest.approx(0.1)
    assert out["delta"]["over_refusal"] > OVER_REFUSAL_MAX_DELTA
    assert out["verdict"].startswith("ESCLUSO") and "over-refusal" in out["verdict"]

    broken = control_runs()
    broken["en-bench"][0] = res("en:0", "en", V, None)
    out = compare(control_runs(), broken, resamples=50, seed=1)
    assert out["candidate"]["errors"] == 1 and "errori di readout" in out["verdict"]

    overconfident = {k: [res(r.case_id, r.language, r.gold, r.predicted, r.surface, conf=1.0)
                         for r in v] for k, v in control_runs().items()}
    out = compare(control_runs(), overconfident, resamples=50, seed=1)
    assert out["delta"]["ece"] > ECE_MAX_DELTA and "ECE" in out["verdict"]

    skewed = control_runs()
    skewed["it-bench"] = [res(r.case_id, r.language, r.gold, N) for r in skewed["it-bench"]]
    out = compare(control_runs(), skewed, resamples=50, seed=1)
    assert abs(out["candidate"]["gap"]) > GAP_MAX and "divario IT-EN" in out["verdict"]


def test_runs_must_cover_the_same_benchmarks():
    partial = control_runs()
    del partial["xstest"]
    with pytest.raises(ValueError, match="xstest"):
        compare(control_runs(), partial)


def test_result_rows_round_trip_keeps_verdict_probabilities_and_errors():
    readout = read_options([2.0, 0.5, -1.0], ["NO_VIOLATION", "VIOLATION", "INSUFFICIENT"])
    results = [EvalResult("a", "it", "prompt", V, readout),
               EvalResult("b", "en", "over_refusal", N, None, error="LETTER_NOT_SINGLE_TOKEN")]
    rows = json.loads(json.dumps(result_rows(results)))
    back = from_rows(rows)
    assert [r.predicted for r in back] == ["NO_VIOLATION", "ERROR"]
    assert back[0].readout.probabilities == pytest.approx(readout.probabilities)
    assert (back[1].error, back[1].surface) == ("LETTER_NOT_SINGLE_TOKEN", "over_refusal")


def test_cli_prints_a_verdict_per_candidate(tmp_path, capsys):
    def save(name, runs):
        folder = tmp_path / name
        folder.mkdir()
        rows = {bench_name: result_rows(results) for bench_name, results in runs.items()}
        (folder / "results.json").write_text(json.dumps(rows), encoding="utf-8")
        return str(folder)

    control = save("bakeoff-qwen35-4b", control_runs())
    candidate = save("bakeoff-gemma4-e4b", control_runs())
    main(["--control", control, "--candidate", candidate, "--resamples", "20"])
    out = json.loads(capsys.readouterr().out)
    assert out["control"] == "bakeoff-qwen35-4b" and out["seed"] == 3407
    assert out["candidates"]["bakeoff-gemma4-e4b"]["verdict"] == "NON SUPERA IL CONTROLLO"
    assert RESAMPLES == 1000
