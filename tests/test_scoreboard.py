import json

import pytest

from scudo import scoreboard
from scudo.scoreboard import ScoreboardError

SHA = "a" * 40


def block(cases=10, f1=0.8):
    return {"cases": cases, "errors": 1, "accuracy": 0.7, "unsupported": 0,
            "binary": {"f1": f1, "false_positive_rate": 0.1,
                       "abstention_rate": {"VIOLATION": 0.0, "NO_VIOLATION": 0.2}},
            "calibration": {"ece": 0.05}}


def report(model, split, *, sha=SHA, languages=("en",), revision="r1"):
    return {"proof_type": "local", "model": model, "scudo_sha": sha, "split": split,
            "format": "scudo.decision-token/1", "harness": "scudo.eval_notebook",
            "runtime": {"name": "unsloth", "version": "1"},
            "weights": [{"repo": "x/y", "revision": revision, "filename": "w", "sha256": "0" * 64}],
            "dedup_verification": {"status": "PASS"},
            "overall": block(), "by_language": {lang: block(5) for lang in languages},
            "by_surface": {}, "benchmark": {}}


def write(root, model, split, **kwargs):
    folder = root / model
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{split}.json").write_text(json.dumps(report(model, split, **kwargs)))
    (folder / "results.json").write_text(json.dumps({split: [{"case_id": "segreto"}]}))


def test_collect_keeps_aggregates_and_identity_never_cases(tmp_path):
    write(tmp_path, "fast", "xstest")
    write(tmp_path, "fast", "nemotron3-test", languages=("en", "it"))
    [entry] = scoreboard.collect(tmp_path, "2026-10-16")
    assert entry["scudo_sha"] == SHA and entry["on"] == "2026-10-16"
    assert entry["weights"][0]["sha256"] == "0" * 64 and entry["dedup"] == "PASS"
    assert set(entry["benchmarks"]["xstest"]) == {"all"}  # una lingua: niente duplicato
    assert set(entry["benchmarks"]["nemotron3-test"]) == {"all", "en", "it"}
    assert entry["benchmarks"]["xstest"]["all"]["abstention_no_violation"] == 0.2
    assert "segreto" not in json.dumps(entry)


def test_reports_with_different_identity_are_refused(tmp_path):
    write(tmp_path, "fast", "xstest")
    write(tmp_path, "fast", "jbb", revision="r2")
    with pytest.raises(ScoreboardError, match="identità"):
        scoreboard.collect(tmp_path, "2026-10-16")


def test_empty_folders_are_refused(tmp_path):
    with pytest.raises(ScoreboardError, match="nessuna cartella"):
        scoreboard.collect(tmp_path, "2026-10-16")
    (tmp_path / "fast").mkdir()
    with pytest.raises(ScoreboardError, match="nessun report"):
        scoreboard.collect(tmp_path, "2026-10-16")


def test_update_appends_runs_and_replaces_a_repeated_one(tmp_path):
    out_json, out_md = tmp_path / "s.json", tmp_path / "s.md"
    first, second = tmp_path / "e1", tmp_path / "e2"
    write(first, "fast", "xstest")
    write(second, "fast", "xstest", sha="b" * 40)
    scoreboard.update(first, "2026-10-15", out_json, out_md)
    scoreboard.update(second, "2026-10-16", out_json, out_md)
    entries = scoreboard.update(first, "2026-10-15", out_json, out_md)  # ripetuto: sostituito
    assert [e["scudo_sha"] for e in entries] == [SHA, "b" * 40]
    assert json.loads(out_json.read_text())["runs"] == entries
    md = out_md.read_text()
    assert md.index("bbbbbbb") < md.index("aaaaaaa")  # run più recente in alto
    assert "| xstest | all | 0.800 / 0.100 / 0.050 |" in md


def test_render_marks_benchmarks_a_model_did_not_run(tmp_path):
    write(tmp_path, "fast", "xstest")
    write(tmp_path, "guard", "jbb")
    md = scoreboard.render(scoreboard.collect(tmp_path, "2026-10-16"))
    assert "| jbb | all | non valutato | 0.800 / 0.100 / 0.050 |" in md
