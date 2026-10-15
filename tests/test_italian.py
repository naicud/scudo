import importlib.util
import json

import pytest

from scudo import italian
from test_italian_run import ENTRIES, RECORDS
from test_teacher import CONFIG


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def teachers(tmp_path):
    lines = ["[[teacher]]"] + [f"{k} = {json.dumps(v)}" for k, v in CONFIG.items()]
    path = tmp_path / "teachers.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_plan_estimate_and_run_without_volume_confirmation(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(italian, "load", lambda: ENTRIES)
    cases = write_jsonl(tmp_path / "aegis2-train.jsonl", RECORDS)
    out, ledger = tmp_path / "it", tmp_path / "spesa.jsonl"
    assert italian.main(["plan", "--out", str(out), "--cases", str(cases), "--translate", "2",
                         "--native", "1", "--mixed", "1"]) == 0
    assert (out / "plan.jsonl").exists()
    common = ["--out", str(out), "--teachers", str(teachers(tmp_path)),
              "--teacher", CONFIG["model_id"], "--ledger", str(ledger)]
    capsys.readouterr()
    assert italian.main(["estimate", *common]) == 0
    estimate = json.loads(capsys.readouterr().out)
    assert estimate["items"] == 4 and estimate["volume_confirmation"] is None
    assert italian.main(["run", *common, "--confirm-spend", estimate["confirm"]]) == 1
    assert "conferma umana" in capsys.readouterr().err
    assert not ledger.exists() or ledger.read_text() == ""


def test_unknown_teacher_and_inside_repo_output_exit_with_an_error(tmp_path, capsys):
    from scudo.ingest import REPO_ROOT

    cases = write_jsonl(tmp_path / "c.jsonl", RECORDS)
    assert italian.main(["plan", "--out", str(REPO_ROOT / "it"), "--cases", str(cases),
                         "--translate", "1", "--native", "0", "--mixed", "0"]) == 1
    assert italian.main(["estimate", "--out", str(tmp_path / "it"), "--teachers",
                         str(teachers(tmp_path)), "--teacher", "org/none", "--ledger",
                         str(tmp_path / "l.jsonl")]) == 1


def test_there_is_no_side_door_mix_the_fast_mix_is_scudo_mix_cli(tmp_path, capsys):
    with pytest.raises(SystemExit):
        italian.main(["mix", "--dest", str(tmp_path / "mix"), "--n", "4"])
    assert "invalid choice" in capsys.readouterr().err
    assert importlib.util.find_spec("scudo.language_mix") is None
