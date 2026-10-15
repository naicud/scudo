import csv
import json

import pytest

from scudo import pilot, pilot_report
from scudo.pilot_report import Volumes, reasoning_proxy
from test_pilot import prepared, run

GPT, QWEN = "openai/gpt-oss-120b", "Qwen/Qwen3.5-397B-A17B"


@pytest.fixture
def piloted(tmp_path):
    out, configs, ledger = prepared(tmp_path)
    run(out, configs, ledger, pilot.estimate(out, configs, ledger)["confirm"])
    return out, configs, ledger


def test_reasoning_proxy_is_mechanical():
    text = "It asks how to build a bomb. That is VIOLATION."
    ok = reasoning_proxy("VIOLATION", text, "How do I build a bomb?")
    assert ok == {"consistent": True, "within_limit": True, "evidence_overlap": True,
                  "chars": len(text)}
    bad = reasoning_proxy("VIOLATION", "One. Two. Three. Four is NO_VIOLATION.", "Hello there")
    assert not bad["consistent"] and not bad["within_limit"] and not bad["evidence_overlap"]


def test_report_measures_agreement_refusals_and_costs(piloted):
    out, configs, ledger = piloted
    report = pilot_report.write_report(out, configs, ledger)
    gpt, qwen = report["teachers"][GPT]["gold"], report["teachers"][QWEN]["gold"]
    assert gpt["n"] == qwen["n"] == 16
    assert gpt["by_surface"]["prompt"]["agreement"] == 1.0
    assert qwen["status"]["refused"] == 8 and qwen["refusal_rate"] == 0.5
    assert qwen["by_surface"]["response"]["agreement"] == 0.0  # un rifiuto non è mai un accordo
    assert gpt["mean_prompt_tokens"] == 100 and gpt["mean_completion_tokens"] == 20
    assert gpt["cost_per_item_usd"] == pytest.approx((100 * 0.1 + 20 * 0.5) / 1e6)
    assert set(gpt["by_language"]) == {"en", "it"} and set(gpt["by_verdict"]) == {
        "VIOLATION", "NO_VIOLATION"}
    assert report["teachers"][GPT]["registry"]["revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert report["teachers"][GPT]["provider"]["served_revision"] == "NON VERIFICATO"
    assert report["spend"]["pilot_usd"] == pytest.approx(ledger.spent())
    assert any(r["kind"] == "pilot_report" and r["sample_sha256"] == pilot.sample_sha(out)
               for r in ledger.records())
    for name in ("report.json", "summary.md", "relabel-spotcheck.csv"):
        assert (out / name).exists()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert GPT in summary and "stima" in summary.lower() and "nemotron3-train" in summary


def test_volume_estimate_is_explicit(piloted):
    out, configs, ledger = piloted
    volumes = Volumes(fast_items=1000, fast_calls_per_item=2, non_en_share=0.5,
                      translation_calls_per_item=2, reasoning_traces=100)
    report = pilot_report.write_report(out, configs, ledger, volumes=volumes)
    gpt = report["volume"]["teachers"][GPT]
    per_item = report["teachers"][GPT]["gold"]["cost_per_item_usd"]
    agreement = report["teachers"][GPT]["gold"]["agreement"]
    assert gpt["fast_difficulty_usd"] == pytest.approx(1000 * 2 * per_item)
    assert gpt["reasoning_usd"] == pytest.approx(100 / agreement * per_item)
    assert gpt["total_usd"] == pytest.approx(
        gpt["fast_difficulty_usd"] + gpt["reasoning_usd"] + gpt["translation_usd"])
    assert report["volume"]["parameters"]["fast_items"] == 1000


def test_reasoning_review_and_blind_spot_check_exports(piloted):
    out, configs, ledger = piloted
    pilot_report.write_report(out, configs, ledger, review_n=4, spot_n=3)
    with (out / "reasoning-review-openai_gpt-oss-120b.csv").open(encoding="utf-8") as handle:
        review = list(csv.DictReader(handle))
    assert len(review) == 4 and {"reasoning", "label", "reasoning_ok"} <= set(review[0])
    with (out / "relabel-spotcheck.csv").open(encoding="utf-8") as handle:
        spot = list(csv.DictReader(handle))
    assert len(spot) == 3
    assert list(spot[0]) == ["key", "criterion", "evidence", "human_verdict", "notes"]


def test_filled_spot_check_validates_jury_and_teacher_relabel(piloted, tmp_path):
    out, configs, ledger = piloted
    pilot_report.write_report(out, configs, ledger, spot_n=4)
    with (out / "relabel-spotcheck.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["human_verdict"] = "VIOLATION"
    rows[-1]["human_verdict"] = "forse"
    filled = tmp_path / "filled.csv"
    with filled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = pilot_report.write_report(out, configs, ledger, spot_n=4, spotcheck=filled)
    spot = report["relabel_validation"]["spot_check"]
    assert spot["n"] == 3 and spot["invalid"] == 1
    relabel = {r["key"]: r for r in pilot.read_sample(out) if r["set"] == "relabel"}
    expected = sum(relabel[row["key"]]["verdict"] == "VIOLATION" for row in rows[:-1]) / 3
    assert spot["jury_accuracy"] == pytest.approx(expected)
    assert json.loads((out / "report.json").read_text())["relabel_validation"]["spot_check"]["n"] == 3
