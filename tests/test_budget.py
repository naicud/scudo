import pytest

from scudo import budget
from scudo.budget import CAP_USD, BudgetError, Guard, Ledger, authorise
from scudo.ingest import REPO_ROOT
from test_teacher import config


def ledger(tmp_path, *spent):
    book = Ledger(tmp_path / "ledger.jsonl")
    for cost in spent:
        book.append({"kind": "call", "run_id": "old", "teacher": "t", "key": "k", "cost_usd": cost})
    return book


def test_tokens_are_estimated_as_chars_over_four_rounded_up():
    assert budget.estimate_tokens("") == 0
    assert budget.estimate_tokens("abcd") == 1
    assert budget.estimate_tokens("abcde") == 2


def test_cost_uses_the_human_price_table():
    cfg = config(input_usd_per_mtok=2.0, output_usd_per_mtok=10.0, max_tokens=1000)
    assert budget.cost_usd(cfg, 1_000_000, 0) == pytest.approx(2.0)
    assert budget.cost_usd(cfg, 500, 100) == pytest.approx(0.002)
    messages = [{"role": "user", "content": "x" * 3998}]
    worst = budget.worst_case_usd(cfg, messages)
    assert worst == pytest.approx(budget.cost_usd(cfg, budget.estimate_tokens(
        budget.messages_text(messages)), 1000))


def test_ledger_outside_the_repository_only(tmp_path):
    with pytest.raises(ValueError):
        Ledger(REPO_ROOT / "ledger.jsonl")
    assert ledger(tmp_path, 1.5, 2.25).spent() == pytest.approx(3.75)
    assert ledger(tmp_path).remaining() == pytest.approx(CAP_USD - 3.75)


def test_spend_is_refused_without_an_exact_confirmation(tmp_path):
    book = ledger(tmp_path)
    with pytest.raises(BudgetError, match="--confirm-spend 1.24"):
        authorise(book, "run", 1.231, None)
    with pytest.raises(BudgetError):
        authorise(book, "run", 1.231, "1.23")
    with pytest.raises(BudgetError):
        authorise(book, "run", 1.231, "5")
    assert not [r for r in book.records() if r["kind"] == "authorisation"]
    assert authorise(book, "run", 1.231, "1.24") == pytest.approx(1.24)
    [record] = [r for r in book.records() if r["kind"] == "authorisation"]
    assert record["amount_usd"] == pytest.approx(1.24) and record["run_id"] == "run"


def test_spend_over_the_remaining_cap_is_refused(tmp_path):
    book = ledger(tmp_path, 149.0)
    with pytest.raises(BudgetError, match="residuo"):
        authorise(book, "run", 1.5, "1.50")


def test_guard_stops_before_exceeding_the_authorised_amount(tmp_path):
    book = ledger(tmp_path)
    guard = Guard(book, "run", 0.10)
    guard.check(0.06)
    guard.charge("t", "k1", 0.06, usage_missing=False)
    with pytest.raises(BudgetError, match="stop"):
        guard.check(0.05)
    assert guard.spent == pytest.approx(0.06)
    calls = [r for r in book.records() if r["kind"] == "call"]
    assert calls[-1]["cost_usd"] == pytest.approx(0.06) and calls[-1]["run_id"] == "run"
    assert book.spent_by_run("run") == pytest.approx(0.06)


def test_volume_needs_a_recorded_human_confirmation(tmp_path):
    book = ledger(tmp_path, 3.0)
    with pytest.raises(BudgetError):
        budget.require_volume_confirmation(book)
    with pytest.raises(BudgetError):
        budget.confirm_volume(book, teacher="openai/gpt-oss-120b", amount_usd=10.0, by=" ",
                              rationale="pilot ok")
    with pytest.raises(BudgetError):
        budget.confirm_volume(book, teacher="org/unknown", amount_usd=10.0, by="Mario",
                              rationale="pilot ok")
    with pytest.raises(BudgetError, match="residuo"):
        budget.confirm_volume(book, teacher="openai/gpt-oss-120b", amount_usd=148.0, by="Mario",
                              rationale="pilot ok")
    budget.confirm_volume(book, teacher="openai/gpt-oss-120b", amount_usd=100.0, by="Mario",
                          rationale="pilot ok")
    record = budget.require_volume_confirmation(book)
    assert record["teacher"] == "openai/gpt-oss-120b" and record["by"] == "Mario"
