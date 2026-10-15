import pytest

from scudo import budget
from scudo.budget import CAP_USD, BudgetError, Guard, Ledger, authorise
from scudo.ingest import REPO_ROOT
from scudo.teacher_client import Outcome
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


GPT, QWEN = "openai/gpt-oss-120b", "Qwen/Qwen3.5-397B-A17B"


def confirm(book, teachers, amount, rationale="pilot ok"):
    budget.confirm_volume(book, teachers=teachers, amount_usd=amount, by="Mario",
                          rationale=rationale)


def test_volume_needs_a_recorded_human_confirmation(tmp_path):
    book = ledger(tmp_path, 3.0)
    with pytest.raises(BudgetError):
        budget.require_volume_confirmation(book)
    with pytest.raises(BudgetError):
        budget.confirm_volume(book, teachers=[GPT], amount_usd=10.0, by=" ", rationale="pilot ok")
    with pytest.raises(BudgetError, match="registro"):
        confirm(book, [GPT, "org/unknown"], 10.0)  # ogni teacher deve essere un candidato
    with pytest.raises(BudgetError, match="nessun teacher"):
        confirm(book, [], 10.0)
    with pytest.raises(BudgetError, match="residuo"):
        confirm(book, [GPT], 148.0)
    confirm(book, [QWEN, GPT, GPT], 100.0)
    record = budget.require_volume_confirmation(book)
    assert record["teachers"] == sorted([GPT, QWEN])  # insieme, senza duplicati
    assert record["by"] == "Mario" and record["amount_usd"] == 100.0


def test_volume_remaining_counts_only_calls_after_the_latest_confirmation(tmp_path):
    book = ledger(tmp_path, 5.0)  # spesa del pilot, prima della conferma
    with pytest.raises(BudgetError):
        budget.volume_remaining(book)
    confirm(book, [GPT], 20.0, "report del pilot")
    book.append({"kind": "call", "run_id": "it", "cost_usd": 3.0})
    assert budget.volume_remaining(book) == pytest.approx(17.0)
    confirm(book, [GPT], 10.0, "nuova conferma")
    assert budget.volume_remaining(book) == pytest.approx(10.0)


def test_volume_check_covers_the_teacher_set_of_the_latest_confirmation(tmp_path):
    book = ledger(tmp_path)
    confirm(book, [GPT, QWEN], 10.0)
    assert budget.check_volume(book, [GPT, QWEN], 10.0)["teachers"] == sorted([GPT, QWEN])
    with pytest.raises(BudgetError, match="volume confermato"):  # somma dei teacher oltre
        budget.check_volume(book, [GPT, QWEN], 10.001)
    confirm(book, [QWEN], 20.0, "cambio")
    with pytest.raises(BudgetError, match="senza conferma del volume"):
        budget.check_volume(book, [GPT], 1.0)  # confermato prima, non nell'ultima conferma
    assert budget.check_volume(book, [QWEN], 20.0)["teachers"] == [QWEN]
    with pytest.raises(BudgetError, match="volume confermato"):
        budget.check_volume(book, [QWEN], 20.001)
    with pytest.raises(BudgetError, match="tetto"):
        budget.check_volume(book, [QWEN], 15.0, cap=10.0)


def test_confirmation_written_before_teacher_sets_still_counts(tmp_path):
    book = ledger(tmp_path)
    book.append({"kind": "volume_confirmation", "teacher": GPT, "amount_usd": 10.0,
                 "by": "Mario", "rationale": "registro precedente"})
    assert budget.check_volume(book, [GPT], 5.0)["teacher"] == GPT
    with pytest.raises(BudgetError, match="senza conferma del volume"):
        budget.check_volume(book, [QWEN], 5.0)


def test_paid_call_applies_one_charging_rule(tmp_path):
    cfg = config()
    chat = [{"role": "user", "content": "x" * 400}]
    worst = budget.worst_case_usd(cfg, chat)
    book = ledger(tmp_path)
    guard = Guard(book, "run", 1.0)

    def paid(outcome):
        return budget.paid_call(guard, cfg, "k", chat, lambda: outcome)

    used = budget.cost_usd(cfg, 100, 20)
    assert paid(Outcome("ok", prompt_tokens=100, completion_tokens=20))[1:] == (
        pytest.approx(used), False)
    assert paid(Outcome("error", http_status=400))[1] == 0.0  # respinta: nessuna generazione
    assert paid(Outcome("timeout"))[1:] == (pytest.approx(worst), True)
    assert paid(Outcome("error", http_status=503))[1] == pytest.approx(worst)
    with pytest.raises(budget.KeyRefused, match="chiave"):
        paid(Outcome("error", http_status=401, detail="HTTP 401"))

    def cut():
        raise RuntimeError("connessione persa a metà risposta")

    with pytest.raises(RuntimeError):  # può essere stata fatturata: caso peggiore
        budget.paid_call(guard, cfg, "k", chat, cut)
    assert [c["cost_usd"] for c in book.of_kind("call")] == pytest.approx(
        [used, 0.0, worst, worst, worst])
    assert guard.spent == pytest.approx(book.spent())
