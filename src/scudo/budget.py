"""Tetto di spesa dei teacher: stima, autorizzazione esplicita, registro e stop.

Il registro è un JSONL fuori dal repository con una riga per evento:
`authorisation` (importo confermato dalla persona per una run), `call` (costo di
una chiamata dall'uso reale di token, o il caso peggiore se il provider non lo
riporta), `pilot_report` (report del pilot scritto) e `volume_confirmation`
(conferma umana dei teacher e di un solo importo in volume per tutti). Il residuo è il tetto v0.1
meno la somma delle chiamate; il residuo del volume è l'importo dell'ultima
conferma meno le chiamate registrate dopo di essa.

La stima dei token è `ceil(caratteri / 4)`: una stima, non un conteggio del
tokenizer. Il caso peggiore di una chiamata è la stima dell'ingresso più
`max_tokens` in uscita; il pilot si ferma prima di una chiamata che potrebbe
superare l'importo autorizzato. Se l'ingresso reale supera la stima, lo
sforamento è limitato a quella chiamata.

Ogni run in volume passa da `check_volume`; ogni chiamata a pagamento passa da
`paid_call`, l'unico posto dove vive la regola di addebito.
"""

import json
import math
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from scudo.teacher import CANDIDATES, TeacherConfig
from scudo.teacher_client import Outcome

CAP_USD = 150.0  # tetto complessivo v0.1 (issue #5)
CHARS_PER_TOKEN = 4
_EPS = 1e-9


class BudgetError(RuntimeError):
    """Spesa non autorizzata o oltre il tetto: nessuna chiamata parte."""


class KeyRefused(BudgetError):
    """401/403: chiave o permessi da verificare; la run si ferma senza addebito."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def messages_text(chat: list[dict]) -> str:
    return json.dumps(chat, ensure_ascii=False)


def cost_usd(config: TeacherConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * config.input_usd_per_mtok
            + completion_tokens * config.output_usd_per_mtok) / 1_000_000


def worst_case_usd(config: TeacherConfig, chat: list[dict]) -> float:
    return cost_usd(config, estimate_tokens(messages_text(chat)), config.max_tokens)


def cents_up(amount: float) -> float:
    return math.ceil(round(amount * 100, 6)) / 100


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path):
        from scudo.ingest import check_outside_repo

        self.path = check_outside_repo(path)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": _now(), **record}, ensure_ascii=False) + "\n")

    def spent(self) -> float:
        return math.fsum(r["cost_usd"] for r in self.records() if r["kind"] == "call")

    def spent_by_run(self, run_id: str) -> float:
        return math.fsum(r["cost_usd"] for r in self.records()
                         if r["kind"] == "call" and r["run_id"] == run_id)

    def remaining(self, cap: float = CAP_USD) -> float:
        return cap - self.spent()

    def of_kind(self, kind: str) -> list[dict]:
        return [r for r in self.records() if r["kind"] == kind]


def authorise(ledger: Ledger, run_id: str, estimate_usd: float, confirm: str | None,
              cap: float = CAP_USD) -> float:
    """Importo autorizzato: la stima arrotondata al centesimo, confermata alla lettera."""
    amount = cents_up(estimate_usd)
    remaining = ledger.remaining(cap)
    if amount > remaining + _EPS:
        raise BudgetError(f"stima {amount:.2f} USD oltre il residuo {remaining:.2f} USD "
                          f"del tetto v0.1 di {cap:.2f} USD")
    if confirm is None or confirm.strip() != f"{amount:.2f}":
        raise BudgetError(f"stima {amount:.2f} USD (residuo {remaining:.2f} USD): per "
                          f"autorizzarla ripetere con --confirm-spend {amount:.2f}")
    ledger.append({"kind": "authorisation", "run_id": run_id, "amount_usd": amount,
                   "estimate_usd": estimate_usd})
    return amount


class Guard:
    """Stop rigido: nessuna chiamata che possa portare la run oltre l'importo autorizzato."""

    def __init__(self, ledger: Ledger, run_id: str, authorised_usd: float):
        self.ledger, self.run_id, self.authorised = ledger, run_id, authorised_usd
        self.spent = 0.0

    def check(self, worst_case: float) -> None:
        if self.spent + worst_case > self.authorised + _EPS:
            raise BudgetError(f"stop: spesi {self.spent:.4f} USD su {self.authorised:.2f} "
                              f"autorizzati, la prossima chiamata può costare {worst_case:.4f}")

    def charge(self, teacher: str, key: str, cost: float, *, usage_missing: bool) -> None:
        self.spent += cost
        self.ledger.append({"kind": "call", "run_id": self.run_id, "teacher": teacher,
                            "key": key, "cost_usd": cost, "usage_missing": usage_missing})


def paid_call(guard: Guard, cfg: TeacherConfig, key: str, chat: list[dict],
              send: Callable[[], Outcome]) -> tuple[Outcome, float, bool]:
    """Una chiamata sotto stop rigido, addebitata con un'unica regola; (esito, costo, uso mancante).

    401/403 fermano la run senza addebito (configurazione, non un esito); un'altra
    4xx è respinta e costa zero; con l'uso riportato si paga l'uso; senza (timeout,
    rete, 5xx, corpo troncato, provider muto) il caso peggiore. Se `send` solleva,
    la richiesta può essere stata elaborata e fatturata: caso peggiore, poi rilancia.
    """
    worst = worst_case_usd(cfg, chat)
    guard.check(worst)
    try:
        outcome = send()
    except BaseException:
        guard.charge(cfg.model_id, key, worst, usage_missing=True)
        raise
    if outcome.http_status in (401, 403):
        raise KeyRefused(f"{cfg.model_id}: {outcome.detail}; chiave o permessi da verificare")
    usage_missing = outcome.prompt_tokens is None or outcome.completion_tokens is None
    rejected = outcome.http_status is not None and 400 <= outcome.http_status < 500
    cost = 0.0 if rejected else worst if usage_missing else cost_usd(
        cfg, outcome.prompt_tokens, outcome.completion_tokens)
    guard.charge(cfg.model_id, key, cost, usage_missing=usage_missing)
    return outcome, cost, usage_missing


def confirm_volume(ledger: Ledger, *, teachers: Iterable[str], amount_usd: float, by: str,
                   rationale: str, cap: float = CAP_USD) -> None:
    """Registra i teacher scelti e un solo importo per tutti: la esegue una persona, mai il codice."""
    chosen = sorted(set(teachers))
    if not by.strip() or not rationale.strip():
        raise BudgetError("conferma del volume senza chi e perché")
    if not chosen:
        raise BudgetError("conferma del volume senza nessun teacher")
    if unknown := [t for t in chosen if t not in CANDIDATES]:
        raise BudgetError(f"teacher {unknown} non nel registro {sorted(CANDIDATES)}")
    if not 0 < amount_usd <= ledger.remaining(cap) + _EPS:
        raise BudgetError(f"importo {amount_usd:.2f} USD fuori dal residuo "
                          f"{ledger.remaining(cap):.2f} USD")
    ledger.append({"kind": "volume_confirmation", "teachers": chosen, "amount_usd": amount_usd,
                   "by": by.strip(), "rationale": rationale.strip()})


def confirmed_teachers(confirmation: dict) -> set[str]:
    """Teacher coperti da una conferma; le conferme scritte prima degli insiemi ne hanno uno."""
    return set(confirmation.get("teachers") or [confirmation["teacher"]])


def require_volume_confirmation(ledger: Ledger) -> dict:
    """L'ultima conferma umana del volume, o BudgetError: senza di essa niente volume."""
    confirmations = ledger.of_kind("volume_confirmation")
    if not confirmations:
        raise BudgetError("generazione in volume rifiutata: manca la conferma umana "
                          "(python -m scudo.pilot confirm-volume)")
    return confirmations[-1]


def volume_remaining(ledger: Ledger) -> float:
    """Importo dell'ultima conferma del volume meno le chiamate registrate dopo di essa."""
    require_volume_confirmation(ledger)
    records = ledger.records()
    last = max(i for i, r in enumerate(records) if r["kind"] == "volume_confirmation")
    spent = math.fsum(r["cost_usd"] for r in records[last + 1:] if r["kind"] == "call")
    return records[last]["amount_usd"] - spent


def check_volume(ledger: Ledger, teachers: Iterable[str], planned_usd: float,
                 cap: float = CAP_USD) -> dict:
    """Controllo di ogni run in volume, prima di qualsiasi chiamata; restituisce l'ultima conferma.

    Vale solo l'ultima conferma: ogni teacher della run deve essere fra i suoi, e la
    stima della run (somma su tutti i teacher) arrotondata al centesimo deve stare nel
    residuo del volume confermato e del tetto.
    """
    confirmation = require_volume_confirmation(ledger)
    covered = confirmed_teachers(confirmation)
    if others := sorted(set(teachers) - covered):
        raise BudgetError(f"teacher senza conferma del volume: {others}; l'ultima conferma vale "
                          f"solo per i teacher confermati {sorted(covered)}")
    amount = cents_up(planned_usd)
    if amount > (left := volume_remaining(ledger)) + _EPS:
        raise BudgetError(f"stima {amount:.2f} USD oltre il residuo {left:.2f} USD "
                          "del volume confermato")
    if amount > (left := ledger.remaining(cap)) + _EPS:
        raise BudgetError(f"stima {amount:.2f} USD oltre il residuo {left:.2f} USD "
                          f"del tetto v0.1 di {cap:.2f} USD")
    return confirmation
