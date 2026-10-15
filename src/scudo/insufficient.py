"""Casi INSUFFICIENT costruiti dai casi etichettati, e loro quota nel mix.

Ogni metodo lavora solo su una fonte dove l'evidenza decisiva è nota per
struttura (un turno, un record, un campo), mai su testo libero:

- `evidence_removed` (CantTalkAboutThis): si toglie il turno utente giudicato;
- `criterion_absent_info` (procedural, `record_eligibility`): il criterio chiede
  di un record che lo stato non descrive;
- `context_truncated` (procedural, `record_eligibility`): lo stato è troncato
  prima del record sondato;
- `facts_not_provided` (Open-Jev, aritmetica di reasoning-control): la risposta
  proposta dipende dai valori delle variabili, che vengono tolti.

Regola di esclusione: mai INSUFFICIENT su un testo che basta di per sé a
violare. Si toglie solo l'evidenza decisiva nota, e un controllo strutturale
verifica che ciò che resta non la contenga; ogni altra fonte è esclusa.
Il figlio tiene fonte, lingua e superficie del genitore; `source_id` è
`<genitore>#insufficient:<metodo>`.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace

from scudo import canttalk, open_jev, procedural
from scudo.cases import Case

METHODS = ("evidence_removed", "criterion_absent_info", "context_truncated", "facts_not_provided")
DEFAULT_SHARE = 0.12  # quota INSUFFICIENT nel mix, dentro il 10–15% dell'issue #6

# Espressioni verificate sul train della config redistributable: ogni variabile
# compare una volta e il risultato dipende da tutte, quindi senza valori non si calcola.
ARITHMETIC_EXPRESSIONS = frozenset({"a + (b * c)", "(a - b) * c", "(a * b) + c"})
_ARITHMETIC_TASK = "Task: Calculate the exact expression result.\n"
_ARITHMETIC_KEYS = {"expression", "record_id", "semantics", "values"}


class Excluded(ValueError):
    """Il caso non può generare un INSUFFICIENT fedele; `reason` dice perché."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _remove_judged_turn(case: Case) -> tuple[str, str]:
    turns = json.loads(case.evidence)
    if not turns or turns[-1]["role"] != "user":
        raise Excluded("no_judged_turn")
    retained = turns[:-1]
    if retained and retained[-1]["role"] == "user":
        raise Excluded("retained_ends_with_user")  # resterebbe un altro turno da giudicare
    return case.criterion, json.dumps(retained, ensure_ascii=False)


def _record_probe(case: Case):
    """Costruttore del criterio e argomenti (record, requisito) del caso sondato."""
    for pattern, build in ((r"Is record (R\d+) ineligible", procedural.eligibility_criterion),
                           (r"Does record (R\d+) violate requirement (C\d+) ",
                            procedural.requirement_criterion)):
        match = re.match(pattern, case.criterion)
        if match and build(*match.groups()) == case.criterion:
            return build, match.groups()
    raise Excluded("not_record_probe")


def _ask_absent_record(case: Case) -> tuple[str, str]:
    build, args = _record_probe(case)
    ids = [record["id"] for record in json.loads(case.evidence)["records"]]
    number = max((int(id_[1:]) for id_ in ids), default=0) + 1
    while re.search(rf"\bR{number}\b", case.evidence):
        number += 1
    return build(f"R{number}", *args[1:]), case.evidence


def _truncate_before_record(case: Case) -> tuple[str, str]:
    _, (record, *_) = _record_probe(case)
    marker = '{"id": ' + json.dumps(record)
    if case.evidence.count(marker) != 1:
        raise Excluded("probed_record_not_found")
    retained = case.evidence[: case.evidence.index(marker)]
    if re.search(rf"\b{record}\b", retained):
        raise Excluded("decisive_evidence_retained")
    return case.criterion, retained


def _remove_values(case: Case) -> tuple[str, str]:
    try:
        state = json.loads(case.evidence)
    except json.JSONDecodeError:
        raise Excluded("not_arithmetic_state") from None
    if (not case.criterion.startswith(_ARITHMETIC_TASK) or not isinstance(state, dict)
            or set(state) != _ARITHMETIC_KEYS or set(state["values"]) != {"a", "b", "c"}):
        raise Excluded("not_arithmetic_state")
    if state["expression"] not in ARITHMETIC_EXPRESSIONS:
        raise Excluded("unverified_expression")
    del state["values"]
    return case.criterion, json.dumps(state, ensure_ascii=False)


_METHODS = {
    "evidence_removed": (canttalk.DATASET, _remove_judged_turn),
    "criterion_absent_info": (procedural.DATASET, _ask_absent_record),
    "context_truncated": (procedural.DATASET, _truncate_before_record),
    "facts_not_provided": (open_jev.DATASET, _remove_values),
}


def derive(case: Case, method: str) -> Case:
    """Un caso INSUFFICIENT dal genitore etichettato, o `Excluded` con il motivo."""
    if method not in _METHODS:
        raise ValueError(f"metodo sconosciuto: {method!r}")
    source, build = _METHODS[method]
    if case.source != source:
        raise Excluded("unsupported_source")
    if case.verdict == "INSUFFICIENT":
        raise Excluded("parent_insufficient")
    criterion, evidence = build(case)
    return replace(case, source_id=f"{case.source_id}#insufficient:{method}",
                   criterion=criterion, evidence=evidence, verdict="INSUFFICIENT")


def derive_all(cases: Iterable[Case], methods: Sequence[str] = METHODS) -> tuple[list[Case], Counter]:
    """Tutti i figli possibili, senza doppioni di criterio ed evidenza; esclusioni contate."""
    children, skipped, seen = [], Counter(), set()
    for case in cases:
        for method in methods:
            try:
                child = derive(case, method)
            except Excluded as excluded:
                skipped[(method, excluded.reason)] += 1
                continue
            if (child.criterion, child.evidence) in seen:
                skipped[(method, "duplicate")] += 1
                continue
            seen.add((child.criterion, child.evidence))
            children.append(child)
    return children, skipped


def _key(seed: int, case: Case) -> str:
    # Tutti i campi: due casi con la stessa chiave sono uguali, quindi l'ordine non conta.
    fields = (case.source, case.source_id, case.language, case.surface, case.criterion,
              case.evidence, case.verdict)
    return hashlib.sha256(json.dumps([seed, *fields]).encode()).hexdigest()


def mix_insufficient(cases: Sequence[Case], generated: Sequence[Case], *,
                     share: float = DEFAULT_SHARE, seed: int) -> list[Case]:
    """Aggiunge INSUFFICIENT generati finché sono `share` del mix, contando quelli già presenti.

    Deterministico e indipendente dall'ordine; fallisce invece di riempire a metà.
    """
    if not 0 < share < 1:
        raise ValueError(f"quota fuori da (0, 1): {share}")
    if any(case.verdict != "INSUFFICIENT" for case in generated):
        raise ValueError("i casi generati devono essere tutti INSUFFICIENT")
    present = sum(case.verdict == "INSUFFICIENT" for case in cases)
    needed = round((share * len(cases) - present) / (1 - share))
    if needed < 0:
        raise ValueError(f"i casi di partenza superano già la quota INSUFFICIENT {share}")
    if needed > len(generated):
        raise ValueError(f"pool INSUFFICIENT troppo piccolo: servono {needed}, ci sono {len(generated)}")
    picked = sorted(generated, key=lambda case: _key(seed, case))[:needed]
    return sorted([*cases, *picked], key=lambda case: _key(seed, case))
