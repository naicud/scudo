"""Readout dell'Accurate (`scudo.reasoned-decision/1`) sull'harness di `scudo.runner`.

Aggiunge solo il passo ragionato: prompt decision-token + `Reasoning:\\n`,
generazione greedy entro il budget, token fino allo stop, poi `Answer:\\n` e logit
delle lettere, nella sequenza del frame nativo di open-guard. Caricamento dei
pesi verificati, `run` e report restano quelli di `scudo.runner`. Python puro.
"""

import random
from collections.abc import Callable, Sequence

from scudo.benchmarks import Loaded
from scudo.cases import Case
from scudo.eval_report import EvalResult, token_cost
from scudo.protocol import OPTION_LETTERS, SEMANTIC_IDS, Readout, read_options
from scudo.reasoned import (
    ANSWER_PREFIX,
    MAX_NEW_TOKENS,
    REASONING_PREFIX,
    ReasonedError,
    render_reasoned_prompt,
    tokens_through_stop,
)
from scudo.runner import Backend, RunnerError, Tokenizer, letter_ids, reports_for


def accurate_scorer(tokenizer: Tokenizer, backend: Backend, *, seed: int,
                    budget: int = MAX_NEW_TOKENS) -> tuple[Callable[[Case], Readout], dict]:
    """Funzione caso -> Readout e `usage[caso]`; permutazioni delle opzioni come il Fast.

    `usage[caso]` ha i token generati (anche per un caso in errore, fine della
    generazione compresa) e, se letto, prompt, ragionamento e logit per il confronto live.
    Gli errori del ragionamento diventano `RunnerError` con il loro codice.
    """
    letters = letter_ids(tokenizer)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    answer = encode(ANSWER_PREFIX)
    if any(encode(ANSWER_PREFIX + x) != answer + [letters[x]] for x in OPTION_LETTERS):
        raise RunnerError("TOKENIZER_UNSUPPORTED", "Answer: + lettera non è esatto")
    rng = random.Random(seed)
    usage: dict[Case, dict] = {}

    def score(case: Case) -> Readout:
        semantic_ids = list(SEMANTIC_IDS)
        rng.shuffle(semantic_ids)
        prompt = render_reasoned_prompt(case.criterion, case.evidence, semantic_ids)
        ids = encode(prompt[:-len(REASONING_PREFIX)]) + encode(REASONING_PREFIX)
        if encode(prompt) != ids:
            raise RunnerError("TOKENIZER_UNSUPPORTED", "prompt + Reasoning: non è esatto")
        generated, _ = backend.generate(ids, budget)
        try:
            used = tokens_through_stop(generated, decode, budget=budget)
        except ReasonedError as error:
            usage[case] = {"generated_tokens": len(generated)}
            raise RunnerError(error.code, str(error)) from None
        logits = backend.next_logits(ids + generated[:used] + answer)
        readout = read_options([float(logits[letters[x]]) for x in OPTION_LETTERS],
                               semantic_ids)
        usage[case] = {"prompt": prompt, "semantic_ids": semantic_ids,
                       "reasoning": decode(generated[:used]), "generated_tokens": used,
                       "logits": list(readout.logits)}
        return readout

    return score, usage


def accurate_reports(loaded: Loaded, results: Sequence[EvalResult], usage: dict, *,
                     budget: int, dedup: dict, **header) -> dict[str, dict]:
    """`runner.reports_for` più `token_cost` sui casi di ogni report."""
    reports = reports_for(loaded, results, dedup=dedup, **header)
    generated = {case_id: usage[case]["generated_tokens"]
                 for case_id, case in zip(loaded.case_ids, loaded.cases, strict=True)
                 if case in usage}
    for split, report in reports.items():
        ids = [case_id for case_id in loaded.case_ids
               if split == loaded.name or case_id not in loaded.machine_labelled]
        if counts := [generated[case_id] for case_id in ids if case_id in generated]:
            report["token_cost"] = token_cost(counts, budget=budget)
    return reports
