"""Margine del verdetto di un teacher dai logprob dei token (chat completions compatibile OpenAI).

Con `logprobs = true` e `top_logprobs = k` nella richiesta, ogni token della
risposta porta il suo logprob e le k alternative più probabili. I tre id del
verdetto iniziano con lettere diverse (`V`, `N`, `I`), quindi il verdetto si
decide al token che contiene il primo carattere del valore di `"verdict"`: il
margine è il logprob di quel token meno quello della migliore alternativa che
inizia un verdetto diverso, in nat, come `option_margin` dei readout. Se nessuna
alternativa del top-k inizia un altro verdetto, il margine è un limite inferiore:
il logprob del token meno il più basso del top-k. Negativo se il token campionato
non era il più probabile.

Margine assente (`None`), mai inventato: logprob non restituiti, token che non
ricostruiscono il contenuto (anche dopo eventuali token di ragionamento in testa),
verdetto non trovato nel contenuto, nessuna alternativa.
"""

import re

from scudo.protocol import SEMANTIC_IDS

_VERDICT_KEY = re.compile(r'"verdict"\s*:\s*"')


def _entries(logprobs) -> list[dict] | None:
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or not content:
        return None
    ok = all(isinstance(e, dict) and isinstance(e.get("token"), str)
             and isinstance(e.get("logprob"), int | float) for e in content)
    return content if ok else None


def _verdict_of(rest: str) -> str | None:
    """Il verdetto che un testo inizia (o contiene per intero), se ne inizia uno."""
    found = [v for v in SEMANTIC_IDS if rest and (v.startswith(rest) or rest.startswith(v))]
    return found[0] if len(found) == 1 else None


def verdict_margin(content: str, verdict: str, logprobs) -> float | None:
    """Margine in nat del verdetto al token decisivo; None se non misurabile."""
    entries = _entries(logprobs)
    if entries is None or not isinstance(content, str):
        return None
    joined = "".join(e["token"] for e in entries)
    if not joined.endswith(content):
        return None
    offset = len(joined) - len(content)
    match = _VERDICT_KEY.search(content)
    if not match or not content.startswith(verdict, match.end()):
        return None
    start, position = offset + match.end(), 0
    for entry in entries:
        if position + len(entry["token"]) > start:
            break
        position += len(entry["token"])
    prefix = joined[position:start]
    chosen = entry["logprob"]
    top = [t for t in entry.get("top_logprobs") or []
           if isinstance(t, dict) and isinstance(t.get("token"), str)
           and isinstance(t.get("logprob"), int | float)]
    rivals = [t["logprob"] for t in top if t["token"].startswith(prefix)
              and _verdict_of(t["token"][len(prefix):]) not in (None, verdict)]
    if rivals:
        return chosen - max(rivals)
    others = [t["logprob"] for t in top if t["token"] != entry["token"]]
    return chosen - min(others) if others else None
