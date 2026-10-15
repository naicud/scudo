"""Forma di regola dei criteri, per le baseline che giudicano contro una policy.

Granite policy-guardrails è addestrato su testi di policy (una regola da
rispettare, es. «hiring managers should steer away from …»), mentre i nostri
criteri sono domande con la coda «Missing essential context means
INSUFFICIENT». Ogni criterio che l'harness usa sui benchmark ha qui la sua
regola: stesso contenuto come divieto, senza la coda, scritta accanto al
criterio (`aegis`, `benchmarks`, `canttalk`, `default_criteria`). Un criterio
senza regola scritta, come le domande libere di Open-Jev, non ha forma di
regola: `rule_for` dà None e la baseline lo rifiuta, mai una traduzione
improvvisata. I nostri prompt Fast e Accurate non leggono questo modulo.
"""

from scudo import aegis, benchmarks, canttalk
from scudo.default_criteria import DEFAULT_CRITERIA

VERSION = "scudo.policy-rules/1"  # entra nel `prompt_format` delle baseline che la usano

_RULES = {
    aegis.HARMFUL_REQUEST: aegis.HARMFUL_REQUEST_RULE,
    aegis.HARMFUL_RESPONSE: aegis.HARMFUL_RESPONSE_RULE,
    benchmarks.JAILBREAK_ATTEMPT: benchmarks.JAILBREAK_ATTEMPT_RULE,
    **{criterion.text: criterion.rule for criterion in DEFAULT_CRITERIA},
}
# Il criterio di CantTalkAboutThis contiene le istruzioni del dialogo: si riconosce
# dal testo fisso che le circonda.
_TOPIC_HEAD, _TOPIC_TAIL = canttalk.criterion("\0").split("\0")


def rule_for(criterion: str) -> str | None:
    """Regola scritta per il criterio, o None se non ne ha una."""
    if criterion in _RULES:
        return _RULES[criterion]
    if criterion.startswith(_TOPIC_HEAD) and criterion.endswith(_TOPIC_TAIL):
        instructions = criterion[len(_TOPIC_HEAD):len(criterion) - len(_TOPIC_TAIL)]
        if instructions:
            return canttalk.rule(instructions)
    return None
