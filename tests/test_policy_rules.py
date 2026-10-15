import pytest

from scudo import aegis, canttalk, nemotron
from scudo.benchmarks import JAILBREAK_ATTEMPT, JAILBREAK_ATTEMPT_RULE
from scudo.default_criteria import DEFAULT_CRITERIA, INJECTION_DIRECT
from scudo.policy_rules import VERSION, rule_for

TOPIC = "You are a travel agent. Only discuss flight bookings."


def benchmark_criteria():
    return [aegis.HARMFUL_REQUEST, aegis.HARMFUL_RESPONSE, nemotron.HARMFUL_REQUEST,
            nemotron.HARMFUL_RESPONSE, JAILBREAK_ATTEMPT, canttalk.criterion(TOPIC),
            *(c.text for c in DEFAULT_CRITERIA)]


@pytest.mark.parametrize("criterion", benchmark_criteria())
def test_every_harness_criterion_has_a_prohibition_rule_without_our_tri_state_tail(criterion):
    rule = rule_for(criterion)
    assert rule and "must not" in rule
    assert "INSUFFICIENT" not in rule and "?" not in rule


def test_aegis_and_jailbreak_rules_are_the_declared_texts():
    assert rule_for(aegis.HARMFUL_REQUEST) == aegis.HARMFUL_REQUEST_RULE
    assert rule_for(aegis.HARMFUL_RESPONSE) == aegis.HARMFUL_RESPONSE_RULE
    assert rule_for(JAILBREAK_ATTEMPT) == JAILBREAK_ATTEMPT_RULE
    assert aegis.HARMFUL_REQUEST_RULE.startswith(
        "The user's message must not request, promote or contain violence, threats")
    assert aegis.HARMFUL_RESPONSE_RULE.startswith(
        "The assistant's response must not provide, promote or contain violence")


def test_default_criterion_rule_is_the_same_question_as_a_prohibition():
    rule = rule_for(INJECTION_DIRECT.text)
    question = INJECTION_DIRECT.text.removeprefix("Does the text ").split("?")[0]
    assert rule.startswith("The text must not " + question + ". Ordinary requests")
    assert rule.endswith("are not violations.")


def test_canttalk_rule_keeps_the_dialogue_instructions():
    rule = rule_for(canttalk.criterion(TOPIC))
    assert rule == canttalk.rule(TOPIC)
    assert TOPIC in rule and "final turn" in rule


@pytest.mark.parametrize("criterion", [
    "Does the state show that the refund was approved?",  # domanda libera di Open-Jev
    aegis.HARMFUL_REQUEST + " ",
    canttalk.criterion(TOPIC).replace("off-topic", "on-topic"),
])
def test_criterion_without_a_written_rule_has_none(criterion):
    assert rule_for(criterion) is None


def test_rules_have_a_version_for_the_report_format():
    assert VERSION == "scudo.policy-rules/1"
