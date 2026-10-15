import json

import pytest

from scudo.procedural import (
    CONFIG,
    DATASET,
    LICENCE,
    REVISION,
    cases_from_row,
    convert,
    eligibility_criterion,
    requirement_criterion,
)

ELIGIBILITY_STATE = json.dumps({
    "user_request": "Select eligible venue packages under all of the following conditions:\n"
                    "C1. Capacity must accommodate at least 20 people.",
    "records": [{"id": "R4", "description": "Capacity is 19 people."},
                {"id": "R2", "description": "No information about capacity is provided."}],
    "semantics": "Explicitly absent information and conflicting claims count as unknown, not false.",
})


def noul(instructions, value):
    return {"type": "noul", "instructions": instructions}, {"type": "noul", "noul": value}


def choice(instructions, options, picked, confidence=1.0):
    question = {"type": "choice", "instructions": instructions,
                "criteria": {option: option for option in options}}
    probabilities = {option: 0.0 for option in options}
    probabilities[picked] = confidence
    if confidence < 1.0:
        probabilities[next(o for o in options if o != picked)] = 1.0 - confidence
    return question, {"type": "choice", "choice": picked, "probabilities": probabilities,
                      "confidence": confidence}


def score(instructions):
    return ({"type": "score", "instructions": instructions, "criteria": ["0", "1", "2"]},
            {"type": "score", "score": 1.0, "legend": {"0": "0", "1": "1", "2": "2"},
             "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}, "confidence": 1.0})


def row(task="graph_navigation", state='{"nodes": ["N1", "N2"]}', **questions):
    return {
        "id": f"{task}:train:7", "task": task, "level": 0, "state": state,
        "questions": json.dumps({name: q for name, (q, _) in questions.items()}),
        "answers": json.dumps({name: a for name, (_, a) in questions.items()}),
    }


def test_pinned_constants():
    assert DATASET == "tasksource/procedural-typed-decisions"
    assert CONFIG == "all"
    assert REVISION == "e4b21b7b55f6cc9e76e33563e9d604ac5fcd25ad"
    assert LICENCE == "Apache-2.0"


def test_yes_no_noul_maps_to_violation_and_no_violation():
    cases, dropped = cases_from_row(row(
        reachable=noul("Does any valid route from start to finish exist?", 1.0),
        within_budget=noul("Does a valid route exist with total cost at most 4?", 0.0),
    ))
    assert not dropped
    reachable, budget = cases
    assert (reachable.verdict, budget.verdict) == ("VIOLATION", "NO_VIOLATION")
    assert reachable.criterion == "Does any valid route from start to finish exist?"
    assert reachable.evidence == '{"nodes": ["N1", "N2"]}'
    assert reachable.source_id == "graph_navigation:train:7:reachable"
    assert (reachable.source, reachable.surface, reachable.language) == (DATASET, "prompt", "en")


@pytest.mark.parametrize("task, name, value", [
    ("arithmetic", "random_is_long", 0.666667),
    ("arithmetic", "random_is_long", 1.0),  # probabilità esatta, non sì/no, anche se vale 1
    ("partial_observation_calibration", "incident_real", 0.441379),
    ("policy_under_uncertainty", "access_allowed", 0.0),
    ("graph_navigation", "reachable", 0.5),
])
def test_graded_noul_is_dropped_even_when_it_looks_binary(task, name, value):
    cases, dropped = cases_from_row(row(task=task, **{name: noul("How likely?", value)}))
    assert cases == []
    assert dropped == {"graded_noul": 1}


def test_policy_applicability_access_allowed_is_binary_and_kept():
    cases, _ = cases_from_row(row(task="policy_applicability", access_allowed=noul(
        "Does the governing policy allow the requested action?", 0.0)))
    assert [case.verdict for case in cases] == ["NO_VIOLATION"]


def test_multiway_choice_and_score_are_dropped_with_reason():
    cases, dropped = cases_from_row(row(
        first_hop=choice("What is the first hop?", ["N1", "N2", "none"], "N1"),
        shortest_cost=score("How many hops?"),
    ))
    assert cases == []
    assert dropped == {"choice_not_tristate": 1, "score_not_tristate": 1}


@pytest.mark.parametrize("picked, verdict", [
    ("ineligible", "VIOLATION"), ("eligible", "NO_VIOLATION"), ("undetermined", "INSUFFICIENT"),
])
def test_three_valued_eligibility_maps_to_tri_state(picked, verdict):
    [case], _ = cases_from_row(row(task="record_eligibility", state=ELIGIBILITY_STATE,
                                   probe_eligibility=choice(
                                       "What is the eligibility status of R2?",
                                       ["eligible", "ineligible", "undetermined"], picked)))
    assert case.verdict == verdict
    assert case.criterion == eligibility_criterion("R2")
    assert case.evidence == ELIGIBILITY_STATE


@pytest.mark.parametrize("picked, verdict", [
    ("violated", "VIOLATION"), ("met", "NO_VIOLATION"), ("unknown", "INSUFFICIENT"),
])
def test_three_valued_requirement_status_maps_to_tri_state(picked, verdict):
    [case], _ = cases_from_row(row(task="record_eligibility", state=ELIGIBILITY_STATE,
                                   requirement_status=choice(
                                       "For R2, is C1 met, violated or unknown?",
                                       ["met", "violated", "unknown"], picked)))
    assert case.verdict == verdict
    assert case.criterion == requirement_criterion("R2", "C1")


def test_tri_state_choice_with_unexpected_wording_or_soft_label_is_dropped():
    cases, dropped = cases_from_row(row(
        task="record_eligibility", state=ELIGIBILITY_STATE,
        probe_eligibility=choice("Is R2 eligible?", ["eligible", "ineligible", "undetermined"],
                                 "eligible"),
        requirement_status=choice("For R2, is C1 met, violated or unknown?",
                                  ["met", "violated", "unknown"], "met", confidence=0.7),
    ))
    assert cases == []
    assert dropped == {"unparsed_question": 1, "soft_label": 1}


def test_unknown_tri_state_value_and_unknown_type_are_errors():
    with pytest.raises(ValueError):
        cases_from_row(row(task="record_eligibility", probe_eligibility=choice(
            "What is the eligibility status of R2?", ["eligible", "maybe"], "maybe")))
    with pytest.raises(ValueError):
        cases_from_row(row(odd=({"type": "ranking", "instructions": "?"}, {"type": "ranking"})))


def test_convert_aggregates_cases_and_drops():
    rows = [row(reachable=noul("Reachable?", 1.0)), row(shortest_cost=score("Cost?"))]
    cases, dropped = convert(rows)
    assert len(cases) == 1
    assert dropped == {"score_not_tristate": 1}
