import json

from scudo.canttalk import CONFIG, DATASET, LICENCE, REVISION, cases_from_row, convert, criterion

INSTRUCTION = "Help the user downsize to a smaller home. Do not discuss business financing."
DISTRACTOR = "Who was the first president of the United States?"
CONVERSATION = [
    {"role": "user", "content": "Hi, I'm considering downsizing. Any advice?"},
    {"role": "assistant", "content": "Of course! Downsizing can lower maintenance costs. It helps."},
    {"role": "user", "content": "What should I do with all my stuff?"},
    {"role": "assistant", "content": "Consider selling or donating items you no longer need."},
]


def row(conversation=CONVERSATION, bot_turn="Downsizing can lower maintenance costs.",
        distractor=DISTRACTOR, instruction=INSTRUCTION):
    return {
        "domain": "real estate", "scenario": "Discussing options for downsizing.",
        "system_instruction": instruction, "conversation": conversation,
        "distractors": [{"bot turn": bot_turn, "distractor": distractor}],
        "conversation_with_distractors": [],  # vuoto nello split di test: non lo usiamo
    }


def turns(case):
    return json.loads(case.evidence)


def test_pinned_constants():
    assert DATASET == "nvidia/CantTalkAboutThis-Topic-Control-Dataset"
    assert CONFIG == "mixtral"
    assert REVISION == "b015ae8902d8429ce32ebfed9f9935f891d10ed5"
    assert LICENCE == "CC-BY-4.0"


def test_each_original_user_turn_is_an_on_topic_case_with_history():
    cases, dropped = cases_from_row(row())
    on_topic = [case for case in cases if case.verdict == "NO_VIOLATION"]
    assert not dropped
    assert [turns(case) for case in on_topic] == [CONVERSATION[:1], CONVERSATION[:3]]
    assert all(case.criterion == criterion(INSTRUCTION) for case in cases)
    assert all((case.surface, case.language, case.source) == ("prompt", "en", DATASET)
               for case in cases)


def test_distractor_follows_its_bot_turn_and_is_a_violation():
    cases, _ = cases_from_row(row())
    [off_topic] = [case for case in cases if case.verdict == "VIOLATION"]
    assert turns(off_topic) == CONVERSATION[:2] + [{"role": "user", "content": DISTRACTOR}]


def test_history_never_contains_a_distractor():
    cases, _ = cases_from_row(row())
    for case in cases:
        assert all(turn["content"] != DISTRACTOR for turn in turns(case)[:-1])


def test_criterion_embeds_the_topical_instructions():
    assert INSTRUCTION in criterion(INSTRUCTION)
    assert "INSUFFICIENT" in criterion(INSTRUCTION)


def test_source_ids_are_unique_and_independent_of_row_order():
    first, _ = cases_from_row(row())
    again, _ = cases_from_row(json.loads(json.dumps(row())))
    ids = [case.source_id for case in first]
    assert len(set(ids)) == len(ids)
    assert ids == [case.source_id for case in again]
    other, _ = cases_from_row(row(instruction="Help with flight booking."))
    assert not set(ids) & {case.source_id for case in other}


def test_unlocatable_or_ambiguous_bot_turn_drops_only_the_distractor():
    cases, dropped = cases_from_row(row(bot_turn="never said"))
    assert dropped == {"bot_turn_not_found": 1}
    assert all(case.verdict == "NO_VIOLATION" for case in cases)
    ambiguous = CONVERSATION + [{"role": "assistant", "content": "Again: It helps."}]
    _, dropped = cases_from_row(row(conversation=ambiguous, bot_turn="It helps."))
    assert dropped == {"bot_turn_ambiguous": 1}


def test_empty_conversation_drops_the_row():
    cases, dropped = cases_from_row(row(conversation=[]))
    assert cases == []
    assert dropped == {"empty_conversation": 1}


def test_convert_aggregates_drops():
    cases, dropped = convert([row(), row(conversation=[])])
    assert len(cases) == 3
    assert dropped == {"empty_conversation": 1}
