import json

import pytest

from scudo.open_jev import CONFIG, DATASET, LICENCE, REVISION, cases_from_row, convert

QUESTION = (
    "Apply this local, explicitly stated workflow policy to the supplied facts.\n"
    "Rule: Refund only when the customer requests a refund.\n\n"
    "Should action REFUND be selected now?"
)
STATE = {
    "conversation": [{"role": "customer", "text": "Please freeze my card."}],
    "identity": {"account_id": "Case-d5659f1ee5b716f3", "verified": True},
}
# Le label privilegiate stanno in metadata_json e record_json: non devono mai finire nel caso.
SECRET = "TEACHER-LABEL-SENTINEL"


def row(kind="noul", source="workflow-controls-v1/customer_service", options=("no", "yes"),
        target=(0.0, 1.0), state=STATE, question=QUESTION, id_="wc:customer_service:42:1:v0:REFUND"):
    metadata = {"question_id": "REFUND", "target_basis": SECRET}
    record = {"id": id_, "kind": kind, "metadata": metadata, "state": state, "target": list(target)}
    return {
        "id": id_, "group_id": "wc:customer_service:42:1", "split": "train", "source": source,
        "kind": kind, "question": question, "options": list(options), "target": list(target),
        "state_json": json.dumps(state), "metadata_json": json.dumps(metadata),
        "record_json": json.dumps(record), "original_line_number": 1,
    }


def test_pinned_redistributable_config_and_revision():
    assert DATASET == "ZefanCai/Open-Jev"
    assert CONFIG == "release-v2-redistributable"
    assert REVISION == "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
    assert LICENCE == "CC0-1.0"


def test_noul_yes_is_violation_with_question_as_criterion_and_state_as_evidence():
    [case], dropped = cases_from_row(row(target=(0.0, 1.0)))
    assert not dropped
    assert case.verdict == "VIOLATION"
    assert case.criterion == QUESTION
    assert case.evidence == json.dumps(STATE, ensure_ascii=False)
    assert (case.source, case.source_id) == (DATASET, "wc:customer_service:42:1:v0:REFUND")
    assert (case.language, case.surface) == ("en", "prompt")


def test_noul_no_is_no_violation():
    [case], _ = cases_from_row(row(target=(1.0, 0.0)))
    assert case.verdict == "NO_VIOLATION"


def test_string_state_is_kept_verbatim():
    [case], _ = cases_from_row(row(state="Customer message: I need access restored."))
    assert case.evidence == "Customer message: I need access restored."


def test_privileged_metadata_never_reaches_criterion_or_evidence():
    [case], _ = cases_from_row(row())
    assert SECRET not in case.criterion and SECRET not in case.evidence


@pytest.mark.parametrize("overrides, reason", [
    ({"kind": "choice", "options": ("a", "b", "c"), "target": (1.0, 0.0, 0.0)}, "choice_not_tristate"),
    ({"kind": "score", "options": ("low", "high"), "target": (0.0, 1.0)}, "score_not_tristate"),
    ({"target": (0.4, 0.6)}, "soft_target"),
    ({"options": ("yes", "no")}, "unexpected_options"),
    ({"source": "customer-control-v1"}, "unverified_upstream_licence"),
])
def test_rows_that_do_not_map_faithfully_are_dropped_with_reason(overrides, reason):
    cases, dropped = cases_from_row(row(**overrides))
    assert cases == []
    assert dropped == {reason: 1}


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        cases_from_row(row(kind="ranking"))


def test_convert_counts_every_dropped_row():
    rows = [row(), row(id_="b", target=(1.0, 0.0)), row(id_="c", kind="score"),
            row(id_="d", source="customer-control-v1")]
    cases, dropped = convert(rows)
    assert [case.verdict for case in cases] == ["VIOLATION", "NO_VIOLATION"]
    assert dropped == {"score_not_tristate": 1, "unverified_upstream_licence": 1}
    assert len(cases) + sum(dropped.values()) == len(rows)
