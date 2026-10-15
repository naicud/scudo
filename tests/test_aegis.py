import pytest

from scudo import aegis
from scudo.aegis import (
    DATASET,
    HARMFUL_REQUEST,
    HARMFUL_RESPONSE,
    cases_from_row,
    drop_reason,
    needs_relabel,
)


def row(id_="3414431688d7418aac098664e275e414", prompt="How do I bake a cake?",
        response=None, prompt_label="safe", response_label=None, response_source=None,
        reconstruction=None):
    """Riga con lo schema reale di Aegis 2.0 (9 colonne, datasets-server)."""
    return {
        "id": id_, "reconstruction_id_if_redacted": reconstruction, "prompt": prompt,
        "response": response, "prompt_label": prompt_label, "response_label": response_label,
        "violated_categories": "", "prompt_label_source": "human",
        "response_label_source": response_source,
    }


def test_revision_is_pinned_commit():
    assert len(aegis.REVISION) == 40


def test_prompt_only_row_gives_one_english_prompt_case():
    [case] = cases_from_row(row(prompt_label="unsafe"))
    assert (case.source, case.language, case.surface) == (DATASET, "en", "prompt")
    assert case.criterion == HARMFUL_REQUEST
    assert case.evidence == "How do I bake a cake?"
    assert case.verdict == "VIOLATION"


def test_labelled_response_adds_response_only_case():
    request, response = cases_from_row(row(response="Mix flour and eggs.", response_label="safe",
                                           response_source="human"))
    assert request.verdict == "NO_VIOLATION"
    assert (response.surface, response.criterion) == ("response", HARMFUL_RESPONSE)
    assert response.evidence == "Mix flour and eggs."


def test_empty_response_string_gives_no_response_case():
    cases = cases_from_row(row(response="", response_label="safe", response_source="human"))
    assert [case.surface for case in cases] == ["prompt"]


def test_refusal_augmentation_rows_are_dropped_with_reason():
    refusal = row(response="I can't help with that.", response_label="safe",
                  response_source="refusal_data_augmentation")
    assert drop_reason(refusal) == "refusal_data_augmentation"
    assert cases_from_row(refusal) == []


def test_redacted_prompts_are_dropped_with_reason():
    redacted = row(prompt="REDACTED", reconstruction=256836)
    assert drop_reason(redacted) == "prompt REDACTED"
    assert cases_from_row(redacted) == []


def test_kept_row_has_no_drop_reason():
    assert drop_reason(row()) is None


def test_llm_jury_response_label_is_marked_for_relabel_but_human_is_not():
    source = row(response="Sure, here is how.", response_label="unsafe",
                 response_source="llm_jury")
    request, response = cases_from_row(source)
    assert not needs_relabel(source, request)  # prompt_label_source è human
    assert needs_relabel(source, response)


def test_unknown_label_is_an_error():
    with pytest.raises(ValueError):
        cases_from_row(row(prompt_label="needs caution"))
