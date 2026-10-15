import random

import pytest

from scudo.nemotron import (
    HARMFUL_REQUEST,
    HARMFUL_RESPONSE,
    REVISION,
    cases_from_row,
    sample_cases,
    training_rows,
)


def row(id_="a" * 32, language="it", prompt="Come si fa una torta?", prompt_label="safe",
        response="None", response_label="", response_source="None"):
    return {
        "id": id_, "prompt": prompt, "response": response, "prompt_label": prompt_label,
        "response_label": response_label, "violated_categories": "",
        "prompt_label_source": "human", "response_label_source": response_source,
        "tag": "generic", "language": language, "reconstruction_id_if_redacted": None,
    }


def test_revision_is_pinned_commit():
    assert len(REVISION) == 40


def test_prompt_only_row_becomes_one_request_case():
    [case] = cases_from_row(row(prompt_label="unsafe"))
    assert case.criterion == HARMFUL_REQUEST
    assert case.evidence == "Come si fa una torta?"
    assert case.verdict == "VIOLATION"
    assert case.surface == "input"
    assert case.language == "it"
    assert case.source_id == "a" * 32


def test_row_with_labelled_response_adds_response_case_with_response_only_evidence():
    cases = cases_from_row(row(response="Ecco la ricetta.", response_label="safe",
                               response_source="human"))
    request, response = cases
    assert request.verdict == "NO_VIOLATION"
    assert response.criterion == HARMFUL_RESPONSE
    assert response.evidence == "Ecco la ricetta."  # come la superficie output di open-guard
    assert response.verdict == "NO_VIOLATION"
    assert response.surface == "output"


def test_refusal_data_augmentation_rows_are_dropped_entirely():
    assert cases_from_row(row(response="No.", response_label="safe",
                              response_source="refusal_data_augmentation")) == []


def test_redacted_prompts_are_dropped():
    assert cases_from_row(row(prompt="REDACTED")) == []


@pytest.mark.parametrize("language", ["de", "zh-CN"])
def test_languages_outside_it_en_are_dropped(language):
    assert cases_from_row(row(language=language)) == []


def test_unknown_label_is_an_error_not_a_no_violation():
    with pytest.raises(ValueError):
        cases_from_row(row(prompt_label="maybe"))


def test_response_without_label_is_skipped_but_request_kept():
    cases = cases_from_row(row(response="testo", response_label="", response_source="None"))
    assert [case.surface for case in cases] == ["input"]


def rows(language, count):
    return [row(id_=f"{language}{i:030d}", language=language,
                prompt_label="unsafe" if i % 3 == 0 else "safe") for i in range(count)]


def test_sample_is_deterministic_and_balanced_by_language():
    source = rows("it", 50) + rows("en", 80)
    first = sample_cases(source, per_language=20, seed=1)
    again = sample_cases(list(reversed(source)), per_language=20, seed=1)
    assert first == again
    assert sum(case.language == "it" for case in first) == 20
    assert sum(case.language == "en" for case in first) == 20
    assert first != sample_cases(source, per_language=20, seed=2)


def test_sample_fails_when_a_language_is_short():
    with pytest.raises(ValueError):
        sample_cases(rows("it", 5) + rows("en", 50), per_language=10, seed=0)


def test_training_rows_carry_case_metadata_and_protocol_fields():
    cases = sample_cases(rows("it", 10) + rows("en", 10), per_language=4, seed=0)
    out = training_rows(cases, seed=0)
    assert len(out) == 8
    assert {"prompt", "completion", "semantic_ids", "format", "language", "source_id",
            "surface", "verdict"} <= set(out[0])
    assert out == training_rows(cases, seed=0)
    assert out != training_rows(cases, seed=1)


def test_training_rows_do_not_touch_global_random_state():
    cases = sample_cases(rows("it", 10) + rows("en", 10), per_language=4, seed=0)
    random.seed(5)
    expected = random.random()
    random.seed(5)
    training_rows(cases, seed=0)
    assert random.random() == expected


def test_heldout_excludes_training_sources_and_identical_texts():
    from scudo.nemotron import heldout_cases

    train = sample_cases(rows("it", 10) + rows("en", 10), per_language=5, seed=0)
    test_rows = rows("it", 10) + rows("en", 10)  # stessi id e testi del training
    test_rows += [row(id_=f"t{i:031d}", language=lang, prompt=f"nuovo {lang} {i}")
                  for lang in ("it", "en") for i in range(6)]
    test_rows += [row(id_="d" * 32, language="it", prompt=train[0].evidence)]
    held = heldout_cases(test_rows, train, per_language=6, seed=0)
    train_ids = {case.source_id for case in train}
    train_texts = {case.evidence for case in train}
    assert len(held) == 12
    assert not any(case.source_id in train_ids for case in held)
    assert not any(case.evidence in train_texts for case in held)
