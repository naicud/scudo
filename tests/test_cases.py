import pytest

from scudo.cases import LANGUAGES, SURFACES, Case, training_rows


def case(**overrides):
    fields = {
        "source": "nvidia/Nemotron-Safety-Guard-Dataset-v3", "source_id": "a1",
        "language": "it", "surface": "prompt", "criterion": "Il testo chiede di fare danno?",
        "evidence": "Come si fa una torta?", "verdict": "NO_VIOLATION",
    }
    return Case(**{**fields, **overrides})


def test_surfaces_are_the_v1_coverage():
    assert SURFACES == (
        "prompt", "response", "injection_direct", "injection_indirect", "over_refusal",
    )


def test_languages_include_mixed_it_en():
    assert LANGUAGES == ("it", "en", "mixed")


@pytest.mark.parametrize("field, value", [
    ("surface", "tool"),
    ("surface", "input"),
    ("language", "de"),
    ("verdict", "SAFE"),
    ("criterion", "  "),
    ("source", ""),
    ("source_id", ""),
])
def test_case_rejects_values_outside_the_contract(field, value):
    with pytest.raises(ValueError):
        case(**{field: value})


def test_training_rows_carry_source_and_case_metadata():
    rows = training_rows([case(), case(source_id="b2", verdict="VIOLATION")], seed=0)
    assert {"prompt", "completion", "semantic_ids", "format", "source", "source_id",
            "language", "surface", "verdict"} <= set(rows[0])
    assert rows[0]["source"] == "nvidia/Nemotron-Safety-Guard-Dataset-v3"
    assert rows == training_rows([case(), case(source_id="b2", verdict="VIOLATION")], seed=0)
