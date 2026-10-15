from scudo.default_criteria import (
    AEGIS_CORE,
    AEGIS_FINE_GRAINED,
    DEFAULT_CRITERIA,
    VERSION,
    by_id,
)
from scudo.protocol import SEMANTIC_IDS, render_prompt


def test_set_is_versioned_v0():
    assert VERSION == "scudo.default-criteria/0"
    assert all(criterion.version == 0 for criterion in DEFAULT_CRITERIA)


def test_aegis_taxonomy_as_published_in_the_dataset_card():
    # Table 2 del paper e README del dataset: 12 categorie core, 9 fine-grained.
    assert AEGIS_CORE == (
        "Hate/Identity Hate", "Sexual", "Suicide and Self Harm", "Violence",
        "Guns/Illegal Weapons", "Threat", "PII/Privacy", "Sexual Minor",
        "Criminal Planning/Confessions", "Harassment", "Controlled/Regulated substances",
        "Profanity",
    )
    assert AEGIS_FINE_GRAINED == (
        "Illegal Activity", "Immoral/Unethical", "Unauthorized Advice",
        "Political/Misinformation/Conspiracy", "Fraud/Deception",
        "Copyright/Trademark/Plagiarism", "High Risk Gov. Decision Making", "Malware",
        "Manipulation",
    )


def test_one_criterion_per_risk_category():
    categories = [criterion.aegis_category for criterion in DEFAULT_CRITERIA]
    assert sorted(categories) == sorted(AEGIS_CORE + AEGIS_FINE_GRAINED)
    ids = [criterion.id for criterion in DEFAULT_CRITERIA]
    assert len(set(ids)) == len(ids)
    assert all(id_.startswith("aegis.") for id_ in ids)


def test_criteria_are_semantic_questions_in_the_tracer_style():
    for criterion in DEFAULT_CRITERIA:
        assert criterion.text.startswith("Does the text ")
        assert "?" in criterion.text
        assert criterion.text.endswith("Missing essential context means INSUFFICIENT.")
        assert render_prompt(criterion.text, "Ciao", SEMANTIC_IDS)


def test_lookup_by_id():
    assert by_id("aegis.violence").aegis_category == "Violence"
