from scudo.default_criteria import (
    AEGIS_CORE,
    AEGIS_FINE_GRAINED,
    DEFAULT_CRITERIA,
    INJECTION_DIRECT,
    INJECTION_INDIRECT,
    VERSION,
    by_id,
)
from scudo.protocol import SEMANTIC_IDS, render_prompt

AEGIS = [criterion for criterion in DEFAULT_CRITERIA if criterion.aegis_category]


def test_set_is_versioned_v1_and_each_criterion_is_at_version_0():
    assert VERSION == "scudo.default-criteria/1"
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
    categories = [criterion.aegis_category for criterion in AEGIS]
    assert sorted(categories) == sorted(AEGIS_CORE + AEGIS_FINE_GRAINED)
    assert all(criterion.id.startswith("aegis.") for criterion in AEGIS)
    ids = [criterion.id for criterion in DEFAULT_CRITERIA]
    assert len(set(ids)) == len(ids)


def test_injection_criteria_cover_the_two_v1_injection_surfaces():
    assert [c.id for c in DEFAULT_CRITERIA if not c.aegis_category] == [
        "injection.direct", "injection.indirect"]
    assert by_id("injection.direct") is INJECTION_DIRECT
    assert by_id("injection.indirect") is INJECTION_INDIRECT
    # Una richiesta dannosa posta in chiaro non è injection: la coprono i criteri Aegis.
    assert "harmful questions asked plainly" in INJECTION_DIRECT.text
    assert "third-party content" in INJECTION_INDIRECT.text
    assert "the user did not ask for" in INJECTION_INDIRECT.text


def test_criteria_are_semantic_questions_in_the_tracer_style():
    for criterion in DEFAULT_CRITERIA:
        assert criterion.text.startswith("Does the text ")
        assert "?" in criterion.text
        assert criterion.text.endswith("Missing essential context means INSUFFICIENT.")
        assert render_prompt(criterion.text, "Ciao", SEMANTIC_IDS)


def test_lookup_by_id():
    assert by_id("aegis.violence").aegis_category == "Violence"
