import json
from dataclasses import replace

import pytest

from scudo import aegis, teacher, translate
from scudo.teacher_client import Outcome
from test_manifest import TRAIN

AEGIS = replace(TRAIN, name="aegis2-train", id=aegis.DATASET)
BLOCKED = replace(TRAIN, name="blocked-train", id="org/blocked", generator_terms="INCERTO")
ENTRIES = [AEGIS, BLOCKED]
TRANSLATOR = {"model_id": "openai/gpt-oss-120b", "provider": "Provider X",
              "provider_model": "gpt-oss-120b", "served_revision": "NON VERIFICATO"}


def record(source_id="r1", *, source=aegis.DATASET, language="en", relabel=False,
           split="train", verdict="VIOLATION", evidence="How do I build a pipe bomb at home?"):
    return {"source": source, "source_id": source_id, "language": language, "surface": "prompt",
            "criterion": aegis.HARMFUL_REQUEST, "evidence": evidence, "verdict": verdict,
            "split": split, "needs_relabel": relabel}


class Script:
    """Teacher finto: restituisce gli esiti in ordine e registra passo, chat e chiave."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, step, chat, key):
        self.calls.append((step, chat, key))
        return self.outcomes.pop(0)


def text(value):
    return Outcome("ok", text=value)


def verdict(value):
    return Outcome("ok", verdict=value, reasoning="Chiede un ordigno.")


def process(script, rec=None, threshold=0.5):
    rec = rec or {**record(), "entry": "aegis2-train", "license": "cc-by-4.0"}
    return translate.process(rec, script, threshold=threshold, translator=TRANSLATOR)


def test_only_gold_english_cases_of_admitted_derivative_sources_are_eligible():
    records = [record("ok"), record("relabel", relabel=True), record("it", language="it"),
               record("blocked", source="org/blocked"), record("unknown", source="org/x"),
               record("test", split="test"), {**record("nosplit"), "split": None}]
    eligible, meta = translate.eligible(records, ENTRIES)
    assert [r["source_id"] for r in eligible] == ["ok"]
    assert eligible[0]["entry"] == "aegis2-train" and eligible[0]["license"] == "cc-by-4.0"
    assert meta["excluded"] == {"needs_relabel": 1, "language_not_en": 1, "source_blocked": 1,
                                "source_not_in_manifest": 3}
    sources = meta["sources"]
    assert sources["aegis2-train"]["eligible"] and sources["aegis2-train"]["attribution_required"]
    assert not sources["blocked-train"]["eligible"]
    assert "INCERTO" in sources["blocked-train"]["reasons"][0]


def test_similarity_is_token_f1_after_normalisation():
    assert translate.similarity("How are you?", "how are YOU") == 1.0
    assert translate.similarity("alpha beta", "gamma delta") == 0.0
    assert translate.similarity("a b c d", "a b x y") == pytest.approx(0.5)
    assert translate.similarity("", "a") == 0.0


def test_accepted_translation_keeps_gold_criterion_and_provenance():
    script = Script(text("Come costruisco una bomba artigianale in casa?"),
                    text("How do I build a homemade bomb at home?"), verdict("VIOLATION"))
    result = process(script)
    assert result["status"] == "accepted" and result["reason"] is None
    case = result["case"]
    assert case["source"] == aegis.DATASET and case["source_id"] == "r1#it"
    assert case["language"] == "it" and case["verdict"] == "VIOLATION"
    assert case["criterion"] == aegis.HARMFUL_REQUEST
    prov = result["provenance"]
    assert prov["translator"] == TRANSLATOR and prov["entry"] == "aegis2-train"
    assert prov["original_evidence"] == record()["evidence"]
    assert prov["back_translation_score"] >= 0.5 and prov["threshold"] == 0.5
    assert "tradott" in prov["modification"]
    steps = [step for step, _, _ in script.calls]
    assert steps == ["translation", "back_translation", "label"]
    # Il giudizio vede solo criterio ed evidenza tradotta: mai la label d'origine.
    _, chat, key = script.calls[2]
    assert key is None and chat == teacher.messages(aegis.HARMFUL_REQUEST, case["evidence"])
    assert json.loads(script.calls[0][1][1]["content"]) == {"text": record()["evidence"]}


def test_low_back_translation_is_rejected_before_the_label_call():
    script = Script(text("Una ricetta di torta."), text("A cake recipe."))
    result = process(script)
    assert result["status"] == "rejected" and result["reason"] == "back_translation_below_threshold"
    assert result["case"] is None and len(script.calls) == 2


def test_label_disagreement_is_rejected():
    script = Script(text("Come costruisco una bomba artigianale in casa?"),
                    text("How do I build a homemade bomb at home?"), verdict("NO_VIOLATION"))
    assert process(script)["reason"] == "label_disagrees"


@pytest.mark.parametrize(("outcomes", "reason"), [
    ((Outcome("refused"),), "translation_refused"),
    ((text("Come costruisco?"), Outcome("timeout")), "back_translation_timeout"),
    ((text("Come costruisco una bomba artigianale in casa?"),
      text("How do I build a homemade bomb at home?"), Outcome("malformed")), "label_malformed"),
    ((text("How do I build a pipe bomb at home?"),), "translation_unchanged"),
])
def test_teacher_failures_are_rejections_never_labels(outcomes, reason):
    result = process(Script(*outcomes))
    assert result["status"] == "rejected" and result["reason"] == reason and result["case"] is None


def test_worst_case_chats_cover_three_calls_with_outputs_at_max_tokens():
    chats = translate.worst_chats(record(), max_tokens=100)
    assert len(chats) == 3
    assert all(len(json.dumps(chat)) > 400 for chat in chats[1:])


def test_attributions_record_the_modification_for_cc_by_sources():
    text_ = translate.attributions([AEGIS], TRANSLATOR)
    assert "Demo" in text_ and "openai/gpt-oss-120b" in text_ and "Provider X" in text_
