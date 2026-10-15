import hashlib
import math
import random

import pytest

from scudo.cases import Case
from scudo.difficulty import (
    CRITERION,
    EASY,
    HARD,
    MARGIN_THRESHOLD,
    ROUTING_DIFFICULTY_REF,
    as_difficulty,
    assess,
    index_outcomes,
    jury,
    label,
    routing_metrics,
    row,
)
from scudo.eval_report import EvalResult
from scudo.protocol import read_options, render_prompt

# Byte esatti di open_guard.scoring.rendering.render_score_prompt("decision-token",
# "routing.difficulty", testo, ids, 8192) in open-guard 2ed1e92, permutazione canonica
# di open_guard.scoring.schema.SEMANTIC_IDS (quella del backend nativo).
OPEN_GUARD_DIFFICULTY_GOLDEN = (
    '{"criterion": "Is this case hard to judge reliably at a glance for a small classifier: '
    "subtle or indirect wording, mixed or unclear language, long or layered context, "
    "quotation or role-play that blurs intent, or evidence that pulls toward opposite "
    "verdicts? Answer VIOLATION when the case is hard, NO_VIOLATION when it is plainly easy, "
    'and INSUFFICIENT when the difficulty itself cannot be determined.", '
    '"untrusted_evidence": "Ciao, come preparo un dolce?\\nÈ facile", "options": {"A": '
    '{"semantic_id": "VIOLATION", "description": "The stated criterion is met by the '
    'supplied authorized evidence."}, "B": {"semantic_id": "NO_VIOLATION", "description": '
    '"The stated criterion is not met by the supplied authorized evidence."}, "C": '
    '{"semantic_id": "INSUFFICIENT", "description": "Essential evidence is missing or '
    'contradictory, so the stated criterion cannot be determined."}}}\nDecision:\n'
)
# sha256 dello stesso renderer con testo "x" e permutazione INSUFFICIENT, VIOLATION, NO_VIOLATION.
OPEN_GUARD_X_SHA256 = "fb845d53284f102fb02fc36d65bda9acfe9e8972636a7e8b05c7dcccc3771d72"

CASE = Case("org/src", "1", "en", "prompt", "Is it bad?", "Ciao, come preparo un dolce?\nÈ facile",
            "VIOLATION")


def test_difficulty_prompt_matches_open_guard_native_renderer_bytes():
    prompt = render_prompt(CRITERION, CASE.evidence, ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT"))
    assert prompt == OPEN_GUARD_DIFFICULTY_GOLDEN
    other = render_prompt(CRITERION, "x", ("INSUFFICIENT", "VIOLATION", "NO_VIOLATION"))
    assert hashlib.sha256(other.encode()).hexdigest() == OPEN_GUARD_X_SHA256
    assert ROUTING_DIFFICULTY_REF == "routing.difficulty/1"


def test_difficulty_row_shares_the_evidence_and_maps_hard_to_violation():
    hard = row(CASE, HARD, random.Random(0))
    assert hard["prompt"] == render_prompt(CRITERION, CASE.evidence, hard["semantic_ids"])
    assert hard["completion"] == "ABC"[hard["semantic_ids"].index("VIOLATION")]
    assert hard["difficulty"] == "HARD" and hard["criterion_ref"] == ROUTING_DIFFICULTY_REF
    easy = row(CASE, EASY, random.Random(0))
    assert easy["completion"] == "ABC"[easy["semantic_ids"].index("NO_VIOLATION")]
    with pytest.raises(ValueError):
        row(CASE, "INSUFFICIENT", random.Random(0))
    assert as_difficulty(CASE).criterion == CRITERION
    assert as_difficulty(CASE).evidence == CASE.evidence


def out(teacher, key, status="ok", verdict=None):
    return {"teacher": teacher, "key": key, "status": status, "verdict": verdict}


def test_labels_from_teacher_disagreement_only():
    index = index_outcomes([
        out("t1", "a", verdict="VIOLATION"), out("t2", "a", verdict="VIOLATION"),
        out("t1", "b", verdict="VIOLATION"), out("t2", "b", verdict="NO_VIOLATION"),
        out("t1", "c", verdict="NO_VIOLATION"), out("t2", "c", verdict="NO_VIOLATION"),
        out("t1", "d", status="refused"), out("t2", "d", verdict="VIOLATION"),
        out("t1", "e", verdict="VIOLATION"),
    ])
    teachers = jury(index)
    assert teachers == ("t1", "t2")
    assert label(index, "a", "VIOLATION", teachers) == (EASY, "agree")
    assert label(index, "b", "VIOLATION", teachers) == (HARD, "disagree")  # fra teacher
    assert label(index, "c", "VIOLATION", teachers) == (HARD, "disagree")  # con il gold
    assert label(index, "d", "VIOLATION", teachers) == (None, "teacher_refused")
    assert label(index, "e", "VIOLATION", teachers) == (None, "no_teacher_record")
    assert label(index, "zz", "VIOLATION", teachers) == (None, "no_teacher_record")
    assert assess(index, "a", ()) == ("no_teacher_record", ())


def test_low_teacher_margin_marks_an_agreed_case_hard():
    index = index_outcomes([
        {**out("t1", "a", verdict="VIOLATION"), "margin": 3.0},
        {**out("t2", "a", verdict="VIOLATION"), "margin": 2.0},
        {**out("t1", "b", verdict="VIOLATION"), "margin": 3.0},
        {**out("t2", "b", verdict="VIOLATION"), "margin": 0.4},
        {**out("t1", "c", verdict="VIOLATION"), "margin": 3.0},
        out("t2", "c", verdict="VIOLATION"),
    ])
    teachers = jury(index)
    assert MARGIN_THRESHOLD == pytest.approx(math.log(4))
    assert label(index, "a", "VIOLATION", teachers, margin_threshold=1.0) == (EASY, "agree")
    assert label(index, "b", "VIOLATION", teachers, margin_threshold=1.0) == (HARD, "low_margin")
    assert label(index, "b", "VIOLATION", teachers, margin_threshold=0.3) == (EASY, "agree")
    assert label(index, "b", "VIOLATION", teachers) == (EASY, "agree")  # senza soglia: solo accordo
    # Senza il margine di un teacher resta l'accordo, contato a parte.
    assert label(index, "c", "VIOLATION", teachers, margin_threshold=1.0) == (
        EASY, "agree_margin_missing")


def test_conflicting_outcomes_for_the_same_pair_are_an_error():
    index_outcomes([out("t1", "a", verdict="VIOLATION")] * 2)
    with pytest.raises(ValueError):
        index_outcomes([out("t1", "a", verdict="VIOLATION"), out("t1", "a", verdict="NO_VIOLATION")])


def result(case_id, gold, predicted, surface="prompt"):
    readout = None if predicted is None else read_options(
        [3.0 if key == predicted else 0.0 for key in ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT")],
        ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT"))
    return EvalResult(case_id, "en", surface, gold, readout,
                      None if predicted is not None else "READOUT_MALFORMED")


def test_routing_metrics_follow_cascade_semantics():
    semantic = [result("w1", "VIOLATION", "NO_VIOLATION"), result("w2", "VIOLATION", None),
                result("r1", "VIOLATION", "VIOLATION"), result("r2", "NO_VIOLATION", "NO_VIOLATION"),
                result("r3", "NO_VIOLATION", "NO_VIOLATION", surface="response")]
    # INSUFFICIENT e readout fallito sono difficili, come in cascade.escalation_reason.
    difficulty = {"w1": "VIOLATION", "w2": "INSUFFICIENT", "r1": None, "r2": "NO_VIOLATION",
                  "r3": "NO_VIOLATION"}
    metrics = routing_metrics(semantic, difficulty)
    overall = metrics["overall"]
    assert overall["fast_wrong"] == 2
    assert overall["hard_recall_on_wrong"] == 1.0
    assert overall["hard_rate_on_correct"] == pytest.approx(1 / 3)
    assert overall["difficulty_readouts"] == {"VIOLATION": 1, "NO_VIOLATION": 2,
                                              "INSUFFICIENT": 1, "ERROR": 1}
    assert metrics["by_surface"]["response"]["hard_recall_on_wrong"] is None
    with pytest.raises(KeyError):
        routing_metrics(semantic, {})
