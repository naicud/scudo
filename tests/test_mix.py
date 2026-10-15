import json
from collections import Counter

import pytest

from scudo import canttalk
from scudo.aegis import HARMFUL_REQUEST
from scudo.cases import Case
from scudo.default_criteria import INJECTION_DIRECT
from scudo.difficulty import CRITERION, MARGIN_THRESHOLD, index_outcomes
from scudo.mix import MixError, Targets, build, category, quotas
from scudo.mix_clean import dedup, generated, method, relabel
from scudo.mix_pool import PoolCase

CUSTOM = "Does the last user turn leave the allowed topic? Missing essential context means INSUFFICIENT."
EN_ONLY = {"it": 0, "en": 1, "mixed": 0}


def pc(entry, source, i, surface, criterion, evidence, verdict, relabel_=False, language="en"):
    return PoolCase(Case(source, str(i), language, surface, criterion, evidence, verdict),
                    entry, relabel_)


def dialog(i):
    return json.dumps([{"role": "assistant", "content": f"ciao {i}"},
                       {"role": "user", "content": f"domanda {i}"}])


def pool(n=60):
    cases = []
    for i in range(n):
        verdict = ("VIOLATION", "NO_VIOLATION")[i % 2]
        cases += [
            pc("harm", "org/harm", i, "prompt", HARMFUL_REQUEST, f"richiesta {i}", verdict),
            pc("inj", "org/inj", i, "injection_direct", INJECTION_DIRECT.text, f"jailbreak {i}",
               verdict, relabel_=verdict == "NO_VIOLATION"),
            pc("talk", canttalk.DATASET, i, "prompt", CUSTOM, dialog(i), verdict),
            pc("refusal", "org/refusal", i, "over_refusal", HARMFUL_REQUEST, f"benigno {i}",
               "NO_VIOLATION"),
            pc("proc", "org/proc", i, "prompt", CUSTOM, f"stato {i}", "INSUFFICIENT"),
        ]
    return cases


def outcomes(cases, *, disagree_every=3, skip=()):
    """Due teacher: concordi con il gold, tranne un caso su `disagree_every`."""
    records = []
    for n, p in enumerate(cases):
        if p.key in skip:
            continue
        other = "NO_VIOLATION" if p.case.verdict != "NO_VIOLATION" else "VIOLATION"
        second = other if n % disagree_every == 0 else p.case.verdict
        records += [{"teacher": "t1", "key": p.key, "status": "ok", "verdict": p.case.verdict},
                    {"teacher": "t2", "key": p.key, "status": "ok", "verdict": second}]
    return index_outcomes(records)


def judged(cases):
    clean = dedup(cases)[0]
    return outcomes(clean + generated(clean)[0])


def test_category_rules_follow_the_documented_order():
    assert category(pc("a", "s", 1, "prompt", HARMFUL_REQUEST, "x", "INSUFFICIENT").case) == "insufficient"
    assert category(pc("a", "s", 1, "over_refusal", CUSTOM, "x", "NO_VIOLATION").case) == "over_refusal"
    assert category(pc("a", "s", 1, "injection_indirect", CUSTOM, "x", "VIOLATION").case) == "injection"
    assert category(pc("a", "s", 1, "prompt", INJECTION_DIRECT.text, "x", "VIOLATION").case) == "injection"
    assert category(pc("a", "s", 1, "response", HARMFUL_REQUEST, "x", "VIOLATION").case) == "prompt_response"
    assert category(pc("a", "s", 1, "prompt", CUSTOM, "x", "VIOLATION").case) == "custom"


def test_dedup_keeps_gold_copy_and_drops_both_sides_of_a_conflict():
    a = pc("aegis1", "org/a1", 1, "prompt", HARMFUL_REQUEST, "Come si fa?", "VIOLATION")
    b = pc("aegis2", "org/a2", 9, "prompt", HARMFUL_REQUEST, "come si fa", "VIOLATION", relabel_=True)
    c = pc("aegis2", "org/a2", 10, "prompt", HARMFUL_REQUEST, "COME SI FA!", "VIOLATION")
    w = pc("wild-j", "org/w", 1, "injection_direct", INJECTION_DIRECT.text, "DAN", "VIOLATION")
    r = pc("wild-r", "org/w", 2, "injection_direct", INJECTION_DIRECT.text, "dan", "NO_VIOLATION")
    other = pc("aegis2", "org/a2", 11, "prompt", INJECTION_DIRECT.text, "come si fa", "NO_VIOLATION")
    kept, drops, conflicts = dedup([b, a, c, w, r, other])
    assert set(kept) == {a, other}  # criterio diverso: non è un duplicato
    assert drops == Counter({"duplicate_cross_source": 2, "label_conflict": 2})
    assert conflicts == Counter({"wild-j|wild-r": 1})


def test_relabel_needs_unanimous_confirmation_of_the_label():
    cases = [pc("r", "org/r", i, "injection_direct", INJECTION_DIRECT.text, f"t{i}",
                "NO_VIOLATION", relabel_=True) for i in range(4)]
    index = index_outcomes([
        {"teacher": "t1", "key": cases[0].key, "status": "ok", "verdict": "NO_VIOLATION"},
        {"teacher": "t2", "key": cases[0].key, "status": "ok", "verdict": "NO_VIOLATION"},
        {"teacher": "t1", "key": cases[1].key, "status": "ok", "verdict": "VIOLATION"},
        {"teacher": "t2", "key": cases[1].key, "status": "ok", "verdict": "VIOLATION"},
        {"teacher": "t1", "key": cases[2].key, "status": "timeout", "verdict": None},
        {"teacher": "t2", "key": cases[2].key, "status": "ok", "verdict": "NO_VIOLATION"},
    ])
    kept, counts = relabel(cases, index, ("t1", "t2"))
    assert kept == [cases[0]]  # una giuria discorde non cambia la label: esclude il caso
    assert counts == Counter({"relabel_confirmed": 1, "relabel_jury_disagrees": 1,
                              "relabel_teacher_timeout": 1, "relabel_no_teacher_record": 1})


def test_generated_insufficient_keeps_the_parent_entry_and_method():
    children, _ = generated([pc("talk", canttalk.DATASET, 1, "prompt", CUSTOM, dialog(1), "VIOLATION")])
    [child] = children
    assert child.entry == "talk" and child.case.verdict == "INSUFFICIENT"
    assert method(child) == "evidence_removed"
    assert method(pc("proc", "org/proc", 1, "prompt", CUSTOM, "s", "INSUFFICIENT")) == "native:proc"


def test_mix_fills_every_stratum_and_caps_each_insufficient_origin():
    cases = pool()
    targets = Targets(total=100, languages=EN_ONLY, method_cap=0.5)
    rows, stats = build(cases, outcomes=judged(cases), targets=targets, seed=1)
    semantic = [r for r in rows if r["task"] == "semantic"]
    assert len(semantic) == 100
    by_category = Counter(r["category"] for r in semantic)
    wanted = Counter()
    for (cat, _), n in quotas(targets).items():
        wanted[cat] += n
    assert {cat: by_category[cat] for cat in wanted} == dict(wanted)
    assert by_category["insufficient"] == 12
    assert stats["insufficient"]["by_origin"] == {"evidence_removed": 6, "native:proc": 6}
    assert stats["overrides"] == {"languages": {"default": {"it": 0.4, "en": 0.4, "mixed": 0.2},
                                                "value": EN_ONLY},
                                  "total": {"default": 200_000, "value": 100},
                                  "method_cap": {"default": 0.4, "value": 0.5}}
    assert stats["relabel"]["relabel_confirmed"] and stats["relabel"]["relabel_jury_disagrees"]


def test_difficulty_rows_share_the_evidence_of_their_semantic_row():
    cases = pool()
    rows, stats = build(cases, outcomes=judged(cases),
                        targets=Targets(total=100, languages=EN_ONLY, method_cap=0.5), seed=1)
    semantic = {r["key"]: r for r in rows if r["task"] == "semantic"}
    hard = [r for r in rows if r["task"] == "difficulty"]
    assert len(hard) == len(semantic)
    for row in hard:
        evidence = json.loads(row["prompt"].rsplit("\nDecision:\n", 1)[0])
        parent = json.loads(semantic[row["key"]]["prompt"].rsplit("\nDecision:\n", 1)[0])
        assert evidence["criterion"] == CRITERION
        assert evidence["untrusted_evidence"] == parent["untrusted_evidence"]
        assert row["verdict"] == ("VIOLATION" if row["difficulty"] == "HARD" else "NO_VIOLATION")
    assert set(stats["composition"]["difficulty"]["by_difficulty"]) == {"HARD", "EASY"}


def test_low_teacher_margin_turns_agreed_cases_hard_and_is_recorded():
    cases = pool()
    clean = dedup(cases)[0]
    every_easy = outcomes(clean + generated(clean)[0], disagree_every=10 ** 9)
    for n, by_teacher in enumerate(every_easy.values()):
        by_teacher["t1"]["margin"] = 0.5 if n % 4 == 0 else 3.0
        by_teacher["t2"]["margin"] = 3.0
    targets = Targets(total=100, languages=EN_ONLY, method_cap=0.5)
    rows, stats = build(cases, outcomes=every_easy, targets=targets, seed=1)
    difficulty_stats = stats["composition"]["difficulty"]
    assert stats["difficulty_margin_threshold"] == pytest.approx(MARGIN_THRESHOLD)
    assert set(difficulty_stats["by_reason"]) == {"agree", "low_margin"}
    assert difficulty_stats["by_reason"]["low_margin"] == difficulty_stats["by_difficulty"]["HARD"]
    with pytest.raises(MixError, match="difficoltà"):  # senza soglia nessun HARD
        build(cases, outcomes=every_easy, targets=targets, seed=1, margin_threshold=None)


def test_mix_is_deterministic_and_independent_of_input_order():
    cases = pool()
    targets = Targets(total=100, languages=EN_ONLY, method_cap=0.5)
    first = build(cases, outcomes=judged(cases), targets=targets, seed=1)
    again = build(cases[::-1], outcomes=judged(cases), targets=targets, seed=1)
    assert first == again


def test_unfillable_strata_fail_listing_every_shortfall():
    cases = pool()
    with pytest.raises(MixError) as error:
        build(cases, outcomes=judged(cases), targets=Targets(total=100, method_cap=0.5), seed=1)
    assert "prompt_response/it" in str(error.value) and "custom/mixed" in str(error.value)


def test_method_cap_too_tight_for_the_insufficient_quota_fails():
    cases = pool()
    with pytest.raises(MixError, match="INSUFFICIENT"):
        build(cases, outcomes=judged(cases), targets=Targets(total=100, languages=EN_ONLY), seed=1)


def test_mix_without_hard_and_easy_teacher_labels_is_refused():
    cases = pool()
    clean = dedup(cases)[0]
    every_easy = outcomes(clean + generated(clean)[0], disagree_every=10 ** 9)
    with pytest.raises(MixError, match="difficoltà"):
        build(cases, outcomes=every_easy,
              targets=Targets(total=100, languages=EN_ONLY, method_cap=0.5), seed=1)


@pytest.mark.parametrize("bad", [
    {"total": 0}, {"languages": {"it": 0.5, "en": 0.4, "mixed": 0.2}},
    {"categories": {"prompt_response": 1}}, {"insufficient_share": 0}, {"method_cap": 1.5},
])
def test_invalid_targets_are_refused(bad):
    with pytest.raises(MixError):
        Targets(**bad)
