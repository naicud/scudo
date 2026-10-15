import re

import pytest

from scudo import notebook as tracer
from scudo.base_models import CANDIDATES, CONTROL, VERIFIED_ON, candidate
from scudo.notebook_cells import LORA

SHA = re.compile(r"[0-9a-f]{40}")


def test_shortlist_is_the_three_fast_candidates_with_qwen_as_control():
    assert list(CANDIDATES) == ["qwen35-4b", "gemma4-e4b", "ministral3-3b"]
    assert CONTROL == "qwen35-4b"
    assert [c.repo for c in CANDIDATES.values()] == [
        "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it", "mistralai/Ministral-3-3B-Base-2512"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", VERIFIED_ON)


def test_every_candidate_is_pinned_licensed_and_has_its_own_checkpoint_repo():
    repos = set()
    for slug, item in CANDIDATES.items():
        assert item.slug == slug
        assert SHA.fullmatch(item.revision)
        assert item.licence == "apache-2.0"
        assert item.checkpoint_repo == f"Naicud/scudo-bakeoff-{slug}"
        assert item.loader in ("FastLanguageModel", "FastModel")
        assert item.lora_target_modules and item.lora_modules > 0
        repos.add(item.checkpoint_repo)
    assert len(repos) == len(CANDIDATES)
    assert "Naicud/scudo-checkpoints" not in repos  # il checkpoint del tracer resta intatto


def test_control_reruns_the_tracer_base_and_targets():
    control = CANDIDATES[CONTROL]
    assert control.revision in tracer.CONFIG
    assert all(f'"{name}"' in LORA for name in control.lora_target_modules)
    assert control.bos_token_id is None and control.bos_token is None
    assert control.text_only and control.loader == "FastLanguageModel"
    # 24 linear_attention x 3 + 8 full_attention x 4 + 32 MLP x 3
    assert control.lora_modules == 24 * 3 + 8 * 4 + 32 * 3


def test_bos_rule_is_explicit_and_paired_with_the_token_text():
    for item in CANDIDATES.values():
        assert (item.bos_token is None) == (item.bos_token_id is None)
    assert (CANDIDATES["gemma4-e4b"].bos_token, CANDIDATES["gemma4-e4b"].bos_token_id) == ("<bos>", 2)
    assert (CANDIDATES["ministral3-3b"].bos_token,
            CANDIDATES["ministral3-3b"].bos_token_id) == ("<s>", 1)


def test_unverified_facts_are_declared():
    for slug in ("gemma4-e4b", "ministral3-3b"):
        assert any("NON VERIFICATO" in note for note in CANDIDATES[slug].notes)


def test_unknown_slug_is_refused_with_the_known_ones():
    with pytest.raises(KeyError, match="qwen35-4b"):
        candidate("llama-3b")
