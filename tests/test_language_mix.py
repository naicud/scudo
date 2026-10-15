import pytest

from scudo import language_mix, spotcheck_it
from scudo.spotcheck_it import SpotCheckError
from test_spotcheck_it import fill, item, pool
from test_translate import ENTRIES, record


def case(i, language):
    return {"key": f"{language}{i}", "origin": "dataset", "source": "org/x", "source_id": str(i),
            "language": language, "surface": "prompt", "criterion": "Is it unsafe?",
            "evidence": f"text {i}", "verdict": "NO_VIOLATION"}


POOLS = {"it": [case(i, "it") for i in range(10)], "en": [case(i, "en") for i in range(10)],
         "mixed": [case(i, "mixed") for i in range(5)]}


def test_default_shares_are_40_40_20():
    assert language_mix.DEFAULT_SHARES == {"it": 0.4, "en": 0.4, "mixed": 0.2}
    assert language_mix.quotas(10, language_mix.DEFAULT_SHARES) == {"it": 4, "en": 4, "mixed": 2}
    assert language_mix.quotas(7, language_mix.DEFAULT_SHARES) == {"it": 3, "en": 3, "mixed": 1}


def test_shares_are_configurable_and_validated():
    assert language_mix.parse_shares("it=0.5,en=0.5") == {"it": 0.5, "en": 0.5, "mixed": 0.0}
    for bad in ("it=0.5,en=0.4", "it=0.5,de=0.5", "it=-0.2,en=1.2", "it=x", "it=1,it=0"):
        with pytest.raises(ValueError):
            language_mix.parse_shares(bad)


def test_mix_is_deterministic_order_independent_and_exact():
    mix, meta = language_mix.build_mix(POOLS, 10, language_mix.DEFAULT_SHARES, seed=1)
    reversed_pools = {k: list(reversed(v)) for k, v in POOLS.items()}
    assert language_mix.build_mix(reversed_pools, 10, language_mix.DEFAULT_SHARES, seed=1)[0] == mix
    assert [r["language"] for r in mix].count("mixed") == 2
    assert meta["quotas"] == {"it": 4, "en": 4, "mixed": 2} and meta["available"]["mixed"] == 5
    assert language_mix.build_mix(POOLS, 10, language_mix.DEFAULT_SHARES, seed=2)[0] != mix


def test_a_pool_too_small_fails_instead_of_under_filling():
    with pytest.raises(ValueError, match="mixed: 5 casi, ne servono 6"):
        language_mix.build_mix(POOLS, 30, language_mix.DEFAULT_SHARES, seed=0)


def test_dataset_pool_keeps_only_gold_cases_of_admitted_sources():
    records = [record("ok"), record("relabel", relabel=True), record("b", source="org/blocked"),
               record("u", source="org/unknown")]
    kept, excluded = language_mix.dataset_pool(records, ENTRIES)
    assert [r["source_id"] for r in kept] == ["ok"] and kept[0]["origin"] == "dataset"
    assert excluded == {"needs_relabel": 1, "source_blocked": 1, "source_not_in_manifest": 1}


def test_batch_enters_only_with_an_accepted_spot_check(tmp_path):
    out = pool(tmp_path)
    with pytest.raises(SpotCheckError):
        language_mix.batch_pool(out)
    spotcheck_it.export(out, n=20, seed=0)
    spotcheck_it.import_review(out, fill(out, tmp_path, 2), by="Giulia")  # 90%: respinto
    with pytest.raises(SpotCheckError, match="respinto"):
        language_mix.batch_pool(out)


def test_accepted_batch_brings_italian_and_mixed_items(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    spotcheck_it.import_review(out, fill(out, tmp_path, 0), by="Giulia")
    records = language_mix.batch_pool(out)
    assert {r["language"] for r in records} == {"it", "en", "mixed"}
    assert all(r["origin"] == "generated" and r["provenance"] for r in records)
    assert len(records) == 32 and records[0]["key"] == item(0)["key"]
