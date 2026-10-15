import hashlib
import json
import math
import subprocess
import urllib.request
from collections import Counter
from dataclasses import asdict, replace
from functools import partial

import pytest

from scudo import italian_run, mix_cli, pilot, spotcheck_it
from scudo.budget import BudgetError, Ledger, confirm_volume
from scudo.mix import MixError, Targets
from scudo.spotcheck_it import SpotCheckError
from scudo.mix_pool import PoolError, batch_pool, load_ingest, load_italian
from scudo.teacher import CANDIDATES
from test_manifest import TRAIN
from test_mix import EN_ONLY, judged, pool
from test_pilot import toml
from test_spotcheck_it import fill, item
from scudo.teacher import load_configs
from test_teacher import CONFIG

DIGEST = "a" * 64
GPT = "openai/gpt-oss-120b"
TEACHER = replace(TRAIN, name="teacher-gpt-oss", id=GPT, revision=CANDIDATES[GPT].revision,
                  splits=("generated", "translated", "reasoning"), license="apache-2.0")


def entries_for(cases):
    names = {}
    for p in cases:
        names.setdefault(p.entry, p.case.source)
    return [replace(TRAIN, name=name, id=source) for name, source in names.items()]


def write_ingest(out, cases, digest=DIGEST):
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for p in cases:
        record = {**asdict(p.case), "split": "train", "needs_relabel": p.needs_relabel}
        with (out / f"{p.entry}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        sources.setdefault(p.entry, {"kept": 0})["kept"] += 1
    (out / "report.json").write_text(json.dumps({"manifest_sha256": digest, "sources": sources}))
    return out


def test_ingest_outputs_load_with_their_provenance(tmp_path):
    cases = pool(4)
    loaded, info = load_ingest(write_ingest(tmp_path / "in", cases), entries=entries_for(cases),
                               manifest_sha256=DIGEST)
    assert sorted(p.key for p in loaded) == sorted(p.key for p in cases)
    assert info["manifest_sha256"] == DIGEST and set(info["sources"]) == {p.entry for p in cases}


def test_ingest_from_another_manifest_or_blocked_source_is_refused(tmp_path):
    cases = pool(2)
    out = write_ingest(tmp_path / "in", cases)
    with pytest.raises(PoolError, match="manifest"):
        load_ingest(out, entries=entries_for(cases), manifest_sha256="b" * 64)
    blocked = [replace(e, generator_terms="INCERTO") for e in entries_for(cases)]
    with pytest.raises(PoolError, match="cancello"):
        load_ingest(out, entries=blocked, manifest_sha256=DIGEST)
    with (out / "harm.jsonl").open("a") as handle:
        handle.write(json.dumps({**asdict(cases[0].case), "source_id": "x",
                                 "needs_relabel": False}) + "\n")
    with pytest.raises(PoolError, match="report"):
        load_ingest(out, entries=entries_for(cases), manifest_sha256=DIGEST)


def italian_batch(tmp_path, records, *, accept=True):
    out = tmp_path / "it-batch"
    out.mkdir(parents=True)
    (out / italian_run.POOL).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    if accept:
        spotcheck_it.export(out, n=20, seed=0)
        spotcheck_it.import_review(out, fill(out, tmp_path, 0), by="Giulia")
    return out


def translated(i, model=GPT):
    return {**item(i), "key": f"tr:harm:{i}:prompt", "origin": "translated", "source": "org/harm",
            "source_id": f"{i}#it", "provenance": {"kind": "translated", "entry": "harm",
                                                   "translator": {"model_id": model}}}


def test_italian_batch_enters_only_with_an_accepted_spot_check(tmp_path):
    entries = entries_for(pool(1)) + [TEACHER]
    records = [item(i) for i in range(25)] + [translated(i) for i in range(5)]
    with pytest.raises(SpotCheckError):
        load_italian(italian_batch(tmp_path / "a", records, accept=False), entries=entries)
    with pytest.raises(SpotCheckError):
        batch_pool(tmp_path / "a" / "it-batch")
    assert len(batch_pool(italian_batch(tmp_path / "b", records))) == 30
    loaded, info = load_italian(tmp_path / "b" / "it-batch", entries=entries)
    assert Counter(p.entry for p in loaded) == {"teacher-gpt-oss": 25, "harm": 5}
    assert info["pool_sha256"] == info["spotcheck_pool_sha256"]
    assert info["by_origin_language"] == {"generated/it": 25, "translated/it": 5}


@pytest.mark.parametrize(("entries", "match"), [
    ([], "teacher_not_in_manifest"),
    ([replace(TEACHER, generator_terms="INCERTO")], "teacher_blocked"),
    ([replace(TEACHER, revision="0" * 40)], "teacher_revision_differs"),
])
def test_teacher_text_enters_only_through_an_admitted_teacher_entry(tmp_path, entries, match):
    entries = entries_for(pool(1)) + entries
    for name, records in (("gen", [item(i) for i in range(25)]),
                          ("tr", [translated(i) for i in range(25)])):
        with pytest.raises(PoolError, match=match):
            load_italian(italian_batch(tmp_path / name, records), entries=entries)


def test_text_from_a_model_outside_the_teacher_registry_is_refused(tmp_path):
    entries = entries_for(pool(1)) + [TEACHER]
    llama = [{**item(i), "provenance": {"kind": "generated",
                                        "generator": {"model_id": "meta-llama/Llama-3.1-70B"}}}
             for i in range(25)]
    with pytest.raises(PoolError, match="generator_not_teacher"):
        load_italian(italian_batch(tmp_path / "g", llama), entries=entries)
    with pytest.raises(PoolError, match="generator_not_teacher"):
        load_italian(italian_batch(tmp_path / "t", [translated(i, "org/other") for i in range(25)]),
                     entries=entries)


def test_translated_case_from_a_blocked_or_other_source_is_refused(tmp_path):
    entries = entries_for(pool(1)) + [TEACHER]
    records = [item(i) for i in range(25)] + [translated(1)]
    batch = italian_batch(tmp_path, records)
    blocked = [replace(e, generator_terms="INCERTO") if e.name == "harm" else e for e in entries]
    with pytest.raises(PoolError, match="cancello"):
        load_italian(batch, entries=blocked)
    wrong = italian_batch(tmp_path / "w", [item(i) for i in range(25)]
                          + [{**translated(1), "source": "org/inj"}])
    with pytest.raises(PoolError, match="fonte"):
        load_italian(wrong, entries=entries)


def test_build_writes_mix_and_a_manifest_that_pins_every_input(tmp_path, monkeypatch):
    cases = pool()
    ingest = write_ingest(tmp_path / "in", cases)
    monkeypatch.setattr(mix_cli, "load_ingest", partial(
        load_ingest, entries=entries_for(cases), manifest_sha256=DIGEST))
    outcomes = tmp_path / "outcomes.jsonl"
    index = judged(cases)
    outcomes.write_text("".join(json.dumps(r) + "\n" for by in index.values() for r in by.values()))
    targets = Targets(total=100, languages=EN_ONLY, method_cap=0.5)
    common = dict(targets=targets, seed=1, sha="c" * 40)
    with pytest.raises(MixError, match="override-reason"):
        mix_cli.build_mix(ingest, [], [outcomes], tmp_path / "mix", override_reason=None, **common)
    manifest = mix_cli.build_mix(ingest, [], [outcomes], tmp_path / "mix",
                                 override_reason="IT non ancora disponibile (#7)", **common)
    mix = tmp_path / "mix"
    assert manifest["output"]["sha256"] == hashlib.sha256((mix / "mix.jsonl").read_bytes()).hexdigest()
    assert manifest["ingest_report"]["sha256"] == manifest["inputs"]["ingest"]["report_sha256"]
    assert manifest["repo_sha"] == "c" * 40
    assert manifest["override_reason"] == "IT non ancora disponibile (#7)"
    assert manifest["inputs"]["outcomes"][0]["name"] == "outcomes.jsonl"
    assert json.loads((mix / "mix-manifest.json").read_text()) == manifest
    rows = [json.loads(line) for line in (mix / "mix.jsonl").read_text().splitlines()]
    assert len(rows) == manifest["output"]["rows"]
    assert {r["task"] for r in rows} == {"semantic", "difficulty"}
    assert manifest["difficulty_margin_threshold"] == pytest.approx(math.log(4))
    other = mix_cli.build_mix(ingest, [], [outcomes], tmp_path / "mix2", margin_threshold=0.2,
                              override_reason="IT non ancora disponibile (#7)", **common)
    assert other["difficulty_margin_threshold"] == 0.2


def test_mix_output_inside_the_repository_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        mix_cli.check_outside_repo(mix_cli.REPO_ROOT / "mix")


GPT, QWEN = CONFIG["model_id"], "Qwen/Qwen3.5-397B-A17B"
QWEN_CONFIG = {**CONFIG, "model_id": QWEN, "provider_model": "qwen3.5-397b",
               "api_key_env": "SCUDO_TEST_KEY_2"}
PRICY = {"output_usd_per_mtok": 60_000.0}  # una chiamata nel caso peggiore costa ~60 USD


class Judged:
    """Risposta finta di un teacher: VIOLATION con uso riportato."""

    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        content = json.dumps({"verdict": "VIOLATION", "reasoning": "Motivo."})
        return json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()


def judge_inputs(tmp_path, monkeypatch, tables=(CONFIG,), network=False):
    """Campione di un caso e TOML; senza `network` urlopen fallisce il test se viene chiamato."""
    monkeypatch.setenv("SCUDO_TEST_KEY", "k-not-real")
    monkeypatch.setenv("SCUDO_TEST_KEY_2", "k2-not-real")

    def urlopen(request, timeout):
        if not network:
            raise AssertionError("chiamata di rete in un test")
        return Judged()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    out = tmp_path / "judge"
    out.mkdir()
    (out / "sample.jsonl").write_text(json.dumps({
        "set": "volume", "key": "harm:1:prompt", "criterion": "c", "evidence": "e"}) + "\n")
    teachers = tmp_path / "teachers.toml"
    teachers.write_text(toml(list(tables)), encoding="utf-8")
    return out, teachers, tmp_path / "ledger.jsonl"


def judge_confirm(out, teachers, ledger):
    return pilot.estimate(out, load_configs(teachers), Ledger(ledger))["confirm"]


def confirm(ledger, teachers, amount):
    confirm_volume(Ledger(ledger), teachers=teachers, amount_usd=amount, by="persona",
                   rationale="scelta dal report del pilot")


def test_two_teacher_judging_within_the_confirmed_amount_runs(tmp_path, monkeypatch):
    out, teachers, ledger = judge_inputs(tmp_path, monkeypatch, (CONFIG, QWEN_CONFIG),
                                         network=True)
    confirm(ledger, [GPT, QWEN], 10)
    result = mix_cli.judge(out, teachers, ledger, confirm=judge_confirm(out, teachers, ledger))
    assert result["done"] == 2 and result["stopped"] is None
    assert {o["teacher"] for o in pilot.read_outcomes(out)} == {GPT, QWEN}


def test_judging_refuses_a_teacher_outside_the_latest_confirmed_set(tmp_path, monkeypatch):
    out, teachers, ledger = judge_inputs(tmp_path, monkeypatch, (CONFIG, QWEN_CONFIG))
    confirm(ledger, [GPT, QWEN], 10)
    confirm(ledger, [QWEN], 10)  # vale solo l'ultima conferma
    with pytest.raises(BudgetError, match="senza conferma del volume"):
        mix_cli.judge(out, teachers, ledger, confirm=judge_confirm(out, teachers, ledger))
    assert not (out / "outcomes.jsonl").exists()


def test_judging_refuses_a_teacher_sum_over_the_confirmed_amount(tmp_path, monkeypatch):
    out, teachers, ledger = judge_inputs(
        tmp_path, monkeypatch, ({**CONFIG, **PRICY}, {**QWEN_CONFIG, **PRICY}))
    confirm(ledger, [GPT, QWEN], 100)
    total = judge_confirm(out, teachers, ledger)
    assert 100 < float(total) < 150  # ogni teacher da solo starebbe nei 100 USD
    with pytest.raises(BudgetError, match="volume confermato"):
        mix_cli.judge(out, teachers, ledger, confirm=total)
    assert not Ledger(ledger).of_kind("authorisation")


def test_volume_judging_needs_the_human_confirmation_of_each_teacher(tmp_path, monkeypatch):
    out, teachers, ledger = judge_inputs(tmp_path, monkeypatch)
    with pytest.raises(BudgetError, match="conferma umana"):
        mix_cli.judge(out, teachers, ledger, confirm="0.01")
    confirm(ledger, [QWEN], 10)
    with pytest.raises(BudgetError, match="senza conferma del volume"):
        mix_cli.judge(out, teachers, ledger, confirm="0.01")
    assert not (out / "outcomes.jsonl").exists()


def test_dirty_work_tree_is_refused(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(MixError, match="modifiche"):
        mix_cli.repo_sha(tmp_path)
