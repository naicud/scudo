import json
from dataclasses import replace
from pathlib import Path

import pytest

from scudo import aegis, canttalk, procedural
from scudo.ingest import (
    CONVERTERS,
    DEDUP_PARAMS,
    REPO_ROOT,
    UNCONVERTED,
    build,
    check_outside_repo,
    main,
    manifest_digest,
    verify_dedup,
)
from scudo.manifest import ManifestError, benchmarks
from test_manifest import BENCH, TRAIN

BENCH_PROMPT = "How can I kill a Python process that is stuck in an infinite loop on Linux?"
AEGIS = replace(TRAIN, name="aegis2-train", id=aegis.DATASET, surfaces=("prompt", "response"))
OTHER = replace(TRAIN, name="other-train", id="org/other")
HELD = replace(BENCH, name="bench", id="org/bench", text_fields=("prompt", "response"))


def aegis_row(id_, prompt, response=None, response_label=None, source=None):
    return {
        "id": id_, "reconstruction_id_if_redacted": None, "prompt": prompt,
        "response": response, "prompt_label": "safe", "response_label": response_label,
        "violated_categories": "", "prompt_label_source": "human",
        "response_label_source": source,
    }


ROWS = {
    "org/bench": [{"prompt": BENCH_PROMPT, "response": None},
                  {"prompt": "Tell me a joke.", "response": "Why did the chicken cross?"}],
    aegis.DATASET: [
        aegis_row("r1", "What is the capital of Italy?", "Rome.", "safe", "human"),
        aegis_row("r2", "Hurt someone", "I can't help.", "safe", "refusal_data_augmentation"),
        aegis_row("r3", "REDACTED"),
        aegis_row("r4", BENCH_PROMPT.upper()),
        aegis_row("r5", "How can I kill a Python process that's stuck in an infinite loop on linux"),
        aegis_row("r6", "Write a poem about rain.", "Sure, here it is.", "unsafe", "llm_jury"),
    ],
}


def loader(entry, split):
    return iter(ROWS[entry.id])


def test_output_inside_the_repository_is_refused():
    with pytest.raises(ValueError):
        check_outside_repo(REPO_ROOT / "data")
    with pytest.raises(ValueError):
        check_outside_repo(REPO_ROOT)


def test_output_inside_any_git_work_tree_is_refused(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError):
        check_outside_repo(tmp_path / "out")
    assert check_outside_repo(tmp_path.parent / "fuori") == (tmp_path.parent / "fuori").resolve()


def test_blocked_manifest_stops_the_build_before_any_download(tmp_path):
    def no_download(entry, split):
        raise AssertionError("download non atteso")

    with pytest.raises(ManifestError):
        build(tmp_path / "out", entries=[replace(AEGIS, generator_terms="INCERTO"), HELD],
              loader=no_download)
    assert not (tmp_path / "out").exists()


def test_build_filters_marks_dedups_and_reports(tmp_path):
    out = tmp_path / "out"
    report = build(out, entries=[AEGIS, OTHER, HELD], loader=loader)

    source = report["sources"]["aegis2-train"]
    assert source["rows"] == 6
    assert source["dropped"] == {"refusal_data_augmentation": 1, "prompt REDACTED": 1}
    assert source["cases"] == 6  # r1 e r6 danno prompt+response, r4 e r5 solo prompt
    assert source["removed_exact"] == {"bench": 1}
    assert source["removed_near"] == {"bench": 1}
    assert source["kept"] == 4
    assert source["needs_relabel"] == 1
    assert report["benchmarks"]["bench"] == {"texts": 3, "removed_exact": 1, "removed_near": 1}
    assert report["not_ingested"] == {"other-train": "nessun convertitore"}

    lines = (out / "aegis2-train.jsonl").read_text().splitlines()
    kept = [json.loads(line) for line in lines]
    assert [row["source_id"] for row in kept] == ["r1", "r1", "r6", "r6"]
    assert [row["needs_relabel"] for row in kept] == [False, False, False, True]
    assert {row["split"] for row in kept} == {"train"}
    assert json.loads((out / "report.json").read_text()) == report
    assert (out / "data-cards" / "aegis2-train.md").exists()
    assert "Demo" in (out / "data-cards" / "ATTRIBUTIONS.md").read_text()


def test_registry_exposes_the_ingest_contract_for_every_converter():
    for dataset, converter in CONVERTERS.items():
        assert converter.DATASET == dataset
        assert callable(converter.needs_relabel)
        assert callable(getattr(converter, "convert_row", None)) or (
            callable(converter.drop_reason) and callable(converter.cases_from_row))
    assert not set(UNCONVERTED) & set(CONVERTERS)


PTD = replace(TRAIN, name="ptd-train", id=procedural.DATASET, config="all")
PTD_ROW = {
    "id": "graph_navigation:train:7", "task": "graph_navigation", "level": 0,
    "state": '{"nodes": ["N1", "N2"]}',
    "questions": json.dumps({
        "reachable": {"type": "noul", "instructions": "Does any route exist?"},
        "shortest_cost": {"type": "score", "instructions": "Cost?", "criteria": ["0", "1"]}}),
    "answers": json.dumps({
        "reachable": {"type": "noul", "noul": 1.0},
        "shortest_cost": {"type": "score", "score": 1.0}}),
}


def test_partial_drops_inside_a_row_are_counted_by_reason(tmp_path):
    def rows(entry, split):
        return iter([PTD_ROW] if entry.id == procedural.DATASET else ROWS[entry.id])

    report = build(tmp_path / "out", entries=[PTD, HELD], loader=rows)
    source = report["sources"]["ptd-train"]
    assert (source["rows"], source["cases"], source["kept"]) == (1, 1, 1)
    assert source["dropped"] == {"score_not_tristate": 1}


CTAT_ROW = {
    "domain": "real estate", "scenario": "Downsizing.",
    "system_instruction": "Help the user downsize. Do not discuss financing.",
    "conversation": [{"role": "user", "content": "Hi, I'm considering downsizing my home."},
                     {"role": "assistant", "content": "Downsizing can lower costs."}],
    "distractors": [], "conversation_with_distractors": [],
}
CTAT_TRAIN = replace(TRAIN, name="ctat-train", id=canttalk.DATASET, config="mixtral")
CTAT_TEST = replace(BENCH, name="ctat-test", id=canttalk.DATASET, config="mixtral",
                    text_fields=("system_instruction",))


def test_benchmark_with_a_converter_is_indexed_in_the_evidence_form(tmp_path):
    # Il dialogo è una lista: senza il convertitore l'indice vedrebbe solo la policy.
    report = build(tmp_path / "out", entries=[CTAT_TRAIN, CTAT_TEST],
                   loader=lambda entry, split: iter([CTAT_ROW]))
    assert report["benchmarks"]["ctat-test"]["texts"] == 2  # policy + un caso
    assert report["sources"]["ctat-train"]["removed_exact"] == {"ctat-test": 1}


def test_sources_without_a_faithful_mapping_report_why(tmp_path):
    civil = replace(TRAIN, name="civil-comments", id="google/civil_comments")
    report = build(tmp_path / "out", entries=[civil, HELD], loader=loader)
    assert report["not_ingested"]["civil-comments"] == UNCONVERTED["google/civil_comments"]


def test_build_is_deterministic(tmp_path):
    first = build(tmp_path / "a", entries=[AEGIS, HELD], loader=loader)
    second = build(tmp_path / "b", entries=[AEGIS, HELD], loader=loader)
    assert first == second
    assert Path(tmp_path / "a" / "aegis2-train.jsonl").read_bytes() == Path(
        tmp_path / "b" / "aegis2-train.jsonl").read_bytes()


def test_attributions_list_only_sources_that_produced_data(tmp_path):
    from scudo.ingest import write_cards

    cc_by = replace(TRAIN, name="cc-by-train", license="cc-by-4.0")
    write_cards(tmp_path, [cc_by, HELD], {})
    assert cc_by.title not in (tmp_path / "data-cards" / "ATTRIBUTIONS.md").read_text()
    write_cards(tmp_path, [cc_by, HELD])  # solo manifest: tutte le fonti di training
    assert cc_by.title in (tmp_path / "data-cards" / "ATTRIBUTIONS.md").read_text()


def test_verify_passes_only_when_every_registry_benchmark_was_indexed(tmp_path):
    report = build(tmp_path / "out", entries=[AEGIS, HELD], loader=loader)
    result = verify_dedup(report, entries=[AEGIS, HELD], manifest_sha256="a" * 64)
    assert result["status"] == "FAIL"  # report di un manifest senza digest
    report["manifest_sha256"] = "a" * 64
    result = verify_dedup(report, entries=[AEGIS, HELD], manifest_sha256="a" * 64)
    assert result == {"status": "PASS", "manifest_sha256": "a" * 64,
                      "report_manifest_sha256": "a" * 64, "indexed": {"bench": 3},
                      "problems": []}


def test_verify_fails_when_a_benchmark_is_missing_or_empty_in_the_report(tmp_path):
    report = build(tmp_path / "out", entries=[AEGIS, HELD], loader=loader)
    report["manifest_sha256"] = "a" * 64
    other = replace(HELD, name="nemotron3-test")
    result = verify_dedup(report, entries=[AEGIS, HELD, other], manifest_sha256="a" * 64)
    assert result["status"] == "FAIL"
    assert result["problems"] == ["nemotron3-test: non indicizzato dalla deduplica"]
    report["benchmarks"]["bench"]["texts"] = 0
    result = verify_dedup(report, entries=[AEGIS, HELD], manifest_sha256="a" * 64)
    assert result["problems"] == ["bench: non indicizzato dalla deduplica"]


def test_verify_fails_on_other_manifest_or_other_dedup_parameters(tmp_path):
    report = build(tmp_path / "out", entries=[AEGIS, HELD], loader=loader)
    report["manifest_sha256"] = "b" * 64
    assert verify_dedup(report, entries=[AEGIS, HELD], manifest_sha256="a" * 64)["problems"]
    report["manifest_sha256"] = "a" * 64
    report["dedup"]["threshold"] = 0.5
    [problem] = verify_dedup(report, entries=[AEGIS, HELD], manifest_sha256="a" * 64)["problems"]
    assert "parametri" in problem


def test_shipped_registry_requires_wildguardtest_in_the_dedup_report():
    # Oggi il cancello blocca la build (WildGuardTest senza campi verificati): nessun
    # report passa la verifica finché WildGuardTest non è indicizzato.
    indexed = {b.name: {"texts": 10} for b in benchmarks() if b.name != "wildguardtest"}
    report = {"manifest_sha256": manifest_digest(), "dedup": dict(DEDUP_PARAMS),
              "benchmarks": indexed}
    result = verify_dedup(report)
    assert result["status"] == "FAIL"
    assert result["problems"] == ["wildguardtest: non indicizzato dalla deduplica"]


def test_verify_cli_exits_one_on_fail(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"manifest_sha256": None, "dedup": {}, "benchmarks": {}}))
    assert main(["verify", "--report", str(path)]) == 1
    assert '"status": "FAIL"' in capsys.readouterr().out
