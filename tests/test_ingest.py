import json
from dataclasses import replace
from pathlib import Path

import pytest

from scudo import aegis
from scudo.ingest import REPO_ROOT, build, check_outside_repo
from scudo.manifest import ManifestError
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


def test_build_is_deterministic(tmp_path):
    first = build(tmp_path / "a", entries=[AEGIS, HELD], loader=loader)
    second = build(tmp_path / "b", entries=[AEGIS, HELD], loader=loader)
    assert first == second
    assert Path(tmp_path / "a" / "aegis2-train.jsonl").read_bytes() == Path(
        tmp_path / "b" / "aegis2-train.jsonl").read_bytes()
