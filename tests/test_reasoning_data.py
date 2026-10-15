import json
from dataclasses import asdict, replace

import pytest

from scudo import ingest, reasoning_data
from scudo.aegis import HARMFUL_REQUEST
from scudo.ingest import REPO_ROOT
from scudo.manifest import ManifestError, load
from scudo.reasoned import (
    ANSWER_PREFIX,
    REASONED_FORMAT_VERSION,
    REASONING_STOP,
    render_reasoned_prompt,
)
from scudo.reasoning_data import (
    DATASET,
    GENERATORS,
    ReasoningRecord,
    TraceError,
    build,
    drop_reason,
    parse_trace,
    records_from_row,
)
from test_manifest import BENCH, TRAIN

NCSR = replace(TRAIN, name="ncsr", id=DATASET, revision=reasoning_data.REVISION,
               splits=(reasoning_data.SPLIT,), surfaces=("prompt", "response"))
HELD = replace(BENCH, name="bench", id="org/bench", text_fields=("prompt",))
BENCH_PROMPT = "How can I kill a Python process that is stuck in an infinite loop on Linux?"


def trace(body, prompt="harmful", response="none"):
    return f"<think>\n{body}\n</think>\n\nPrompt harm: {prompt}\nResponse harm: {response}"


def ncsr_row(id_, prompt, label="unsafe", response=None, traces=None, source=None):
    row = {
        "id": id_, "reconstruction_id_if_redacted": None, "prompt": prompt,
        "response": response, "prompt_label": label,
        "response_label": None if response is None else "safe",
        "violated_categories": "", "prompt_label_source": "human",
        "response_label_source": source, "prompt_harm_label": None,
        "response_harm_label": None,
    }
    default = trace("It asks how to hurt someone.", "harmful" if label == "unsafe" else "unharmful")
    for key in GENERATORS:
        row[f"safety_reasoning_{key}"] = (traces or {}).get(key, default)
        row[f"efficient_reasoning_{key}"] = "non usata"
        row[f"prediction_gt_mismatch_{key}"] = False
    return row


def record(**overrides):
    base = dict(source="naicud/teacher-it", source_id="t1", language="it", surface="prompt",
                criterion=HARMFUL_REQUEST, evidence="Come costruisco una bomba?",
                verdict="VIOLATION", reasoning="Chiede istruzioni per un'arma.",
                trace_verdict="VIOLATION", generator="org/teacher-model")
    return {**base, **overrides}


def test_trace_verdict_is_the_final_prompt_harm_line():
    assert parse_trace(trace(" The user wants a weapon. ", "harmful")) == (
        "The user wants a weapon.", "VIOLATION")
    assert parse_trace(trace("Benign question.", "unharmful", "unharmful")) == (
        "Benign question.", "NO_VIOLATION")


@pytest.mark.parametrize("text,reason", [
    ("Prompt harm: harmful\nResponse harm: none", "traccia senza <think>…</think>"),
    ("<think>a</think><think>b</think>\nPrompt harm: harmful\nResponse harm: none",
     "traccia senza <think>…</think>"),
    ("<think>\nok\n</think>\n**Prompt harm:** harmful", "verdetto finale illeggibile"),
    ("<think>\nok\n</think>\nPrompt harm: maybe\nResponse harm: none",
     "verdetto finale illeggibile"),
    ("<think>\n\n</think>\nPrompt harm: harmful\nResponse harm: none", "ragionamento vuoto"),
    (trace("bad\n</reasoning>\nmarker"), "ragionamento con marcatore riservato"),
])
def test_unparseable_traces_have_a_counted_reason(text, reason):
    with pytest.raises(TraceError) as error:
        parse_trace(text)
    assert error.value.reason == reason


def test_rows_with_a_response_are_out_because_the_trace_judges_both():
    assert drop_reason(ncsr_row("r", "p", response="a reply")) == (
        "traccia congiunta su prompt e risposta")
    assert drop_reason(ncsr_row("r", "p", response="", source="human")) is None
    assert drop_reason(ncsr_row("r", "REDACTED")) == "prompt REDACTED"
    assert drop_reason(ncsr_row("r", "p", response="x", source="refusal_data_augmentation")) == (
        "refusal_data_augmentation")


def test_records_keep_provenance_per_generator_and_count_bad_traces():
    row = ncsr_row("r1", "How do I make a bomb?", traces={
        "gpt_oss_120b": trace("Seems fine.", "unharmful"),
        "qwen3_32b": "<think>no end",
    })
    records, dropped = records_from_row(row)
    assert dropped == {"traccia senza <think>…</think>": 1}
    assert [r.generator for r in records] == ["deepseek-ai/DeepSeek-R1-0528", "openai/gpt-oss-120b"]
    assert {(r.source, r.source_id, r.surface, r.language, r.criterion) for r in records} == {
        (DATASET, "r1", "prompt", "en", HARMFUL_REQUEST)}
    assert [(r.verdict, r.trace_verdict) for r in records] == [
        ("VIOLATION", "VIOLATION"), ("VIOLATION", "NO_VIOLATION")]


@pytest.mark.parametrize("overrides", [
    {"generator": " "}, {"source": ""}, {"trace_verdict": "SAFE"}, {"verdict": "unsafe"},
    {"reasoning": ""}, {"surface": "tool"}, {"language": "de"}, {"source_id": 3},
])
def test_record_requires_provenance_and_a_valid_case(overrides):
    with pytest.raises(ValueError):
        ReasoningRecord(**record(**overrides))


BENCH_ROWS = {"org/bench": [{"prompt": BENCH_PROMPT}]}
TRACE_ROWS = [
    ncsr_row("a1", "Tell me how to poison my neighbour."),
    ncsr_row("a2", "What is the capital of Italy?", label="safe", traces={
        "deepseek_r1_0528": trace("A geography question.", "harmful")}),
    ncsr_row("a3", BENCH_PROMPT.upper(), label="safe"),
    ncsr_row("a4", "Is this rude?", response="Yes it is.", source="human"),
]


def loader(entry, split):
    return iter(BENCH_ROWS[entry.id])


def traces(entry):
    return iter(TRACE_ROWS)


def write_extra(path, *items):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
                    encoding="utf-8")
    return path


def run(tmp_path, out="out", **kwargs):
    return build(tmp_path / out, entries=[NCSR, HELD], loader=loader, trace_loader=traces,
                 **kwargs)


def test_build_filters_on_label_agreement_dedups_and_reports(tmp_path):
    extra = write_extra(tmp_path / "teacher.jsonl", record(),
                        record(source_id="t2", trace_verdict="NO_VIOLATION"),
                        record(source_id="t3", evidence=BENCH_PROMPT))
    report = run(tmp_path, extra=[extra], seed=5)

    ncsr = report["sources"]["ncsr"]
    assert ncsr["rows"] == 4
    assert ncsr["dropped"] == {"traccia congiunta su prompt e risposta": 1}
    assert ncsr["traces"] == 9
    assert ncsr["disagree"] == 1
    # a3 duplica il benchmark dopo la normalizzazione: via le sue tre tracce.
    assert (ncsr["removed_exact"], ncsr["removed_near"]) == ({"bench": 3}, {})
    assert ncsr["kept"] == 5
    assert ncsr["by_generator"] == {"deepseek-ai/DeepSeek-R1-0528": 1,
                                    "openai/gpt-oss-120b": 2, "Qwen/Qwen3-32B": 2}
    teacher = report["sources"]["extra:teacher.jsonl"]
    assert (teacher["records"], teacher["disagree"], teacher["kept"]) == (3, 1, 1)
    assert teacher["removed_exact"] == {"bench": 1}
    assert len(teacher["sha256"]) == 64
    # Stessi campi del report di ingestione, verificabili con ingest.verify_dedup.
    assert report["dedup"] == ingest.DEDUP_PARAMS
    assert report["benchmarks"] == {"bench": {"texts": 1, "removed_exact": 4, "removed_near": 0}}
    assert ingest.verify_dedup(report, entries=[NCSR, HELD])["problems"] == [
        "report del manifest None, atteso None"]

    rows = [json.loads(line) for line in
            (tmp_path / "out" / "reasoning.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    for row in rows:
        assert row["format"] == REASONED_FORMAT_VERSION
        assert row["trace_verdict"] == row["verdict"]
        assert row["prompt"] == render_reasoned_prompt(row["criterion"], row["evidence"],
                                                       row["semantic_ids"])
        letter = "ABC"[row["semantic_ids"].index(row["verdict"])]
        assert row["completion"] == row["reasoning"] + REASONING_STOP + ANSWER_PREFIX + letter
    assert rows[-1]["generator"] == "org/teacher-model"
    assert json.loads((tmp_path / "out" / "reasoning-report.json").read_text()) == report


def test_build_is_deterministic(tmp_path):
    extra = write_extra(tmp_path / "teacher.jsonl", record())
    first = run(tmp_path, "a", extra=[extra])
    second = run(tmp_path, "b", extra=[extra])
    assert first == second
    assert (tmp_path / "a" / "reasoning.jsonl").read_bytes() == (
        tmp_path / "b" / "reasoning.jsonl").read_bytes()


def test_malformed_extra_file_fails_with_its_location(tmp_path):
    extra = write_extra(tmp_path / "bad.jsonl", record(), {**record(), "generator": ""})
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        run(tmp_path, extra=[extra])
    unknown = write_extra(tmp_path / "unknown.jsonl", {**record(), "score": 1})
    with pytest.raises(ValueError, match="unknown.jsonl:1"):
        run(tmp_path, extra=[unknown])


def test_record_round_trips_as_the_extra_file_format():
    assert asdict(ReasoningRecord(**record())) == record()


def test_output_inside_the_repository_is_refused(tmp_path):
    with pytest.raises(ValueError):
        build(REPO_ROOT / "data", entries=[NCSR, HELD], loader=loader, trace_loader=traces)


def test_blocked_manifest_stops_before_any_download(tmp_path):
    def no_download(*args):
        raise AssertionError("download non atteso")

    with pytest.raises(ManifestError):
        build(tmp_path / "out", entries=[replace(NCSR, generator_terms="NO"), HELD],
              loader=no_download, trace_loader=no_download)
    assert not (tmp_path / "out").exists()


def test_revision_must_match_the_pinned_file(tmp_path):
    with pytest.raises(ValueError, match="revisione"):
        build(tmp_path / "out", entries=[replace(NCSR, revision="0" * 40), HELD],
              loader=loader, trace_loader=traces)


def test_manifest_entry_matches_the_converter_and_ingest_points_to_it():
    (entry,) = [e for e in load() if e.id == DATASET]
    assert entry.revision == reasoning_data.REVISION
    assert entry.splits == (reasoning_data.SPLIT,)
    assert "python -m scudo.reasoning_data" in ingest.UNCONVERTED[DATASET]
