import ast
import json
from pathlib import Path

from scudo import notebook as tracer
from scudo.base_models import CANDIDATES
from scudo.baselines import BASELINES
from scudo.eval_notebook import FAST_BASE_REVISION, NOTEBOOK_PATH, build
from test_notebook import code_cells, python_only

ROOT = Path(__file__).resolve().parents[1]


def text():
    return "\n".join(code_cells(build()))


def test_committed_eval_notebook_is_generated_from_repository():
    committed = json.loads((ROOT / NOTEBOOK_PATH).read_text())
    assert committed == build(), "rigenera con: uv run python -m scudo.eval_notebook"


def test_eval_notebook_is_valid_nbformat_4_with_parsable_code():
    notebook = build()
    assert notebook["nbformat"] == 4
    for source in code_cells(notebook):
        ast.parse(python_only(source))


def test_one_run_per_baseline_and_our_fast_last():
    cells = [cell["id"] for cell in build()["cells"]]
    source = text()
    for name in BASELINES:
        assert f'evaluate_baseline("{name}")' in source
    assert cells.index("scudo-fast") > max(
        cells.index(f"scudo-{key}") for key in ("granite-policy", "guardian", "qwen3guard",
                                                "rizzo", "apus"))
    assert "import unsloth" not in "\n".join(
        "".join(cell["source"]) for cell in build()["cells"] if cell["id"] != "scudo-fast")


def test_reports_declare_proof_type_weights_runtime_and_cite_dedup():
    source = text()
    assert 'proof_type="reference"' in source  # baseline nel runtime transformers
    assert 'proof_type="local"' in source  # il nostro checkpoint in training
    assert "weights=weights" in source and "weights=[*base_weights, *adapter_weights]" in source
    assert 'runtime={"name": "transformers"' in source
    assert "dedup=DEDUP" in source and "verify_dedup" in source
    assert '"NOT_RUN"' in source  # verifica non configurata: dichiarata, mai PASS
    assert "upload_folder" in source and "summary.json" in source


def test_headline_prints_tri_state_and_binary_detection():
    source = text()
    assert '"accuracy": overall["accuracy"]' in source
    assert '"binary": overall["binary"]' in source
    assert '"unsupported": overall["unsupported"]' in source


def test_refusals_are_recorded_never_silent():
    source = text()
    assert 'SUMMARY["skipped_models"][name] = str(error)' in source
    assert "REFUSED[entry.name] = str(error)" in source
    assert "except Exception" not in source


def test_fast_uses_the_tracer_base_revision():
    assert FAST_BASE_REVISION in tracer.CONFIG
    assert FAST_BASE_REVISION in text()


def test_eval_notebook_never_embeds_tokens():
    serialized = json.dumps(build())
    assert "hf_" not in serialized.replace("hf_hub", "").replace("hf_token", "")
    assert 'userdata.get("HF_TOKEN")' in text()


def test_benchmarks_are_sampled_per_benchmark_with_a_seed():
    source = "\n".join(code_cells(build()))
    assert '"per_benchmark": 500' in source
    assert "sample(full, EVAL[\"per_benchmark\"], seed=EVAL[\"seed\"])" in source


def test_fast_points_at_the_tracer_adapter_upload():
    assert '"adapter_revision": "e7124a687ce3721a29bfb6b68a461357c0400659"' in "\n".join(
        code_cells(build()))


def fast_entries():
    source = next(c for c in code_cells(build()) if "EVAL = {" in c)
    body = source.split("EVAL = ", 1)[1].split("\nRUN_LOG", 1)[0]
    return ast.literal_eval(body)["fast"]


def test_fast_is_a_list_with_the_tracer_first_then_every_bakeoff_candidate():
    entries = fast_entries()
    assert entries[0]["name"] == "scudo-fast"
    assert entries[0]["base_revision"] == FAST_BASE_REVISION
    assert entries[0]["adapter_revision"] == "e7124a687ce3721a29bfb6b68a461357c0400659"
    assert (entries[0]["loader"], entries[0]["text_only"], entries[0]["bos_token_id"]) == (
        "FastLanguageModel", True, None)
    bakeoff = {e["name"]: e for e in entries[1:]}
    assert list(bakeoff) == [f"bakeoff-{slug}" for slug in CANDIDATES]
    for slug, item in CANDIDATES.items():
        entry = bakeoff[f"bakeoff-{slug}"]
        assert (entry["base_model"], entry["base_revision"]) == (item.repo, item.revision)
        assert entry["adapter_repo"] == item.checkpoint_repo
        assert entry["adapter_revision"] == ""  # commit dell'upload, da fissare dopo il training
        assert (entry["loader"], entry["text_only"]) == (item.loader, item.text_only)
        assert (entry["bos_token"], entry["bos_token_id"]) == (item.bos_token, item.bos_token_id)
    assert len({e["name"] for e in entries}) == len(entries)


def test_fast_candidates_share_the_runner_path_and_save_per_case_results():
    source = text()
    assert 'for FAST in EVAL["fast"]:' in source
    assert 'fast_scorer(tok, TransformersBackend(model), seed=EVAL["seed"],' in source
    assert 'bos_id=FAST["bos_token_id"]' in source
    assert 'Loader = getattr(unsloth, FAST["loader"])' in source
    assert 'text_only=FAST["text_only"]' in source
    assert 'SUMMARY["skipped_models"][FAST["name"]]' in source
    assert "result_rows(results)" in source and '"results.json"' in source
