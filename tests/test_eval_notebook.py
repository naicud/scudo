import ast
import json
from pathlib import Path

from scudo import notebook as tracer
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
