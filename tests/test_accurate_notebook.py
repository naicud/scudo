import ast
import json
import re
from pathlib import Path

from scudo import notebook_cells as shared
from scudo import reasoning_data
from scudo.accurate_notebook import BASE_REVISION, NOTEBOOK_PATH, build
from test_notebook import code_cells, python_only

ROOT = Path(__file__).resolve().parents[1]


def text():
    return "\n".join(code_cells(build()))


def test_committed_notebook_is_generated_from_repository():
    committed = json.loads((ROOT / NOTEBOOK_PATH).read_text())
    assert committed == build(), "rigenera con: uv run python -m scudo.accurate_notebook"


def test_code_cells_parse_and_no_placeholder_is_left():
    notebook = build()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for source in code_cells(notebook):
        ast.parse(python_only(source))
    assert not re.search(r"@[A-Z_]+@", json.dumps(notebook))


def test_shared_cells_are_the_shared_module_cells_not_copies():
    sources = list(code_cells(build()))
    assert shared.SETUP in sources and shared.TRAIN in sources
    assert shared.named(shared.GGUF, "dell'Accurate") in sources
    assert shared.named(shared.ADAPTER, "dell'Accurate") in "\n".join(sources)


def test_notebook_pins_base_and_data_and_verifies_weights_before_loading():
    source = text()
    assert re.fullmatch(r"[0-9a-f]{40}", BASE_REVISION)
    assert f'"base_revision": "{BASE_REVISION}"' in source
    assert '"base_license": "apache-2.0"' in source
    assert f'"dataset_revision": "{reasoning_data.REVISION}"' in source
    assert source.index("fetch_verified(") < source.index("FastLanguageModel.from_pretrained(")
    assert "verify_dedup(data_report)" in source


def test_training_is_bf16_lora_with_completion_only_loss_and_guarded_resume():
    source = text()
    assert '"load_in_4bit": False' in source  # Unsloth sconsiglia QLoRA su Qwen3.5
    assert "completion_only_loss=True" in source
    assert "resume_from_checkpoint=resume" in source
    assert "run_config.json" in source and "RUN_FINGERPRINT" in source
    assert 'RUN["data_sha256"]' in source  # i dati entrano nel fingerprint di ripresa
    for module in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "o_proj"):
        assert f'"{module}"' in source


def test_eval_reuses_the_harness_with_the_reasoned_step():
    source = text()
    assert "accurate_scorer(tok, TransformersBackend(model)" in source
    assert "load_cases(entry, loader)" in source
    assert "run(loaded, score)" in source
    assert "accurate_reports(" in source and "format=REASONED_FORMAT_VERSION" in source
    assert 'proof_type="local"' in source
    assert "load_verified" in source


def test_unverified_steps_are_marked_and_tokens_never_embedded():
    assert "NON VERIFICATO" in json.dumps(build())
    serialized = json.dumps(build())
    assert "hf_" not in serialized.replace("hf_hub", "").replace("hf_token", "")
    assert 'userdata.get("HF_TOKEN")' in text()
