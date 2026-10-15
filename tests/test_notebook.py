import ast
import json
from pathlib import Path

from scudo import nemotron
from scudo.notebook import NOTEBOOK_PATH, build

ROOT = Path(__file__).resolve().parents[1]


def code_cells(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            yield "".join(cell["source"])


def python_only(source: str) -> str:
    # Le righe shell di IPython (`!`, `%`) non sono Python.
    return "\n".join("" if line.lstrip().startswith(("!", "%")) else line
                     for line in source.splitlines())


def test_committed_notebook_is_generated_from_repository():
    committed = json.loads((ROOT / NOTEBOOK_PATH).read_text())
    assert committed == build(), "rigenera con: uv run python -m scudo.notebook"


def test_notebook_is_valid_nbformat_4_with_parsable_code():
    notebook = build()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for source in code_cells(notebook):
        ast.parse(python_only(source))


def test_notebook_pins_inputs_and_resumes_from_hub_checkpoints():
    text = "\n".join(code_cells(build()))
    assert nemotron.REVISION in text
    assert 'hub_strategy="checkpoint"' in text
    assert "hub_private_repo=True" in text
    assert "resume_from_checkpoint=resume" in text
    assert "last-checkpoint" in text
    assert "run_config.json" in text  # la ripresa verifica la configurazione del run
    assert "RUN_FINGERPRINT" in text
    assert "except Exception" in text  # un readout fallito non interrompe l'eval
    assert "load_verified" in text  # il GGUF esportato si rilegge solo dopo il digest
    # Il report dichiara superficie, pesi con digest, SHA di scudo e runtime.
    assert 'row["surface"], row["verdict"]' in text
    assert "weights=[*base_weights, adapter_identity]" in text
    assert 'scudo_sha=RUN_LOG["scudo_sha"]' in text
    assert "runtime=" in text


def test_notebook_never_embeds_tokens():
    serialized = json.dumps(build())
    assert "hf_" not in serialized.replace("hf_hub", "").replace("hf_token", "")
    assert 'userdata.get("HF_TOKEN")' in "\n".join(code_cells(build()))
