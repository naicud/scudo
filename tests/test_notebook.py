import ast
import json
import re
from pathlib import Path

import pytest

from scudo import nemotron
from scudo.base_models import CANDIDATES
from scudo.notebook import NOTEBOOK_PATH, bakeoff_path, build

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


# --- bake-off dei modelli base


def bakeoff_text(slug):
    return "\n".join(code_cells(build(slug)))


def run_config(notebook):
    """RUN del notebook valutato senza Colab: il dizionario fra `RUN = ` e `SEED`."""
    source = next(c for c in code_cells(notebook) if "RUN = {" in c)
    body = source.split("RUN = ", 1)[1].split("\nSEED", 1)[0]
    body = body.replace("nemotron.DATASET", '"dataset"').replace("FORMAT_VERSION", '"format"')
    return ast.literal_eval(body)


@pytest.mark.parametrize("slug", list(CANDIDATES))
def test_committed_bakeoff_notebooks_are_generated_from_repository(slug):
    committed = json.loads((ROOT / bakeoff_path(slug)).read_text())
    assert committed == build(slug), f"rigenera con: uv run python -m scudo.notebook --base {slug}"


@pytest.mark.parametrize("slug", list(CANDIDATES))
def test_bakeoff_notebook_parses_and_pins_the_candidate(slug):
    notebook = build(slug)
    for source in code_cells(notebook):
        ast.parse(python_only(source))
    assert not re.search(r"@[A-Z_]+@", json.dumps(notebook))
    item = CANDIDATES[slug]
    run = run_config(notebook)
    assert (run["base_model"], run["base_revision"]) == (item.repo, item.revision)
    assert run["checkpoint_repo"] == item.checkpoint_repo
    assert run["lora_target_modules"] == list(item.lora_target_modules)
    assert run["lora_modules"] == item.lora_modules
    assert (run["loader"], run["text_only"]) == (item.loader, item.text_only)
    assert (run["bos_token"], run["bos_token_id"]) == (item.bos_token, item.bos_token_id)
    assert "Naicud/scudo-checkpoints" not in bakeoff_text(slug)


def test_bakeoff_keeps_the_tracer_recipe_and_changes_only_the_base_fields():
    tracer_run = run_config(build())
    per_base = {"base_model", "base_revision", "checkpoint_repo", "lora_target_modules",
                "lora_modules", "loader", "text_only", "bos_token", "bos_token_id",
                "ignore_files"}

    def recipe(run):
        return {k: v for k, v in run.items() if k not in per_base}

    runs = {}
    for slug in CANDIDATES:
        run = run_config(build(slug))
        assert recipe(run) == recipe(tracer_run)
        runs[slug] = json.dumps(run, sort_keys=True)
    assert len(set(runs.values())) == len(CANDIDATES)  # impronte del run diverse


def test_bakeoff_reads_lora_loader_and_bos_from_run_and_checks_them_early():
    text = bakeoff_text("gemma4-e4b")
    assert 'target_modules=RUN["lora_target_modules"]' in text
    assert 'Loader = getattr(unsloth, RUN["loader"])' in text
    assert 'text_only=RUN["text_only"]' in text
    assert "FastLanguageModel" not in text
    # Errori espliciti: torri caricate, BOS diverso, moduli LoRA attesi, id di training.
    assert "text_only ignorato" in text
    assert 'RUN["lora_modules"]' in text
    assert "BOS_IDS + tok.encode(" in text
    assert "trainer.train_dataset[0]" in text
    assert 'BOS_TEXT + row["prompt"]' in text
    assert 'harness="scudo.notebook/bakeoff_gemma4-e4b"' in text


def test_default_build_is_still_the_tracer():
    assert NOTEBOOK_PATH == "notebooks/fast_tracer.ipynb"
    assert bakeoff_path("qwen35-4b") == "notebooks/bakeoff_qwen35-4b.ipynb"
    with pytest.raises(KeyError):
        build("llama-3b")
