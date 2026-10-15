import ast
import json
import re
from pathlib import Path

from scudo import notebook as tracer
from scudo import notebook_cells as shared
from scudo.eval_notebook import FAST_BASE_REVISION
from scudo.fast_notebook import NOTEBOOK_PATH, RUN_NAME, build
from test_notebook import code_cells, python_only

ROOT = Path(__file__).resolve().parents[1]


def text():
    return "\n".join(code_cells(build()))


def test_committed_notebook_is_generated_from_repository():
    committed = json.loads((ROOT / NOTEBOOK_PATH).read_text())
    assert committed == build(), "rigenera con: uv run python -m scudo.fast_notebook"


def test_code_cells_parse_and_no_placeholder_is_left():
    notebook = build()
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU"
    for source in code_cells(notebook):
        ast.parse(python_only(source))
    assert not re.search(r"@[A-Z_]+@", json.dumps(notebook))


def test_shared_cells_come_from_the_shared_module_not_copies():
    sources = list(code_cells(build()))
    assert shared.SETUP_HUB_CACHE in sources
    assert shared.LORA in sources and shared.TRAIN in sources
    assert shared.named(shared.GGUF, RUN_NAME) in sources
    assert shared.named(shared.ADAPTER, RUN_NAME) in text()
    assert FAST_BASE_REVISION in tracer.CONFIG and FAST_BASE_REVISION in text()


def test_mix_is_read_only_after_its_digest_and_pinned_in_the_fingerprint():
    source = text()
    assert '"mix_manifest_sha256": ""' in source and '"mix_revision": ""' in source
    assert "raise RuntimeError" in source  # senza revisione e digest il run non parte
    assert "load_verified(path, identity" in source
    assert 'verified_text(output["filename"], output["sha256"])' in source
    assert "verify_dedup" in source
    assert "RUN_FINGERPRINT" in source  # RUN contiene il digest del mix


def test_eval_uses_the_harness_and_reports_the_difficulty_routing():
    source = text()
    for name in ("load_cases", "sample(full", "fast_scorer", "reports_for", "routing_metrics",
                 "as_difficulty", 'report["difficulty_routing"]', "hard_recall_on_wrong"):
        assert name in source
    assert 'RUN_LOG["mix"]' in source and 'RUN_LOG["scudo_sha"]' in source
    assert 'RUN_LOG["gguf"]' in source and 'RUN_LOG["adapter"]' in source


def test_notebook_never_embeds_tokens():
    serialized = json.dumps(build())
    assert "hf_" not in serialized.replace("hf_hub", "").replace("hf_token", "").replace(
        "hf-cache", "")
