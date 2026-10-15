import subprocess
import sys

import pytest

HEAVY_MODULES = ("torch", "unsloth", "transformers", "datasets", "huggingface_hub", "peft")


@pytest.mark.parametrize("module", ["scudo", "scudo.ingest", "scudo.pilot", "scudo.pilot_report",
                                    "scudo.teacher_client", "scudo.benchmarks", "scudo.runner",
                                    "scudo.eval_notebook", "scudo.reasoned", "scudo.reasoning_data",
                                    "scudo.accurate_notebook", "scudo.reasoned_runner",
                                    "scudo.italian", "scudo.difficulty", "scudo.mix",
                                    "scudo.mix_cli", "scudo.fast_notebook", "scudo.notebook_cells"])
def test_import_does_not_load_training_stack(module):
    probe = (
        f"import sys, {module}\n"
        f"loaded = [name for name in {HEAVY_MODULES!r} if name in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == ""


def test_version_is_exposed():
    import scudo

    assert scudo.__version__ == "0.0.0"
