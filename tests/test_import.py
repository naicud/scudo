import subprocess
import sys

HEAVY_MODULES = ("torch", "unsloth", "transformers")


def test_import_does_not_load_training_stack():
    probe = (
        "import sys, scudo\n"
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
