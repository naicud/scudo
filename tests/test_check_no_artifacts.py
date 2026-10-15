import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_artifacts.py"


def run_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("Scudo\n")
    return tmp_path


def stage(root: Path, *names: str) -> None:
    subprocess.run(["git", "add", "-f", *names], cwd=root, check=True)


def test_clean_tree_passes(repo: Path):
    stage(repo, "README.md")
    result = run_check(repo)
    assert result.returncode == 0, result.stdout


def test_weight_file_is_rejected(repo: Path):
    (repo / "model.gguf").write_bytes(b"GGUF")
    stage(repo, "README.md", "model.gguf")
    result = run_check(repo)
    assert result.returncode == 1
    assert "model.gguf" in result.stdout


def test_token_is_rejected(repo: Path):
    fake_token = "hf_" + "A1b2C3d4E5" * 4
    (repo / "notes.txt").write_text(f"token={fake_token}\n")
    stage(repo, "notes.txt")
    result = run_check(repo)
    assert result.returncode == 1
    assert "Hugging Face" in result.stdout


def test_large_file_is_rejected(repo: Path):
    (repo / "big.txt").write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    stage(repo, "big.txt")
    result = run_check(repo)
    assert result.returncode == 1
    assert "big.txt" in result.stdout


def test_fixture_jsonl_is_allowed(repo: Path):
    fixtures = repo / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "cases.jsonl").write_text('{"verdict": "NO_VIOLATION"}\n')
    stage(repo, "tests/fixtures/cases.jsonl")
    result = run_check(repo)
    assert result.returncode == 0, result.stdout
