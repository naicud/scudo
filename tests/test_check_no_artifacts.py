import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_artifacts.py"


def run_check(root: Path, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    import os

    base = {k: v for k, v in os.environ.items() if k not in LOCAL_TOKEN_VARS}
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True,
        env={**base, "HOME": str(root / "no-home"), **(env or {})},
    )


LOCAL_TOKEN_VARS = ("HF_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "OPENAI_API_KEY", "SCUDO_NO_GH")


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


@pytest.mark.parametrize("label, secret", [
    ("Hugging Face OAuth", "hf_oauth_" + "Ab3dEf6hIj" * 4),
    ("GitHub", "github_pat_" + "11AB22CD33" * 6),
    ("GitHub", "ghs_" + "Ab3dEf6hIj" * 4),
    ("Bearer", "Authorization: Bearer " + "Ab3dEf6hIj" * 4),
    ("JWT", "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" + ".c2lnbmF0dXJlc2lnbmF0dXJl"),
    ("chiave privata", "-----BEGIN OPENSSH " + "PRIVATE KEY-----"),
    ("URL con credenziali", "https://x-access-token:" + "Ab3dEf6hIj" * 3 + "@github.com/o/r.git"),
    ("AWS", "AKIA" + "ABCDEFGHIJ123456"),
    ("Google API", "AIza" + "Ab3dEf6hIj" * 3 + "Ab3dE"),
])
def test_more_token_formats_are_rejected(repo: Path, label: str, secret: str):
    (repo / "notes.txt").write_text(f"x = {secret!r}\n")
    stage(repo, "notes.txt")
    result = run_check(repo)
    assert result.returncode == 1
    assert label in result.stdout


def test_template_placeholders_are_not_secrets(repo: Path):
    (repo / "cell.py").write_text(
        'url = f"https://x-access-token:{GH_TOKEN}@github.com/naicud/scudo.git"\n'
        'headers = {"Authorization": f"Bearer {self._key}"}\n')
    stage(repo, "cell.py")
    assert run_check(repo).returncode == 0


def test_exact_value_of_a_local_token_is_rejected_and_never_printed(repo: Path):
    secret = "Zq9" + "x7Kp2mW4" * 5  # nessun formato noto: lo riconosce solo il valore esatto
    (repo / "notes.txt").write_text(f"token: {secret}\n")
    stage(repo, "notes.txt")
    result = run_check(repo, env={"HF_TOKEN": secret, "SCUDO_NO_GH": "1"})
    assert result.returncode == 1
    assert "valore di un token locale" in result.stdout
    assert secret not in result.stdout + result.stderr


def test_local_hf_token_file_is_used(repo: Path):
    secret = "Qw8" + "r5Tn1vB3" * 5
    token_file = repo / "no-home" / ".cache" / "huggingface" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text(secret + "\n")
    (repo / "notes.txt").write_text(secret)
    stage(repo, "notes.txt")
    result = run_check(repo, env={"SCUDO_NO_GH": "1"})
    assert result.returncode == 1
    assert secret not in result.stdout
