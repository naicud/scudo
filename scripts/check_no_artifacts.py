"""Blocca dati, pesi e token nei file tracciati o in stage. Solo standard library."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = {
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".npz",
    ".parquet",
    ".arrow",
    ".jsonl",
}
ALLOWED_PREFIXES = ("tests/fixtures/",)
TOKEN_PATTERNS = {
    "Hugging Face": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "Hugging Face OAuth": re.compile(rb"hf_oauth_[A-Za-z0-9_-]{20,}"),
    "OpenAI": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
    "Bearer": re.compile(rb"Bearer\s+[A-Za-z0-9._~+/-]{25,}=*"),
    "JWT": re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "chiave privata": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "URL con credenziali": re.compile(rb"https?://[^\s/:@\"'{}]+:[^\s@\"'{}]{8,}@"),
    "AWS": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "Google API": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
}
# Variabili d'ambiente che possono contenere un token di questa macchina.
TOKEN_VARS = ("HF_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN",
              "OPENAI_API_KEY")


def local_secrets() -> list[bytes]:
    """Valori esatti dei token di questa macchina: variabili, login HF, `gh`. Mai stampati."""
    values = [os.environ.get(name, "") for name in TOKEN_VARS]
    hf_file = Path.home() / ".cache" / "huggingface" / "token"
    if hf_file.is_file():
        values.append(hf_file.read_text(errors="ignore"))
    if not os.environ.get("SCUDO_NO_GH") and shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        values.append(result.stdout if result.returncode == 0 else "")
    return sorted({v.strip().encode() for v in values if len(v.strip()) >= 16})


def candidate_files(root: Path) -> list[str]:
    """File tracciati più quelli in stage, senza i cancellati."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return sorted({name for name in output.decode().split("\0") if name})


def problems_for(root: Path, name: str, secrets: list[bytes] = ()) -> list[str]:
    path = root / name
    if not path.is_file():
        return []
    problems = []
    allowed = name.startswith(ALLOWED_PREFIXES)
    if path.suffix.lower() in BLOCKED_SUFFIXES and not allowed:
        problems.append(f"{name}: estensione di dati o pesi ({path.suffix})")
    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{name}: {size} byte oltre il limite di {MAX_BYTES}")
        return problems
    content = path.read_bytes()
    for label, pattern in TOKEN_PATTERNS.items():
        if pattern.search(content):
            problems.append(f"{name}: possibile token {label}")
    if any(secret in content for secret in secrets):
        problems.append(f"{name}: contiene il valore di un token locale")
    return problems


def check(root: Path) -> int:
    names = candidate_files(root)
    secrets = local_secrets()
    problems = [problem for name in names for problem in problems_for(root, name, secrets)]
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"Controllati {len(names)} file; problemi: {len(problems)}")
    return int(bool(problems))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    raise SystemExit(check(target))
