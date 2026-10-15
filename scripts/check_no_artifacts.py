"""Blocca dati, pesi e token nei file tracciati o in stage. Solo standard library."""

import re
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
    "OpenAI": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub": re.compile(rb"gh[po]_[A-Za-z0-9]{30,}"),
}


def candidate_files(root: Path) -> list[str]:
    """File tracciati più quelli in stage, senza i cancellati."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return sorted({name for name in output.decode().split("\0") if name})


def problems_for(root: Path, name: str) -> list[str]:
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
    return problems


def check(root: Path) -> int:
    names = candidate_files(root)
    problems = [problem for name in names for problem in problems_for(root, name)]
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"Controllati {len(names)} file; problemi: {len(problems)}")
    return int(bool(problems))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    raise SystemExit(check(target))
