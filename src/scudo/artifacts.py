"""Identità di un artefatto pubblicato: repository, revisione, file e digest.

Il digest si verifica prima di passare il file a qualsiasi provider pesante
(llama.cpp, transformers): un file mancante o diverso blocca, senza fallback.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class ArtifactIdentity:
    repo: str
    revision: str  # commit completo, mai un branch
    filename: str
    sha256: str

    def __post_init__(self):
        if not _REPO.match(self.repo):
            raise ValueError(f"repository non valido: {self.repo!r}")
        if not _REVISION.match(self.revision):
            raise ValueError("la revisione deve essere uno SHA di commit completo")
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError(f"nome file non valido: {self.filename!r}")
        if not _SHA256.match(self.sha256):
            raise ValueError("sha256 deve avere 64 cifre esadecimali minuscole")

    def manifest_entry(self) -> dict:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified(path: Path, identity: ArtifactIdentity, loader: Callable[[Path], T]) -> T:
    """Chiama `loader` solo su un file con nome e digest dell'identità dichiarata."""
    path = Path(path)
    if path.name != identity.filename:
        raise ArtifactError("ARTIFACT_NAME_MISMATCH", f"{path.name} != {identity.filename}")
    if not path.is_file():
        raise ArtifactError("ARTIFACT_MISSING", str(path))
    actual = file_sha256(path)
    if actual != identity.sha256:
        raise ArtifactError("ARTIFACT_DIGEST_MISMATCH", f"{actual} != {identity.sha256}")
    return loader(path)
