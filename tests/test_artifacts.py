import hashlib

import pytest

from scudo.artifacts import ArtifactError, ArtifactIdentity, load_verified

REVISION = "0" * 40


def identity_for(tmp_path, content: bytes) -> tuple:
    path = tmp_path / "scudo-fast-4b.Q8_0.gguf"
    path.write_bytes(content)
    identity = ArtifactIdentity(
        repo="naicud/scudo-fast-4b-GGUF",
        revision=REVISION,
        filename=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    return path, identity


def test_verified_artifact_reaches_the_loader(tmp_path):
    path, identity = identity_for(tmp_path, b"GGUF fake weights")
    assert load_verified(path, identity, lambda p: ("loaded", p)) == ("loaded", path)


def test_digest_mismatch_blocks_before_the_heavy_loader(tmp_path):
    path, identity = identity_for(tmp_path, b"GGUF fake weights")
    path.write_bytes(b"GGUF tampered weights")
    calls = []
    with pytest.raises(ArtifactError) as error:
        load_verified(path, identity, calls.append)
    assert error.value.code == "ARTIFACT_DIGEST_MISMATCH"
    assert calls == []


def test_missing_file_blocks_before_the_heavy_loader(tmp_path):
    path, identity = identity_for(tmp_path, b"x")
    path.unlink()
    calls = []
    with pytest.raises(ArtifactError) as error:
        load_verified(path, identity, calls.append)
    assert error.value.code == "ARTIFACT_MISSING"
    assert calls == []


def test_filename_must_match_identity(tmp_path):
    path, identity = identity_for(tmp_path, b"x")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"x")
    with pytest.raises(ArtifactError) as error:
        load_verified(other, identity, lambda p: p)
    assert error.value.code == "ARTIFACT_NAME_MISMATCH"


@pytest.mark.parametrize(
    "field, value",
    [("sha256", "abc"), ("revision", "main"), ("repo", "scudo-fast"), ("filename", "../x.gguf")],
)
def test_identity_rejects_unpinned_or_malformed_fields(field, value):
    fields = {
        "repo": "naicud/scudo-fast-4b-GGUF",
        "revision": REVISION,
        "filename": "m.gguf",
        "sha256": "a" * 64,
    }
    fields[field] = value
    with pytest.raises(ValueError):
        ArtifactIdentity(**fields)
