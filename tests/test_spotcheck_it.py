import csv
import json

import pytest

from scudo import italian_run, spotcheck_it
from scudo.spotcheck_it import SpotCheckError


def item(i, language="it"):
    return {"key": f"k{i}", "origin": "generated", "source": "scudo/teacher-generated",
            "source_id": f"s{i}", "language": language, "surface": "prompt",
            "criterion": "Is it unsafe?", "evidence": f"Testo numero {i}\ncon a capo",
            "verdict": "VIOLATION", "provenance": {"generator": {"model_id": "m"}}}


def pool(tmp_path, n_it=30):
    out = tmp_path / "it"
    out.mkdir()
    records = [item(i) for i in range(n_it)] + [item(100, "en"), item(101, "mixed")]
    (out / italian_run.POOL).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return out


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fill(out, tmp_path, wrong, *, blank=0, name="review.csv"):
    """La persona compila il CSV: `wrong` righe errate, `blank` lasciate vuote."""
    data = rows(out / spotcheck_it.SHEET)
    for i, row in enumerate(data):
        row["judgement"] = "" if i < blank else "incorrect" if i < blank + wrong else "correct"
        row["notes"] = "registro innaturale" if row["judgement"] == "incorrect" else ""
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spotcheck_it.COLUMNS)
        writer.writeheader()
        writer.writerows(data)
    return path


def test_export_is_deterministic_blind_and_only_italian(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=1)
    first = (out / spotcheck_it.SHEET).read_bytes()
    spotcheck_it.export(out, n=20, seed=1)
    assert (out / spotcheck_it.SHEET).read_bytes() == first
    data = rows(out / spotcheck_it.SHEET)
    assert len(data) == 20 and tuple(data[0]) == spotcheck_it.COLUMNS
    assert all(r["judgement"] == "" and r["notes"] == "" for r in data)
    text = first.decode("utf-8-sig")
    assert "scudo/teacher-generated" not in text and "generated" not in text and "k1" not in text
    keys = set(json.loads((out / spotcheck_it.META).read_text())["items"].values())
    assert keys <= {f"k{i}" for i in range(30)}  # mai en o mixed


def test_export_fails_when_the_italian_pool_is_too_small(tmp_path):
    with pytest.raises(SpotCheckError, match="servono 300"):
        spotcheck_it.export(pool(tmp_path))


@pytest.mark.parametrize(("wrong", "accepted"), [(1, True), (2, False)])
def test_batch_is_accepted_only_at_or_above_the_threshold(tmp_path, wrong, accepted):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    outcome = spotcheck_it.import_review(out, fill(out, tmp_path, wrong), by="Giulia Verdi")
    assert outcome["accepted"] is accepted and outcome["n"] == 20
    assert outcome["accuracy"] == pytest.approx((20 - wrong) / 20)
    assert outcome["threshold"] == 0.95 and outcome["by"] == "Giulia Verdi" and outcome["at"]
    assert len(outcome["incorrect_items"]) == wrong
    if accepted:
        assert spotcheck_it.require_accepted(out)["accepted"]
    else:
        with pytest.raises(SpotCheckError, match="respinto"):
            spotcheck_it.require_accepted(out)
        with pytest.raises(SpotCheckError, match="respinto"):
            spotcheck_it.export(out, n=20, seed=1)  # niente nuovo campione sullo stesso lotto


def test_incomplete_or_foreign_review_is_refused(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    with pytest.raises(SpotCheckError, match="judgement"):
        spotcheck_it.import_review(out, fill(out, tmp_path, 0, blank=1), by="Giulia")
    with pytest.raises(SpotCheckError, match="chi"):
        spotcheck_it.import_review(out, fill(out, tmp_path, 0), by=" ")
    path = fill(out, tmp_path, 0)
    data = rows(path)[1:]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spotcheck_it.COLUMNS)
        writer.writeheader()
        writer.writerows(data)
    with pytest.raises(SpotCheckError, match="item"):
        spotcheck_it.import_review(out, path, by="Giulia")
    with pytest.raises(SpotCheckError):
        spotcheck_it.require_accepted(out)  # nessun esito registrato


def test_outcome_is_recorded_once_and_bound_to_the_pool(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    spotcheck_it.import_review(out, fill(out, tmp_path, 0), by="Giulia")
    with pytest.raises(SpotCheckError, match="già registrato"):
        spotcheck_it.import_review(out, fill(out, tmp_path, 0), by="Giulia")
    with (out / italian_run.POOL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item(999)) + "\n")
    with pytest.raises(SpotCheckError, match="pool"):
        spotcheck_it.require_accepted(out)


def test_pool_changed_after_export_is_refused_at_import(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    review = fill(out, tmp_path, 0)
    with (out / italian_run.POOL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item(999)) + "\n")
    with pytest.raises(SpotCheckError, match="pool"):
        spotcheck_it.import_review(out, review, by="Giulia")
