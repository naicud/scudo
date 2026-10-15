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


def pool(tmp_path, n_it=30, n_mixed=1):
    out = tmp_path / "it"
    out.mkdir()
    records = [item(i) for i in range(n_it)] + [item(100, "en")] + [
        item(101 + i, "mixed") for i in range(n_mixed)]
    (out / italian_run.POOL).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return out


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def languages(out):
    meta = json.loads((out / spotcheck_it.META).read_text())
    return meta["languages"]


def fill(out, tmp_path, wrong, *, blank=0, name="review.csv", language=None):
    """La persona compila il CSV: `wrong` righe errate, `blank` vuote; solo `language` se dato."""
    data = rows(out / spotcheck_it.SHEET)
    lang = languages(out)
    eligible = [i for i, row in enumerate(data)
                if language is None or lang[row["item_id"]] == language]
    bad = set([i for i in eligible if i >= blank][:wrong])
    for i, row in enumerate(data):
        row["judgement"] = "" if i < blank else "incorrect" if i in bad else "correct"
        row["notes"] = "registro innaturale" if row["judgement"] == "incorrect" else ""
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spotcheck_it.COLUMNS)
        writer.writeheader()
        writer.writerows(data)
    return path


def test_export_is_deterministic_blind_and_recorded(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=1, now="2026-10-20T10:00:00+00:00")
    first = (out / spotcheck_it.SHEET).read_bytes()
    spotcheck_it.export(out, n=20, seed=1)
    assert (out / spotcheck_it.SHEET).read_bytes() == first
    data = rows(out / spotcheck_it.SHEET)
    assert len(data) == 21 and tuple(data[0]) == spotcheck_it.COLUMNS  # 20 it + 1 misto
    assert all(r["judgement"] == "" and r["notes"] == "" for r in data)
    text = first.decode("utf-8-sig")
    assert "scudo/teacher-generated" not in text and "generated" not in text and "k1" not in text
    meta = json.loads((out / spotcheck_it.META).read_text())
    assert set(meta["items"].values()) <= {f"k{i}" for i in range(30)} | {"k101"}  # mai en
    [export] = spotcheck_it.exports(out)  # una sola registrazione per pool e seme
    assert export["seed"] == 1 and export["at"] == "2026-10-20T10:00:00+00:00"
    assert export["pool_sha256"] == meta["pool_sha256"] and export["n"] == {"it": 20, "mixed": 1}


def test_sample_adds_mixed_items_in_proportion_on_top_of_the_italian_ones(tmp_path):
    out = pool(tmp_path, n_it=30, n_mixed=15)
    spotcheck_it.export(out, n=20, seed=0)
    counts = {}
    for language in languages(out).values():
        counts[language] = counts.get(language, 0) + 1
    assert counts == {"it": 20, "mixed": 10}  # 15/30 dei misti per item italiano


def test_export_fails_when_the_italian_pool_is_too_small(tmp_path):
    with pytest.raises(SpotCheckError, match="servono 300"):
        spotcheck_it.export(pool(tmp_path))


def test_another_seed_or_size_for_the_same_pool_is_refused(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    with pytest.raises(SpotCheckError, match="già esportato"):
        spotcheck_it.export(out, n=20, seed=1)  # niente scelta del campione
    with pytest.raises(SpotCheckError, match="già esportato"):
        spotcheck_it.export(out, n=25, seed=0)
    spotcheck_it.export(out, n=20, seed=0)
    with (out / italian_run.POOL).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item(999)) + "\n")
    spotcheck_it.export(out, n=20, seed=1)  # pool cambiato: nuovo campione ammesso
    assert [e["seed"] for e in spotcheck_it.exports(out)] == [0, 1]


def test_import_requires_the_recorded_export(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    review = fill(out, tmp_path, 0)
    meta = json.loads((out / spotcheck_it.META).read_text())
    (out / spotcheck_it.META).write_text(json.dumps({**meta, "seed": 7}))
    with pytest.raises(SpotCheckError, match="export"):
        spotcheck_it.import_review(out, review, by="Giulia")
    (out / spotcheck_it.META).write_text(json.dumps(meta))
    (out / spotcheck_it.EXPORTS).unlink()
    with pytest.raises(SpotCheckError, match="export"):
        spotcheck_it.import_review(out, review, by="Giulia")


@pytest.mark.parametrize(("wrong", "accepted"), [(1, True), (2, False)])
def test_batch_is_accepted_only_at_or_above_the_threshold(tmp_path, wrong, accepted):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0, now="2026-10-20T10:00:00+00:00")
    review = fill(out, tmp_path, wrong, language="it")
    outcome = spotcheck_it.import_review(out, review, by="Giulia Verdi")
    assert outcome["accepted"] is accepted and outcome["n"] == 21
    assert outcome["by_language"]["it"]["accuracy"] == pytest.approx((20 - wrong) / 20)
    assert outcome["by_language"]["mixed"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert outcome["threshold"] == 0.95 and outcome["by"] == "Giulia Verdi" and outcome["at"]
    assert outcome["seed"] == 0 and outcome["exported_at"] == "2026-10-20T10:00:00+00:00"
    assert len(outcome["incorrect_items"]) == wrong
    if accepted:
        assert spotcheck_it.require_accepted(out)["accepted"]
    else:
        with pytest.raises(SpotCheckError, match="respinto"):
            spotcheck_it.require_accepted(out)
        with pytest.raises(SpotCheckError, match="respinto"):
            spotcheck_it.export(out, n=20, seed=1)  # niente nuovo campione sullo stesso lotto


def test_mixed_items_below_the_threshold_reject_the_batch(tmp_path):
    out = pool(tmp_path, n_it=30, n_mixed=15)
    spotcheck_it.export(out, n=20, seed=0)
    outcome = spotcheck_it.import_review(out, fill(out, tmp_path, 1, language="mixed"),
                                         by="Giulia")
    assert outcome["accuracy"] == pytest.approx(29 / 30)  # sopra 0,95 nel complesso
    assert outcome["by_language"]["mixed"]["accuracy"] == pytest.approx(0.9)
    assert outcome["accepted"] is False


@pytest.mark.parametrize("threshold", [0.5, 0.949, 1.01])
def test_threshold_cannot_be_lowered_below_the_spec(tmp_path, threshold):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    with pytest.raises(SpotCheckError, match="soglia"):
        spotcheck_it.import_review(out, fill(out, tmp_path, 0), by="Giulia", threshold=threshold)
    assert not (out / spotcheck_it.OUTCOMES).exists()


def test_threshold_can_be_raised_and_is_recorded(tmp_path):
    out = pool(tmp_path)
    spotcheck_it.export(out, n=20, seed=0)
    outcome = spotcheck_it.import_review(out, fill(out, tmp_path, 1, language="it"), by="Giulia",
                                         threshold=0.97)
    assert outcome["threshold"] == 0.97 and outcome["accepted"] is False


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
