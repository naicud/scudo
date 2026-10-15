"""Spot-check umano del lotto italiano: export del campione, import dell'esito, soglia (issue #7).

`export` sceglie per hash seminato `n` item italiani (default 300) dal pool
deduplicato e scrive `spotcheck.csv` nella cartella del lotto: id opaco,
criterio, evidenza, verdetto proposto e due colonne vuote, `judgement`
(`correct` / `incorrect`) e `notes`. Il foglio è cieco sulla provenienza: niente
fonte, modello, tipo (tradotto o generato) né punteggi; la mappa id → item resta
in `spotcheck-meta.json`.

`import_review` legge il foglio compilato dal madrelingua, rifiuta righe
mancanti, estranee o senza giudizio, e registra in `spotcheck-outcomes.jsonl`
chi, quando, accuratezza e soglia (default 0,95). Il lotto è accettato solo se
l'accuratezza è almeno la soglia; l'esito è legato all'hash del pool, e un
lotto respinto resta respinto: niente nuovo campione, niente mix
(`require_accepted`). Il codice non compila mai il giudizio.
"""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scudo.ingest import check_outside_repo
from scudo.italian_run import POOL, read_pool

SAMPLE_SIZE = 300
THRESHOLD = 0.95
JUDGEMENTS = ("correct", "incorrect")
COLUMNS = ("item_id", "criterion", "evidence", "verdict", "judgement", "notes")
SHEET, META, OUTCOMES = "spotcheck.csv", "spotcheck-meta.json", "spotcheck-outcomes.jsonl"


class SpotCheckError(ValueError):
    """Spot-check mancante, incompleto o respinto."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcomes(out: Path) -> list[dict]:
    path = out / OUTCOMES
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _refuse_rejected(out: Path) -> None:
    if any(not o["accepted"] for o in _outcomes(out)):
        raise SpotCheckError(f"{out}: lotto respinto dallo spot-check umano; serve un nuovo lotto")


def export(out: Path, *, n: int = SAMPLE_SIZE, seed: int = 0) -> Path:
    """Campione deterministico di n item italiani; fallisce se il pool non basta."""
    out = check_outside_repo(out)
    _refuse_rejected(out)
    pool_sha = _sha(out / POOL)
    if any(o["pool_sha256"] == pool_sha for o in _outcomes(out)):
        raise SpotCheckError("esito già registrato per questo pool")
    italian = [r for r in read_pool(out) if r["language"] == "it"]
    if len(italian) < n:
        raise SpotCheckError(f"item italiani nel pool {len(italian)}, ne servono {n}")
    ids = {r["key"]: hashlib.sha256(f"{seed}:{r['key']}".encode()).hexdigest()[:12]
           for r in italian}
    sample = sorted(italian, key=lambda r: ids[r["key"]])[:n]
    with (out / SHEET).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for r in sample:
            writer.writerow({"item_id": ids[r["key"]], "criterion": r["criterion"],
                             "evidence": r["evidence"], "verdict": r["verdict"],
                             "judgement": "", "notes": ""})
    meta = {"seed": seed, "n": n, "pool_sha256": pool_sha, "sheet_sha256": _sha(out / SHEET),
            "items": {ids[r["key"]]: r["key"] for r in sample}}
    (out / META).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return out / SHEET


def _read_review(review: Path, items: dict) -> list[dict]:
    with check_outside_repo(review).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("item_id", "") for row in rows]
    if sorted(ids) != sorted(items):
        raise SpotCheckError(f"gli item del foglio non coincidono con i {len(items)} esportati")
    bad = [row["item_id"] for row in rows
           if (row.get("judgement") or "").strip().lower() not in JUDGEMENTS]
    if bad:
        raise SpotCheckError(f"{len(bad)} righe senza judgement valido {JUDGEMENTS}: {bad[:5]}")
    return rows


def import_review(out: Path, review: Path, *, by: str, threshold: float = THRESHOLD,
                  now: str | None = None) -> dict:
    """Registra l'esito del madrelingua; accettato solo se l'accuratezza è almeno la soglia."""
    out = check_outside_repo(out)
    if not by.strip():
        raise SpotCheckError("esito senza chi ha fatto la revisione")
    if not 0 < threshold <= 1:
        raise SpotCheckError("soglia fuori da (0, 1]")
    if not (out / META).exists():
        raise SpotCheckError("nessun campione esportato: eseguire prima spotcheck-export")
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    pool_sha = _sha(out / POOL)
    if pool_sha != meta["pool_sha256"]:
        raise SpotCheckError("pool cambiato dopo l'export: esportare un nuovo campione")
    if any(o["pool_sha256"] == pool_sha for o in _outcomes(out)):
        raise SpotCheckError("esito già registrato per questo pool")
    rows = _read_review(review, meta["items"])
    wrong = [{"key": meta["items"][row["item_id"]], "notes": (row.get("notes") or "").strip()}
             for row in rows if row["judgement"].strip().lower() == "incorrect"]
    accuracy = (len(rows) - len(wrong)) / len(rows)
    outcome = {"by": by.strip(), "at": now or datetime.now(UTC).isoformat(timespec="seconds"),
               "n": len(rows), "correct": len(rows) - len(wrong), "incorrect": len(wrong),
               "accuracy": accuracy, "threshold": threshold, "accepted": accuracy >= threshold,
               "pool_sha256": pool_sha, "sheet_sha256": meta["sheet_sha256"],
               "review_sha256": _sha(Path(review)), "incorrect_items": wrong}
    with (out / OUTCOMES).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")
    return outcome


def require_accepted(out: Path) -> dict:
    """L'esito accettato per il pool attuale, o SpotCheckError: senza di esso niente mix."""
    out = check_outside_repo(out)
    _refuse_rejected(out)
    pool_sha = _sha(out / POOL)
    found = [o for o in _outcomes(out) if o["pool_sha256"] == pool_sha]
    if not found:
        raise SpotCheckError(f"{out}: nessuno spot-check accettato per il pool attuale")
    return found[-1]
