"""Spot-check umano del lotto italiano: export del campione, import dell'esito, soglia (issue #7).

`export` sceglie per hash seminato `n` item italiani (default 300, il minimo della
specifica) dal pool deduplicato e, in più, gli item misti IT/EN nella stessa
proporzione del pool (`ceil(n · misti / italiani)`): entrano nel lotto, quindi si
rivedono. Scrive `spotcheck.csv` nella cartella del lotto: id opaco, criterio,
evidenza, verdetto proposto e due colonne vuote, `judgement` (`correct` /
`incorrect`) e `notes`. Il foglio è cieco sulla provenienza: niente fonte,
modello, tipo (tradotto o generato) né punteggi; la mappa id → item e lingua
resta in `spotcheck-meta.json`.

Il primo export per un pool si registra in `spotcheck-exports.jsonl` (seme,
dimensioni, hash di pool e foglio, ora UTC): un altro seme o un'altra dimensione
sullo stesso pool si rifiuta, così il campione non si sceglie rieseguendo
l'export. Lo stesso seme riscrive lo stesso foglio. Se il pool cambia, serve e
si ammette un nuovo export.

`import_review` legge il foglio compilato dal madrelingua, rifiuta righe
mancanti, estranee o senza giudizio e un campione che non coincide con l'export
registrato, e registra in `spotcheck-outcomes.jsonl` chi, quando, accuratezza
per lingua, seme dell'export e soglia. La soglia è 0,95 (specifica): si può
alzare, mai abbassare. Il lotto è accettato solo se italiani e misti, ciascuno,
sono almeno alla soglia; l'esito è legato all'hash del pool, e un lotto
respinto resta respinto: niente nuovo campione, niente mix
(`require_accepted`). Il codice non compila mai il giudizio.
"""

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from scudo.ingest import check_outside_repo
from scudo.italian_run import POOL, read_pool

SAMPLE_SIZE = 300
THRESHOLD = 0.95  # minimo della specifica (#7): si alza, non si abbassa
REVIEWED = ("it", "mixed")  # lingue del campione; l'inglese non passa dal madrelingua
JUDGEMENTS = ("correct", "incorrect")
COLUMNS = ("item_id", "criterion", "evidence", "verdict", "judgement", "notes")
SHEET, META = "spotcheck.csv", "spotcheck-meta.json"
EXPORTS, OUTCOMES = "spotcheck-exports.jsonl", "spotcheck-outcomes.jsonl"


class SpotCheckError(ValueError):
    """Spot-check mancante, incompleto o respinto."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _outcomes(out: Path) -> list[dict]:
    return _jsonl(out / OUTCOMES)


def exports(out: Path) -> list[dict]:
    """Gli export registrati, uno per pool."""
    return _jsonl(check_outside_repo(out) / EXPORTS)


def _refuse_rejected(out: Path) -> None:
    if any(not o["accepted"] for o in _outcomes(out)):
        raise SpotCheckError(f"{out}: lotto respinto dallo spot-check umano; serve un nuovo lotto")


def _now(now: str | None) -> str:
    return now or datetime.now(UTC).isoformat(timespec="seconds")


def sample(records: list[dict], n: int, seed: int) -> tuple[list[dict], dict[str, str]]:
    """n italiani più i misti in proporzione, per hash seminato; id opachi per chiave."""
    by_language = {lang: [r for r in records if r["language"] == lang] for lang in REVIEWED}
    italian, mixed = by_language["it"], by_language["mixed"]
    if len(italian) < n:
        raise SpotCheckError(f"item italiani nel pool {len(italian)}, ne servono {n}")
    n_mixed = min(len(mixed), math.ceil(n * len(mixed) / len(italian))) if italian else 0
    ids = {r["key"]: hashlib.sha256(f"{seed}:{r['key']}".encode()).hexdigest()[:12]
           for r in italian + mixed}
    chosen = (sorted(italian, key=lambda r: ids[r["key"]])[:n]
              + sorted(mixed, key=lambda r: ids[r["key"]])[:n_mixed])
    return sorted(chosen, key=lambda r: ids[r["key"]]), ids


def export(out: Path, *, n: int = SAMPLE_SIZE, seed: int = 0, now: str | None = None) -> Path:
    """Campione deterministico; fallisce se il pool non basta o ha già un altro export."""
    out = check_outside_repo(out)
    _refuse_rejected(out)
    pool_sha = _sha(out / POOL)
    if any(o["pool_sha256"] == pool_sha for o in _outcomes(out)):
        raise SpotCheckError("esito già registrato per questo pool")
    chosen, ids = sample(read_pool(out), n, seed)
    sizes = {lang: sum(r["language"] == lang for r in chosen) for lang in REVIEWED}
    previous = [e for e in exports(out) if e["pool_sha256"] == pool_sha]
    if previous and (previous[0]["seed"], previous[0]["n"]) != (seed, sizes):
        first = previous[0]
        raise SpotCheckError(f"campione già esportato per questo pool (seme {first['seed']}, "
                             f"{first['n']}): un altro seme o un'altra dimensione si "
                             "rifiutano finché il pool non cambia")
    with (out / SHEET).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for r in chosen:
            writer.writerow({"item_id": ids[r["key"]], "criterion": r["criterion"],
                             "evidence": r["evidence"], "verdict": r["verdict"],
                             "judgement": "", "notes": ""})
    record = previous[0] if previous else {
        "seed": seed, "n": sizes, "pool_sha256": pool_sha, "sheet_sha256": _sha(out / SHEET),
        "at": _now(now)}
    if not previous:
        _append(out / EXPORTS, record)
    meta = {**{k: v for k, v in record.items() if k != "at"}, "exported_at": record["at"],
            "items": {ids[r["key"]]: r["key"] for r in chosen},
            "languages": {ids[r["key"]]: r["language"] for r in chosen}}
    (out / META).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return out / SHEET


def _registered_meta(out: Path, pool_sha: str) -> dict:
    """Il meta dell'export registrato per il pool, ricontrollato sul campione ricalcolato."""
    if not (out / META).exists():
        raise SpotCheckError("nessun campione esportato: eseguire prima spotcheck-export")
    meta = json.loads((out / META).read_text(encoding="utf-8"))
    if meta.get("pool_sha256") != pool_sha:
        raise SpotCheckError("pool cambiato dopo l'export: esportare un nuovo campione")
    found = [e for e in exports(out) if e["pool_sha256"] == pool_sha]
    keys = ("seed", "n", "sheet_sha256")
    if not found or any(found[0][k] != meta.get(k) for k in keys) \
            or found[0]["at"] != meta.get("exported_at"):
        raise SpotCheckError("il campione non coincide con l'export registrato")
    chosen, ids = sample(read_pool(out), meta["n"]["it"], meta["seed"])
    if {ids[r["key"]]: r["key"] for r in chosen} != meta.get("items"):
        raise SpotCheckError("gli item del meta non coincidono con l'export registrato")
    return meta


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


def _correct(row: dict) -> bool:
    return row["judgement"].strip().lower() == "correct"


def import_review(out: Path, review: Path, *, by: str, threshold: float = THRESHOLD,
                  now: str | None = None) -> dict:
    """Registra l'esito del madrelingua; accettato solo se ogni lingua è almeno alla soglia."""
    out = check_outside_repo(out)
    if not by.strip():
        raise SpotCheckError("esito senza chi ha fatto la revisione")
    if not THRESHOLD <= threshold <= 1:
        raise SpotCheckError(f"soglia {threshold} fuori da [{THRESHOLD}, 1]: si alza, non si "
                             "abbassa")
    pool_sha = _sha(out / POOL)
    meta = _registered_meta(out, pool_sha)
    if any(o["pool_sha256"] == pool_sha for o in _outcomes(out)):
        raise SpotCheckError("esito già registrato per questo pool")
    rows = _read_review(review, meta["items"])
    wrong = [{"key": meta["items"][row["item_id"]], "notes": (row.get("notes") or "").strip()}
             for row in rows if not _correct(row)]
    by_language = {}
    for lang in REVIEWED:
        group = [row for row in rows if meta["languages"][row["item_id"]] == lang]
        if group:
            correct = sum(map(_correct, group))
            by_language[lang] = {"n": len(group), "correct": correct,
                                 "accuracy": correct / len(group)}
    accuracy = (len(rows) - len(wrong)) / len(rows)
    outcome = {"by": by.strip(), "at": _now(now),
               "n": len(rows), "correct": len(rows) - len(wrong), "incorrect": len(wrong),
               "accuracy": accuracy, "by_language": by_language, "threshold": threshold,
               "accepted": all(g["accuracy"] >= threshold for g in by_language.values()),
               "seed": meta["seed"], "exported_at": meta["exported_at"],
               "pool_sha256": pool_sha, "sheet_sha256": meta["sheet_sha256"],
               "review_sha256": _sha(Path(review)), "incorrect_items": wrong}
    _append(out / OUTCOMES, outcome)
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
