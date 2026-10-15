"""Campione del pilot dei teacher: casi gold con label umana e righe `llm_jury`.

Il campione gold è stratificato per lingua, superficie e verdetto con quote
uguali (la quota che uno strato non copre va agli altri) e scelto per hash
seminato, quindi deterministico e indipendente dall'ordine delle righe. Ogni
fonte passa il cancello del manifest nel proprio ruolo; una fonte bloccata non
si legge. I casi di un benchmark sono solo valutazione (`training_eligible`
falso): le uscite del teacher su di essi non entrano mai nel training. Le righe
`llm_jury` vengono solo dalle fonti di training, dove la rietichettatura serve.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from functools import partial
from pathlib import Path

from scudo.cases import LANGUAGES
from scudo.ingest import CONVERTERS, check_outside_repo
from scudo.manifest import Entry, problems

PILOT_CASES = 500
RELABEL_CASES = 200
# Fonte -> lingue usate. nemotron3-train è bloccato dal cancello (termini INCERTO):
# l'italiano viene dal benchmark nemotron3-test, solo valutazione.
DEFAULT_SOURCES = {"aegis2-train": ("en",), "nemotron3-test": ("it",)}
GOLD_LABEL_SOURCE = "human"
RELABEL_LABEL_SOURCE = "llm_jury"
Loader = Callable[[Entry, str], Iterable[dict]]


def allocate(sizes: Mapping, n: int) -> dict:
    """Quote uguali per strato; la quota che uno strato non copre va agli altri."""
    quota = {key: 0 for key in sizes}
    left = n
    open_ = sorted(key for key in sizes if sizes[key] > 0)
    while left and open_:
        share, extra = divmod(left, len(open_))
        for i, key in enumerate(open_):
            take = min(share + (i < extra), sizes[key] - quota[key])
            quota[key] += take
            left -= take
        open_ = [key for key in open_ if quota[key] < sizes[key]]
    if left:
        raise ValueError(f"servono {n} casi, ce ne sono {n - left}")
    return quota


def _key(seed: int, record: dict) -> str:
    return hashlib.sha256(f"{seed}:{record['key']}".encode()).hexdigest()


def _stratified(records: list[dict], n: int, stratum: Callable[[dict], str], seed: int):
    pools: dict[str, list[dict]] = {}
    for record in records:
        pools.setdefault(stratum(record), []).append(record)
    quota = allocate({key: len(pool) for key, pool in pools.items()}, n)
    picked = [record for key in sorted(pools)
              for record in sorted(pools[key], key=partial(_key, seed))[:quota[key]]]
    return sorted(picked, key=partial(_key, seed)), {k: v for k, v in sorted(quota.items())}


def _records(entry: Entry, rows: Iterable[dict], languages: tuple[str, ...]):
    converter = CONVERTERS[entry.id]
    for row in rows:
        if converter.drop_reason(row):
            continue
        for case in converter.cases_from_row(row):
            if case.language in languages:
                yield {"key": f"{entry.name}:{case.source_id}:{case.surface}",
                       "entry": entry.name, "role": entry.role,
                       "training_eligible": entry.role == "train",
                       "label_source": row.get(f"{case.surface}_label_source"),
                       **asdict(case)}


def build_sample(entries: list[Entry], loader: Loader, *, sources: Mapping[str, tuple],
                 n: int = PILOT_CASES, relabel_n: int = RELABEL_CASES, seed: int = 0) -> dict:
    """Campione gold (label umane) e campione `llm_jury` dalle sole fonti di training."""
    by_name = {entry.name: entry for entry in entries}
    gate = {e.name: [p.removeprefix(f"{e.name}: ") for p in found]
            for e in entries if (found := problems([e]))}
    gold, relabel, blocked = [], [], {}
    for name, languages in sources.items():
        entry = by_name[name]
        if name in gate or entry.id not in CONVERTERS:
            blocked[name] = gate.get(name, ["nessun convertitore"])
            continue
        for split in entry.splits:
            for record in _records(entry, loader(entry, split), tuple(languages)):
                if record["label_source"] == GOLD_LABEL_SOURCE:
                    gold.append(record)
                elif record["label_source"] == RELABEL_LABEL_SOURCE and entry.role == "train":
                    relabel.append(record)
    gold, gold_strata = _stratified(
        gold, n, lambda r: f"{r['language']}/{r['surface']}/{r['verdict']}", seed)
    relabel, relabel_strata = _stratified(
        relabel, relabel_n, lambda r: f"{r['language']}/{r['surface']}/{r['verdict']}", seed)
    meta = {"seed": seed, "sources": {k: list(v) for k, v in sources.items()},
            "gold_strata": gold_strata, "relabel_strata": relabel_strata,
            "blocked": blocked, "gate": gate,
            "languages_missing": [lang for lang in LANGUAGES
                                  if not any(r["language"] == lang for r in gold)]}
    return {"gold": gold, "relabel": relabel, "meta": meta}


def write_sample(out: Path, sample: dict) -> Path:
    out = check_outside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sample.jsonl").open("w", encoding="utf-8") as handle:
        for kind in ("gold", "relabel"):
            for record in sample[kind]:
                handle.write(json.dumps({"set": kind, **record}, ensure_ascii=False) + "\n")
    (out / "sample-meta.json").write_text(
        json.dumps(sample["meta"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
