"""Mix del Fast v0.1 (issue #8): composizione, righe semantiche e di difficoltà.

Categoria di un caso, nell'ordine (prima regola che vale):

1. verdetto `INSUFFICIENT` -> `insufficient`;
2. superficie `over_refusal` -> `over_refusal`;
3. superficie `injection_direct`/`injection_indirect`, o criterio di injection dei
   criteri di default -> `injection`;
4. criterio di sicurezza dei contenuti (Aegis `HARMFUL_REQUEST`/`HARMFUL_RESPONSE`,
   criteri di default `aegis.*`) -> `prompt_response`;
5. ogni altro criterio (CantTalkAboutThis, procedural, Open-Jev, ...) -> `custom`.

I target sono parametri con i default dell'issue: categorie non INSUFFICIENT in
proporzione 45/20/15/10, INSUFFICIENT `insufficient.DEFAULT_SHARE` del mix (dentro
il 10–15%), lingua 40/40/20 IT/EN/misto dentro ogni categoria. Uno strato
(categoria, lingua) che non si riempie è un errore con tutte le mancanze: niente
riempimento parziale. Cambiare un target è un override registrato nel manifest del
mix con il motivo scritto dalla persona. Ogni origine degli INSUFFICIENT (metodo di
generazione o fonte) ha al più `method_cap` della quota INSUFFICIENT.
"""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from scudo import difficulty, mix_clean
from scudo.aegis import HARMFUL_REQUEST, HARMFUL_RESPONSE
from scudo.cases import LANGUAGES, Case, training_rows
from scudo.default_criteria import DEFAULT_CRITERIA, INJECTION_DIRECT, INJECTION_INDIRECT
from scudo.insufficient import DEFAULT_SHARE, mix_insufficient
from scudo.mix_pool import PoolCase
from scudo.protocol import FORMAT_VERSION

CATEGORIES = ("prompt_response", "injection", "custom", "over_refusal")  # più `insufficient`
DEFAULT_TOTAL = 200_000
DEFAULT_CATEGORIES = {"prompt_response": 45, "injection": 20, "custom": 15, "over_refusal": 10}
DEFAULT_LANGUAGES = {"it": 0.4, "en": 0.4, "mixed": 0.2}
# Scelta nostra, non un dato: nessuna origine sopra il 40% degli INSUFFICIENT, perché
# `evidence_removed` (dialogo che finisce con l'assistente) era l'87% del pool generato.
DEFAULT_METHOD_CAP = 0.4
INJECTION_CRITERIA = frozenset({INJECTION_DIRECT.text, INJECTION_INDIRECT.text})
SAFETY_CRITERIA = frozenset({HARMFUL_REQUEST, HARMFUL_RESPONSE}) | frozenset(
    c.text for c in DEFAULT_CRITERIA if c.id.startswith("aegis."))


class MixError(ValueError):
    """Il mix non si costruisce con questi target: nessun file viene scritto."""


def category(case: Case) -> str:
    if case.verdict == "INSUFFICIENT":
        return "insufficient"
    if case.surface == "over_refusal":
        return "over_refusal"
    if case.surface.startswith("injection_") or case.criterion in INJECTION_CRITERIA:
        return "injection"
    return "prompt_response" if case.criterion in SAFETY_CRITERIA else "custom"


@dataclass(frozen=True)
class Targets:
    total: int = DEFAULT_TOTAL  # righe semantiche
    categories: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CATEGORIES))
    languages: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_LANGUAGES))
    insufficient_share: float = DEFAULT_SHARE
    method_cap: float = DEFAULT_METHOD_CAP

    def __post_init__(self):
        if self.total <= 0:
            raise MixError("total deve essere positivo")
        for name, weights, keys in (("categorie", self.categories, CATEGORIES),
                                    ("lingue", self.languages, LANGUAGES)):
            if set(weights) != set(keys) or any(w < 0 for w in weights.values()) \
                    or not sum(weights.values()) > 0:
                raise MixError(f"{name}: servono pesi >= 0 per {keys}, non tutti nulli")
        if not math.isclose(sum(self.languages.values()), 1.0):
            raise MixError("le quote delle lingue devono sommare a 1")
        if not 0 < self.insufficient_share < 1 or not 0 < self.method_cap <= 1:
            raise MixError("insufficient_share in (0, 1), method_cap in (0, 1]")

    def overrides(self) -> dict:
        default = asdict(Targets())
        return {key: {"default": default[key], "value": value}
                for key, value in asdict(self).items() if value != default[key]}


def _split(n: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Parti intere che sommano a n (resto più grande, poi ordine delle chiavi)."""
    total = sum(weights.values())
    exact = {key: n * w / total for key, w in weights.items()}
    parts = {key: math.floor(x) for key, x in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - parts[k]), list(weights).index(k)))
    for key in order[:n - sum(parts.values())]:
        parts[key] += 1
    return parts


def quotas(targets: Targets) -> dict[tuple[str, str], int]:
    """Casi non INSUFFICIENT per (categoria, lingua); gli INSUFFICIENT sono il resto del totale."""
    labelled = round(targets.total * (1 - targets.insufficient_share))
    out = {}
    for cat, n in _split(labelled, dict(targets.categories)).items():
        for lang, k in _split(n, dict(targets.languages)).items():
            out[(cat, lang)] = k
    return out


def _order(seed: int, pooled: PoolCase) -> str:
    return hashlib.sha256(f"{seed}:{pooled.key}".encode()).hexdigest()


def _pick(cases: Sequence[PoolCase], n: int, seed: int) -> list[PoolCase]:
    return sorted(cases, key=lambda p: _order(seed, p))[:n]


def _select(cases: Sequence[PoolCase], targets: Targets, seed: int) -> list[PoolCase]:
    strata: dict[tuple[str, str], list[PoolCase]] = {}
    for pooled in cases:
        strata.setdefault((category(pooled.case), pooled.case.language), []).append(pooled)
    picked, short = [], []
    for (cat, lang), n in quotas(targets).items():
        pool = strata.get((cat, lang), [])
        if len(pool) < n:
            short.append(f"{cat}/{lang}: servono {n}, disponibili {len(pool)}")
        picked += _pick(pool, n, seed)
    if short:
        raise MixError("strati non riempibili (override dei target con il motivo):\n- "
                       + "\n- ".join(short))
    return picked


def _capped(cases: Sequence[PoolCase], cap: int, seed: int) -> list[PoolCase]:
    by_origin: dict[str, list[PoolCase]] = {}
    for pooled in cases:
        by_origin.setdefault(mix_clean.method(pooled), []).append(pooled)
    return [p for origin in sorted(by_origin) for p in _pick(by_origin[origin], cap, seed)]


def _insufficient(labelled: list[PoolCase], native: list[PoolCase], pool: list[PoolCase],
                  targets: Targets, seed: int) -> tuple[list[PoolCase], dict]:
    wanted = round(len(labelled) / (1 - targets.insufficient_share)) - len(labelled)
    cap = math.floor(targets.method_cap * wanted)
    available = Counter(mix_clean.method(p) for p in (*native, *pool))
    native, pool = _capped(native, cap, seed), _capped(pool, cap, seed)
    by_case = {p.case: p for p in (*labelled, *native, *pool)}
    try:
        mixed = mix_insufficient([p.case for p in (*labelled, *native)], [p.case for p in pool],
                                 share=targets.insufficient_share, seed=seed)
    except ValueError as error:
        raise MixError(f"INSUFFICIENT: {error} (tetto per origine {cap}, pool dopo il tetto "
                       f"{dict(Counter(mix_clean.method(p) for p in (*native, *pool)))})") from None
    chosen = [by_case[case] for case in mixed]
    origins = Counter(mix_clean.method(p) for p in chosen if p.case.verdict == "INSUFFICIENT")
    total = sum(origins.values())
    return chosen, {"wanted": wanted, "cap_per_origin": cap,
                    "pool_by_origin": dict(sorted(available.items())),
                    "by_origin": dict(sorted(origins.items())),
                    "share_by_origin": {k: v / total for k, v in sorted(origins.items())}}


def _composition(rows: Sequence[dict]) -> dict:
    fields_ = ("category", "surface", "language", "verdict", "entry")
    out = {f"by_{name}": dict(sorted(Counter(r[name] for r in rows).items())) for name in fields_}
    out["by_category_language"] = dict(sorted(Counter(
        f"{r['category']}/{r['language']}" for r in rows).items()))
    return {"rows": len(rows), **out}


def build(candidates: Sequence[PoolCase], *, outcomes: difficulty.Index, targets: Targets,
          seed: int, margin_threshold: float | None = difficulty.MARGIN_THRESHOLD
          ) -> tuple[list[dict], dict]:
    """Righe del mix (semantiche e di difficoltà) e statistiche per il manifest.

    `margin_threshold`: margine del teacher sotto cui un caso concorde è HARD
    (`difficulty.label`); None usa solo il disaccordo.
    """
    teachers = difficulty.jury(outcomes)
    clean, drops, conflicts = mix_clean.dedup(candidates)
    clean, relabelled = mix_clean.relabel(clean, outcomes, teachers)
    pool, derive_counts = mix_clean.generated(clean)
    native = [p for p in clean if p.case.verdict == "INSUFFICIENT"]
    labelled = _select([p for p in clean if p.case.verdict != "INSUFFICIENT"], targets, seed)
    chosen, insufficient = _insufficient(labelled, native, pool, targets, seed)
    insufficient["derive"] = dict(sorted(derive_counts.items()))
    rows = []
    for pooled, semantic in zip(chosen, training_rows([p.case for p in chosen], seed=seed),
                                strict=True):
        rows.append({**semantic, "task": "semantic", "entry": pooled.entry, "key": pooled.key,
                     "category": category(pooled.case)})
    rng, no_label, semantic_rows = random.Random(f"difficulty:{seed}"), Counter(), list(rows)
    reasons = Counter()
    for pooled, semantic in zip(chosen, semantic_rows, strict=True):
        verdict, reason = difficulty.label(outcomes, pooled.key, pooled.case.verdict, teachers,
                                           margin_threshold=margin_threshold)
        if verdict is None:
            no_label[reason] += 1
            continue
        reasons[reason] += 1
        rows.append({**{k: semantic[k] for k in ("source", "source_id", "language", "surface",
                                                  "entry", "key", "category")},
                     **difficulty.row(pooled.case, verdict, rng), "task": "difficulty",
                     "verdict": verdict, "semantic_verdict": pooled.case.verdict})
    difficulty_rows = [r for r in rows if r["task"] == "difficulty"]
    hard = Counter(r["difficulty"] for r in difficulty_rows)
    if not hard["HARD"] or not hard["EASY"]:
        raise MixError(f"righe di difficoltà {dict(hard)}: servono casi HARD ed EASY dagli esiti "
                       f"dei teacher (senza label: {dict(no_label)})")
    rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['task']}:{r['key']}".encode()).hexdigest())
    stats = {
        "format": FORMAT_VERSION, "difficulty_criterion": difficulty.ROUTING_DIFFICULTY_REF,
        "difficulty_margin_threshold": margin_threshold, "seed": seed, "targets": asdict(targets), "overrides": targets.overrides(),
        "jury": list(teachers),
        "candidates": dict(sorted(Counter(p.entry for p in candidates).items())),
        "drops": dict(sorted(drops.items())), "label_conflicts": dict(sorted(conflicts.items())),
        "relabel": dict(sorted(relabelled.items())), "insufficient": insufficient,
        "composition": {"semantic": _composition(semantic_rows),
                        "difficulty": {**_composition(difficulty_rows),
                                       "by_difficulty": dict(sorted(hard.items())),
                                       "by_reason": dict(sorted(reasons.items())),
                                       "no_label": dict(sorted(no_label.items()))}},
    }
    return rows, stats
