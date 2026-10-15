"""Casi del tracer da `nvidia/Nemotron-Safety-Guard-Dataset-v3` (CC-BY-4.0), solo IT/EN.

Una riga dà un caso sulla richiesta (superficie `prompt`, evidenza = prompt) e,
se la risposta ha una label, un caso sulla risposta (superficie `response`,
evidenza = sola risposta, come la superficie output del gateway). Restano fuori
le righe `refusal_data_augmentation` (rifiuti generati con Gemma, termini propri)
e i prompt `REDACTED`. Il tracer non costruisce casi INSUFFICIENT (issue #6).
Lo schema è quello di Aegis 2.0 (`scudo.aegis`), con la colonna `language`; in
ingestione le label `llm_jury` sono marcate da `aegis.needs_relabel`.
"""

import hashlib
from collections.abc import Iterable, Sequence

from scudo import aegis
from scudo.aegis import (  # noqa: F401
    EXCLUDED_RESPONSE_SOURCES,
    HARMFUL_REQUEST,
    HARMFUL_RESPONSE,
    needs_relabel,
)
from scudo.cases import Case

DATASET = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
REVISION = "a3f7ecb3433d1933701a83f18de16c36934a7f51"
LANGUAGES = ("it", "en")


def drop_reason(row: dict) -> str | None:
    """Motivo per cui la riga resta fuori, o None se entra."""
    if row["language"] not in LANGUAGES:
        return "lingua fuori da it/en"
    return aegis.drop_reason(row)


def cases_from_row(row: dict) -> list[Case]:
    """Schema Aegis 2.0 con colonna `language`; criteri e filtri sono quelli di Aegis."""
    if drop_reason(row):
        return []
    return aegis.cases_from_row(row, source=DATASET, language=row["language"])


def _sample_key(seed: int, case: Case) -> str:
    return hashlib.sha256(f"{seed}:{case.source_id}:{case.surface}".encode()).hexdigest()


def sample_cases(rows: Iterable[dict], *, per_language: int, seed: int) -> list[Case]:
    """Campione deterministico, indipendente dall'ordine delle righe, uguale per lingua."""
    by_language: dict[str, list[Case]] = {language: [] for language in LANGUAGES}
    for row in rows:
        for case in cases_from_row(row):
            by_language[case.language].append(case)
    sample = []
    for language in LANGUAGES:
        pool = sorted(by_language[language], key=lambda case: _sample_key(seed, case))
        if len(pool) < per_language:
            raise ValueError(f"{language}: {len(pool)} casi, ne servono {per_language}")
        sample.extend(pool[:per_language])
    return sample


def heldout_cases(
    test_rows: Iterable[dict], train: Sequence[Case], *, per_language: int, seed: int
) -> list[Case]:
    """Campione dello split di test senza fonti né testi già visti in training."""
    seen_ids = {case.source_id for case in train}
    seen_texts = {case.evidence for case in train}
    fresh = (
        row for row in test_rows
        if row["id"] not in seen_ids and row["prompt"] not in seen_texts
        and row.get("response") not in seen_texts
    )
    return sample_cases(fresh, per_language=per_language, seed=seed)
