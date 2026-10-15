"""Pulizia dei candidati al mix: duplicati, conflitti di label, rietichettatura, INSUFFICIENT.

- Duplicati: stesso criterio e stessa evidenza normalizzata (`scudo.dedup.normalize`,
  hash esatto), dentro una fonte o fra fonti (LLMail ripete le email fra livelli,
  Aegis 1.0 e 2.0 vengono da HH-RLHF, jackhhao e in-the-wild da jailbreak_llms).
  Con verdetti uguali resta un caso: prima quello con label gold, poi per voce e id.
- Conflitti: stesso criterio ed evidenza con verdetti diversi. Escono tutte le
  copie, contate: non si sceglie mai una label in silenzio.
- Rietichettatura: un caso `needs_relabel` entra solo se ogni teacher della giuria
  ha un esito `ok` uguale alla sua label; gli esiti del teacher sono candidati,
  non gold, quindi una giuria discorde non cambia la label ma esclude il caso.
- INSUFFICIENT generati (`scudo.insufficient.derive_all`) dai casi puliti; un
  figlio con criterio ed evidenza già presenti fra i candidati esce.
"""

from collections import Counter
from collections.abc import Sequence

from scudo import difficulty
from scudo.dedup import digest, normalize
from scudo.insufficient import derive_all
from scudo.mix_pool import PoolCase


def evidence_key(pooled: PoolCase) -> tuple[str, str]:
    text = pooled.case.evidence
    return pooled.case.criterion, digest(text) if normalize(text) else "raw:" + text


def dedup(cases: Sequence[PoolCase]) -> tuple[list[PoolCase], Counter, Counter]:
    """(tenuti, scarti per motivo, conflitti per coppia di voci)."""
    groups: dict[tuple[str, str], list[PoolCase]] = {}
    for pooled in cases:
        groups.setdefault(evidence_key(pooled), []).append(pooled)
    kept, drops, conflicts = [], Counter(), Counter()
    for group in groups.values():
        if len({p.case.verdict for p in group}) > 1:
            drops["label_conflict"] += len(group)
            conflicts["|".join(sorted({p.entry for p in group}))] += 1
            continue
        best = min(group, key=lambda p: (p.needs_relabel, p.entry, p.case.source_id))
        kept.append(best)
        for other in group:
            if other is not best:
                same = other.entry == best.entry
                drops["duplicate_within_source" if same else "duplicate_cross_source"] += 1
    return kept, drops, conflicts


def relabel(cases: Sequence[PoolCase], index: difficulty.Index,
            teachers: Sequence[str]) -> tuple[list[PoolCase], Counter]:
    """I casi `needs_relabel` restano solo con la conferma unanime della giuria."""
    kept, counts = [], Counter()
    for pooled in cases:
        if not pooled.needs_relabel:
            kept.append(pooled)
            continue
        status, verdicts = difficulty.assess(index, pooled.key, teachers)
        if status != "ok":
            counts[f"relabel_{status}"] += 1
        elif set(verdicts) != {pooled.case.verdict}:
            counts["relabel_jury_disagrees"] += 1
        else:
            counts["relabel_confirmed"] += 1
            kept.append(pooled)
    return kept, counts


def generated(cases: Sequence[PoolCase]) -> tuple[list[PoolCase], Counter]:
    """INSUFFICIENT derivati dai casi; esclusioni per `metodo:motivo`."""
    parents = {(p.case.source, p.case.source_id): p for p in cases}
    children, skipped = derive_all([p.case for p in cases])
    taken = {evidence_key(p) for p in cases}
    out, counts = [], Counter({f"{m}:{r}": n for (m, r), n in skipped.items()})
    for child in children:
        parent_id = child.source_id.rsplit("#insufficient:", 1)[0]
        pooled = PoolCase(child, parents[(child.source, parent_id)].entry, False)
        if evidence_key(pooled) in taken:
            counts[f"{method(pooled)}:collides_with_candidate"] += 1
            continue
        out.append(pooled)
    return out, counts


def method(pooled: PoolCase) -> str:
    """Origine di un INSUFFICIENT: metodo di generazione, o `native:<voce>` se dalla fonte."""
    if "#insufficient:" in pooled.case.source_id:
        return pooled.case.source_id.rsplit("#insufficient:", 1)[1]
    return f"native:{pooled.entry}"
