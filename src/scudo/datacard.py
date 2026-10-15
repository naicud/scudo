"""Data card e attribuzioni generate dal manifest, con i conteggi di una build.

La card segue `docs/templates/data-card.md`; i conteggi (righe, scarti per
motivo, rimozioni per benchmark) vengono dal report di `scudo.ingest`. Senza
build restano «pendente». Le attribuzioni coprono le fonti di training CC-BY e
ODC-BY, nel formato da riportare anche nella model card.
"""

from scudo.dedup import BANDS, BINS, ROWS, SHINGLE, THRESHOLD
from scudo.manifest import Entry

LICENSE_URLS = {
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "odc-by": "https://opendatacommons.org/licenses/by/1-0/",
}
_PENDING = "pendente (nessuna build registrata)"


def _items(values, indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {value}" for value in values)


def _counts(counts: dict) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "nessuna"


def _attribution(entry: Entry) -> str:
    return (
        f"- Titolo: {entry.title}\n"
        f"- Autori: {entry.authors}\n"
        f"- URL: {entry.url} (revisione `{entry.revision}`)\n"
        f"- Licenza: {entry.license}, {LICENSE_URLS.get(entry.license, 'URL non registrato')}\n"
        f"- Modifiche apportate: {entry.changes}"
    )


def render(entry: Entry, stats: dict | None = None) -> str:
    """Data card di una voce; `stats` è la voce della fonte nel report di build."""
    decision = entry.decision
    decided = (f"{decision.by}, {decision.on}: {decision.rationale}" if decision
               else "nessuna")
    if stats:
        rows = (f"{stats['rows']} lette; {stats['cases']} casi; {stats['kept']} tenuti dopo la "
                f"deduplica; {stats['needs_relabel']} da rietichettare (llm_jury)")
        dropped = _counts(stats["dropped"])
        removed = (f"esatte {_counts(stats['removed_exact'])}; "
                   f"near-duplicate {_counts(stats['removed_near'])}")
    else:
        rows = dropped = removed = _PENDING
    return f"""# Data card: `{entry.id}` ({entry.name})

Generata da `src/scudo/manifest.toml`; ruolo `{entry.role}`.

## Fonte

- Id e URL: `{entry.id}`, {entry.url}
- Revisione (commit o versione) letta: `{entry.revision}`
- Data di lettura: {entry.read_on}
- Config e split usati: config `{entry.config}`, split {", ".join(entry.splits)}
- Righe usate, per split e lingua: lingue {", ".join(entry.languages)}; {rows}

## Licenza

- Licenza dichiarata: `{entry.license}`
- Fonte della verifica: {entry.license_source}
- Gated: {"sì" if entry.gated else "no"}

## Generatore e teacher

- Testi scritti da: {entry.texts_by}
- Label prodotte da: {entry.labels_by}
- Generatori e licenze:
{_items(entry.generators)}
- Teacher usati da noi: nessuno in ingestione; le label `llm_jury` sono marcate per la
  giuria di teacher.
- Termini d'uso dei generatori compatibili con il training: {entry.generator_terms}.
  {entry.generator_terms_note}
- Decisione umana: {decided}

## Trasformazioni e filtri

- Superfici: {", ".join(entry.surfaces)}
- Filtri:
{_items(entry.filters)}
- Righe scartate e motivo: {dropped}

## Deduplicazione contro i benchmark

- Benchmark controllati: tutte le voci `benchmark` del manifest
- Metodo: SHA-256 del testo normalizzato; near-duplicate MinHash su shingle di {SHINGLE}
  caratteri, {BINS} bin, LSH {BANDS}×{ROWS}, Jaccard stimata >= {THRESHOLD}
- Righe rimosse: {removed}

## Attribuzioni

{_attribution(entry)}

## Rischi noti

{_items(entry.risks, indent="")}
"""


def attributions(entries: list[Entry]) -> str:
    """Attribuzioni delle fonti di training CC-BY e ODC-BY."""
    used = [e for e in entries if e.role == "train" and e.license in LICENSE_URLS]
    body = "\n\n".join(f"### {entry.title}\n\n{_attribution(entry)}" for entry in used)
    return f"# Attribuzioni dei dati di training\n\n{body}\n"
