# Istruzioni per gli agenti — Scudo

Prefissa i comandi shell con `rtk` (es. `rtk git status`, `rtk proxy uv ...`).
Se `rtk` altera un comando, eseguilo senza prefisso e segnalalo.

## Agent skills

### Issue tracker

L'unico tracker è GitHub Issues in `naicud/scudo`, gestito con `gh`. Prima di
leggere, scegliere o modificare ticket, leggi
[issue-tracker.md](docs/agents/issue-tracker.md).

### Triage labels

La triage usa cinque ruoli canonici; leggi la mappatura in
[triage-labels.md](docs/agents/triage-labels.md) prima di applicare label.

### Domain docs

Il layout è single-context; leggi [domain.md](docs/agents/domain.md) prima di
esplorare il dominio, nominare concetti o proporre decisioni architetturali.

## Confini del sensore

Scudo è un sensore semantico: restituisce `VIOLATION`, `NO_VIOLATION` o
`INSUFFICIENT` per un criterio. Non decide policy, non concede permessi e non
esegue tool; policy engine e gateway vivono in un progetto separato. Un
timeout, un errore o un output malformato non diventano mai `NO_VIOLATION`.
La v1 copre prompt utente, risposta del modello, prompt injection diretta e
indiretta, over-refusal; tool/agent e PII restano alla v2. Segnala
`UNSUPPORTED` per capability non coperte.

## Dati e licenze

Entrano nel training solo dati con licenza Apache-2.0, MIT, CC0, CC-BY o
ODC-BY. Restano fuori NC, CC-BY-SA, gated con termini propri e dati generati o
distillati da modelli la cui licenza limita uso o training. Ogni dataset ha
una data card completa ([template](docs/templates/data-card.md)) prima di
entrare nel manifest. I benchmark di eval si deduplicano contro il training.

Dati, pesi, checkpoint, report sensibili e token non entrano mai nel
repository: restano su Hugging Face privato o nel perimetro autorizzato.
`scripts/check_no_artifacts.py` lo controlla prima di ogni commit.

## Dipendenze e codice

Usa `uv` per Python. Per API e versioni delle librerie consulta la
documentazione ufficiale aggiornata (Context7), non la memoria; preferisci
l'ultima versione stabile compatibile e verifica manifest e lock reali.
Il pacchetto base non ha dipendenze runtime: lo stack pesante (Unsloth, torch,
transformers, datasets, trl, peft) sta solo nell'extra `train`, e l'import di
`scudo` non lo carica.

Punta a 650 righe per file di codice, massimo 700. Corpus, lock e documenti non
sono codice. Non creare struttura per anticipazione: aggiungi moduli quando
una slice li usa.

## Verifica

Prima del commit:

```sh
rtk proxy uv lock --check
rtk proxy uv run pytest
rtk proxy uv run --no-sync python scripts/check_no_artifacts.py
```

Riporta SHA, comandi ed esiti. Distingui smoke locale, eval e training reale su
Colab; non dichiarare PASS per prove saltate o basate solo su fixture. Una
metrica di un modello vale solo con revisione, hash dei pesi e harness
dichiarati.
