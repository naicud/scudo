# Manifest, cancello e deduplica dei dataset

Il manifest dei dataset è [`src/scudo/manifest.toml`](../src/scudo/manifest.toml):
solo metadati, mai dati. Ogni voce è una fonte di training (`role = "train"`) o
un benchmark held-out (`role = "benchmark"`), con revisione fissata a un commit,
licenza, gating, chi ha scritto testi e label, generatori con licenza, termini
dei generatori (`SÌ` / `NO` / `INCERTO`), filtri, attribuzione e rischi. Sono
gli stessi campi della [data card](templates/data-card.md): la card si genera
dal manifest.

## Cancello di licenza e provenienza

`scudo.manifest.gate` blocca la build se una voce non è ammessa (regole
dell'[ADR 0002](adr/0002-generalisti-pubblici-apache-addestrati-su-colab.md)):

- revisione diversa da un commit di 40 caratteri (anche `NON VERIFICATO`);
- licenza fuori da Apache-2.0, MIT, CC0, CC-BY, ODC-BY; un benchmark ammette
  anche CC-BY-NC perché resta solo valutazione, mai training;
- campi della data card vuoti, superfici fuori dalla v1;
- training: fonte gated, termini dei generatori `NO`, oppure `INCERTO` senza
  una tabella `decision` con `by`, `on` e `rationale`. La decisione la scrive
  una persona responsabile, mai il codice o un agente;
- benchmark senza campi di testo verificati su cui deduplicare;
- uno split held-out usato anche per il training; nomi duplicati.

`python -m scudo.ingest check` stampa i motivi di blocco ed esce con 1.

## Build

```sh
rtk proxy uv run --extra train python -m scudo.ingest build --out /percorso/autorizzato
```

La build passa il cancello prima di scaricare, rifiuta una cartella di uscita
dentro il repository o dentro un work tree git, scarica alle revisioni fissate
con `datasets.load_dataset` (file grezzi in `<out>/hf-cache/hub` via
`HF_HUB_CACHE`, tabelle Arrow in `<out>/hf-cache/datasets`; se la cache Hub è
già fissata altrove la build si ferma) e scrive:

- `<nome>.jsonl`: i casi (`scudo.cases.Case`) tenuti, con `split` e
  `needs_relabel`. `needs_relabel` è vero quando la label viene da `llm_jury`:
  il caso non è gold finché la giuria di teacher non lo rietichetta;
- `report.json`: digest del manifest, parametri di deduplica, per fonte righe
  lette, scarti per motivo, casi, rimossi per benchmark (esatti e near), tenuti e
  da rietichettare; per benchmark testi indicizzati e righe rimosse; le fonti
  senza convertitore in `not_ingested`;
- `data-cards/`: una card per voce con i conteggi della build e
  `ATTRIBUTIONS.md` con le attribuzioni CC-BY e ODC-BY delle fonti usate.

Oggi hanno un convertitore Aegis 2.0 (`scudo.aegis`) e Nemotron v3
(`scudo.nemotron`). Entrambi escludono le righe `refusal_data_augmentation` e i
prompt `REDACTED`, con il motivo nel report.

## Deduplica contro i benchmark

`scudo.dedup` (solo standard library, deterministico) indicizza i campi di
testo di tutti i benchmark del manifest e confronta l'evidenza di ogni caso:

- normalizzazione: NFKC, casefold, punteggiatura in spazi, spazi compressi;
- esatta: SHA-256 del testo normalizzato;
- near-duplicate: MinHash a una permutazione su shingle di 5 caratteri, 128 bin
  con densificazione, LSH a 25 bande da 5 righe; duplicato se la Jaccard
  stimata è almeno 0,8 (errore standard circa 0,035 a quella soglia).

Un caso che combacia con più benchmark si conta sul primo nell'ordine del
manifest.

## Registro dei benchmark

`scudo.manifest.benchmarks()` restituisce le voci held-out (`name`, `id`,
`config`, `splits`, `revision`, `license`, `text_fields`, ...): è il punto da
cui partono i convertitori dell'eval.
