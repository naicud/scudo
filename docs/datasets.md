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
  `needs_relabel`. `needs_relabel` è vero quando la label non è gold (giuria
  `llm_jury`, label prodotte da modelli, negativi per assenza di tag): il caso
  non è gold finché la giuria di teacher non lo rietichetta;
- `report.json`: digest del manifest, parametri di deduplica, per fonte righe
  lette, scarti per motivo (righe intere o parti di riga: domande,
  distrattori, risposte), casi, rimossi per benchmark (esatti e near), tenuti e
  da rietichettare; per benchmark testi indicizzati e righe rimosse; le fonti
  senza convertitore in `not_ingested`, con il motivo se la fonte non ha una
  mappatura fedele (`scudo.ingest.UNCONVERTED`);
- `data-cards/`: una card per voce con i conteggi della build e
  `ATTRIBUTIONS.md` con le attribuzioni CC-BY e ODC-BY delle fonti usate.

## Convertitori delle fonti di training

Un convertitore (`scudo.ingest.CONVERTERS`, per id del dataset) espone
`needs_relabel(row, case)` e `drop_reason(row)` con `cases_from_row(row)`,
oppure `convert_row(row) -> (casi, scarti)` se scarta parti di riga. Ogni scarto
ha un motivo; un valore sconosciuto in una label è un errore, mai un
NO_VIOLATION. Lingua `en` per tutte: l'italiano arriva dalla traduzione (#7).
Criteri: `aegis.HARMFUL_REQUEST` e `HARMFUL_RESPONSE` per i danni,
`injection.direct` e `injection.indirect` di `scudo.default_criteria` per
l'injection. Schema letto da datasets-server e dai file alla revisione fissata;
smoke locale del 15 ottobre 2026 sugli stessi file.

| Voce | Modulo | Mappatura | Smoke locale |
| --- | --- | --- | --- |
| `aegis2-train`, `nemotron3-train` | `aegis`, `nemotron` | `prompt_label` / `response_label` safe/unsafe; fuori `refusal_data_augmentation` e `REDACTED`; `llm_jury` da rietichettare | — |
| `aegis1-train` | `aegis1` | solo `text_type = user_message` (label per dialogo, README); annotatori unanimi: tutti `Safe` NO_VIOLATION, tutti categorie di rischio (13, `Other` compresa) VIOLATION; fuori `Needs Caution`, disaccordo, annotazioni mancanti, label fuori tassonomia | 10.798 righe: 1.320 VIOLATION, 967 NO_VIOLATION; scarti 7.699 dialogo, 746 Needs Caution, 43 disaccordo, 18 annotazioni mancanti, 3 testo vuoto, 2 fuori tassonomia |
| `aya-redteaming-en` | `aya` | split `english`; ogni prompt di red-teaming VIOLATION; solo positivi | 987 VIOLATION |
| `gretel-safety-alignment` | `gretel` | prompt VIOLATION; `response` VIOLATION e `safe_response` NO_VIOLATION se Granite Guardian concorda (> / < 0,5); punteggi del giudice non nominato non usati; tutto `needs_relabel` | 5.997 righe: 5.997 prompt, 5.997 + 5.997 risposte, nessuno scarto |
| `llmail-inject` | `llmail` | `injection_indirect` VIOLATION solo con `exfil.sent` vero sui livelli Phi-3; mai NO_VIOLATION: un invio fallito è comunque un attacco; solo positivi | Phase2 90.916 righe: 4.589 casi; scarti 45.077 senza chiamata API, 41.142 vittima GPT-4o mini, 108 esito assente. Phase1 (primi 150 MB, 31.626 righe): 1.547 casi |
| `in-the-wild-jailbreak` | `in_the_wild` | `jailbreak` vero: `injection_direct` VIOLATION | 1.405 VIOLATION |
| `in-the-wild-regular` | `in_the_wild` | `jailbreak` falso: NO_VIOLATION, `needs_relabel` (negativo per assenza del tag, spesso role-play vicino ai jailbreak) | 13.735 NO_VIOLATION da rietichettare |
| `jackhhao-jailbreak` | `jackhhao` | solo `type = jailbreak`: VIOLATION; benigni OpenOrca/GPTeacher fuori | 1.044 righe: 527 VIOLATION, 517 benigni scartati |
| `open-jev-train`, `canttalk-train`, `procedural-train` | `open_jev`, `canttalk`, `procedural` | vedi [criteri personalizzati](criteri-personalizzati.md); `open-jev-train` è bloccato dal cancello (teacher non dichiarato) | vedi il documento |

Non convertite, con il motivo in `not_ingested`: `civil-comments` (punteggi di
tossicità, non label di policy; domanda ai valutatori e soglia non nel README,
numero di valutatori assente: una soglia sarebbe una label inventata) e
`mosscap-prompt-injection` (il README avverte che molti prompt non sono
injection; nessuna label per riga). `nemotron-content-safety-reasoning` è
dell'Accurate (#9).

Per il costruttore del mix:

- Aya, LLMail, jailbreak in-the-wild e jackhhao sono solo positivi: i negativi
  vengono da altre fonti (Aegis, in-the-wild-regular dopo la rietichettatura);
- la deduplica qui è solo contro i benchmark: restano da togliere i duplicati
  dentro la fonte (LLMail ripete lo stesso invio su più livelli, Aegis 1.0 ha
  righe ripetute) e tra fonti (Aegis 1.0 e 2.0 da HH-RLHF; jackhhao e
  in-the-wild da `verazuo/jailbreak_llms`);
- conflitti di label sullo stesso testo: 39 testi sono sia jailbreak sia
  regular in in-the-wild, 12 jailbreak di jackhhao sono regular in in-the-wild.

## Deduplica contro i benchmark

`scudo.dedup` (solo standard library, deterministico) indicizza i campi di
testo di tutti i benchmark del manifest e, se il benchmark ha un convertitore,
le evidenze dei suoi casi nella stessa forma del training (per CantTalkAboutThis
il dialogo è una lista e solo così entra nell'indice); poi confronta l'evidenza
di ogni caso:

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
