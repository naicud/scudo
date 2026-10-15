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

Una voce con `role = "excluded"` resta nel manifest come documentazione (data
card e motivo dell'esclusione) e non entra né nel training né nel registro dei
benchmark. Non blocca la build, ma richiede la stessa tabella `decision`
completa. Al 2026-10-15 sono escluse `nemotron-sft-safety-v2` (termini Riva
`NO`), `deepset-prompt-injections`, `gandalf-ignore-instructions`,
`yanismiraoui-prompt-injections` (provenienza non verificabile o termini
OpenAI) e `wildguardtest` (accesso gated non richiesto); `nemotron3-train`
resta nel training con decisione e rischio registrati.

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
| `open-jev-train`, `canttalk-train`, `procedural-train` | `open_jev`, `canttalk`, `procedural` | vedi [criteri personalizzati](criteri-personalizzati.md); `open-jev-train` passa il cancello (generatori procedurali, nessun LLM) | vedi il documento |

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

Il costruttore del mix (`scudo.mix_cli`) toglie duplicati e conflitti e accetta i
casi `needs_relabel` solo con la conferma della giuria di teacher: vedi
[`fast-v01.md`](fast-v01.md).

## Ragionamenti dell'Accurate

`Case` non porta un ragionamento: le fonti con ragionamenti seguono un
percorso parallelo, `scudo.reasoning_data` (in `report.json` dell'ingestione
compaiono in `not_ingested` con il comando da usare).

```sh
rtk proxy uv run --extra train python -m scudo.reasoning_data build --out /percorso/autorizzato [--extra teacher.jsonl ...]
```

Stesso cancello del manifest, stessa cartella fuori dal repository, stesso
indice dei benchmark. Da `nvidia/Nemotron-Content-Safety-Reasoning-Dataset`
(revisione `792b0715f519c0750d63b73af2bf33ddd9ac3887`) legge il file
`aegis_v2_efficient_reasoning.jsonl` dello split
`nemotron_content_safety_reasoning`, con digest SHA-256 verificato. Schema
(20 colonne, letto da README e datasets-server): le 9 colonne di Aegis 2.0,
`prompt_harm_label` e `response_harm_label` (`harmful`/`unharmful`, identiche
alle label), e per ogni generatore (`deepseek_r1_0528`, `gpt_oss_120b`,
`qwen3_32b`) `safety_reasoning_*` (traccia completa), `efficient_reasoning_*`
(ridotta) e `prediction_gt_mismatch_*`.

Regole:

- restano fuori `refusal_data_augmentation`, i prompt `REDACTED` e le righe con
  una risposta: la traccia giudica insieme prompt e risposta, mentre
  l'evidenza della superficie `response` è la sola risposta. Resta la
  superficie `prompt` (label umane), criterio `HARMFUL_REQUEST`;
- la traccia è `<think>…</think>` seguita da `Prompt harm: harmful|unharmful`
  e `Response harm: …`; il verdetto della traccia è la riga `Prompt harm`
  (`harmful` → VIOLATION). Il ragionamento è il testo dentro `<think>`. Una
  traccia senza questa forma o con ragionamento vuoto esce con il motivo in
  `trace_dropped`;
- accordo con la label: una traccia il cui verdetto non coincide con il gold
  esce e si conta in `disagree`. Il campo `prediction_gt_mismatch_*` non si
  usa: riguarda l'uscita congiunta prompt+risposta;
- si usano le tracce complete: le `efficient_reasoning_*` spesso non hanno il
  verdetto finale, quindi non si può verificare l'accordo.

Ogni riga di `reasoning.jsonl` è un `ReasoningRecord` (`source`, `source_id`,
`language`, `surface`, `criterion`, `evidence`, `verdict`, `reasoning`,
`trace_verdict`, `generator`) più `format`, `prompt`, `completion` e
`semantic_ids` del formato reasoned-decision. I file `--extra` (ragionamenti
italiani del teacher, issue #7) contengono esattamente i campi di
`ReasoningRecord`: fonte, id e generatore sono obbligatori; un record non
valido ferma la build con file e riga; il disaccordo con la label e i
duplicati dei benchmark escono contati, come per Nemotron.
`reasoning-report.json` riporta per fonte righe, scarti per motivo, tracce,
disaccordi, rimossi per benchmark e tenuti per generatore e verdetto; ha anche
`manifest_sha256`, `dedup` e `benchmarks` come il report di ingestione, quindi
`python -m scudo.ingest verify --report reasoning-report.json` ne verifica la
deduplica contro tutto il registro.

Smoke sul file reale (fuori dal repository, senza deduplica): 27.459 righe;
escluse 9.431 con risposta e 3.363 `refusal_data_augmentation`; 43.281 tracce
leggibili, 714 scartate (708 vuote, 4 con verdetto illeggibile, 2 senza
`</think>`), 2.742 in disaccordo; restano 40.539 ragionamenti prima della
deduplica.

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
cui partono i convertitori dell'eval (`scudo.benchmarks`, vedi
[`eval.md`](eval.md#benchmark)).

`python -m scudo.ingest verify --report <out>/report.json` verifica che una
build abbia deduplicato contro **tutti** i benchmark del registro, con lo
stesso manifest e gli stessi parametri: un benchmark mancante o senza testi
indicizzati dà FAIL (uscita 1).
