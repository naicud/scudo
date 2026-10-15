# Harness di eval: metriche e baseline

Un solo harness misura il generalista e le **baseline** (`CONTEXT.md`) sugli
stessi casi. Fonte: `src/scudo/eval_report.py`, `src/scudo/baselines.py`; test
in `tests/test_eval_report.py` e `tests/test_baselines.py`. Il formato del
prompt di Scudo è in [`protocol.md`](protocol.md).

## Report

Un report per modello e per split. Intestazione:

| Campo | Contenuto |
| --- | --- |
| `proof_type` | `local` (checkpoint in training), `live` (GGUF nel backend nativo), `reference` (pesi pubblicati nel runtime di riferimento della card, es. transformers su Colab) |
| `model` | nome del modello o chiave del registro delle baseline |
| `weights` | identità del manifest (`repo`, `revision`, `filename`, `sha256`) di ogni file che determina i readout: shard, adapter, GGUF |
| `scudo_sha` | commit completo di `naicud/scudo` che ha prodotto il report |
| `runtime` | `name` e `version` del runtime (`transformers`, `llama.cpp`, `unsloth`, …) |
| `format` | `scudo.decision-token/1` per i nostri modelli; per una baseline il `prompt_format` del registro |
| `split` | benchmark o split valutato |
| `harness` | id dell'harness che ha prodotto i readout |

Una baseline eseguita con transformers su Colab è una prova `reference`: non è
un checkpoint in training (`local`) né il GGUF nel backend nativo (`live`). La
prova `live` richiede che tutti i `weights` siano `.gguf`.

Le metriche stanno in `overall`, `by_language` (`it`, `en`, `mixed`) e
`by_surface` (`prompt`, `response`, `injection_direct`, `injection_indirect`,
`over_refusal`). Ogni gruppo ha gli stessi campi.

## Metriche

Ogni risultato è un `EvalResult(case_id, language, surface, gold, readout,
error)`: un readout oppure un errore, mai entrambi. Un errore (timeout, uscita
illeggibile, readout malformato) entra nella colonna `ERROR` della matrice di
confusione e conta come sbaglio; non diventa mai `NO_VIOLATION`.

- **`accuracy`**: verdetti uguali al gold diviso tutti i casi, errori inclusi.
- **`precision`, `recall`, `f1`** per verdetto: precision = corretti /
  predetti con quel verdetto; recall = corretti / gold con quel verdetto (gli
  errori sono nel denominatore); F1 = 2PR/(P+R), `0.0` se P+R = 0. Valgono
  `null` quando un denominatore è zero.
- **`confusion`**: gold × predetto, con la colonna `ERROR`.
- **`calibration`**: calcolata solo sui readout con probabilità sui tre
  verdetti; `null` se nessuno le ha. `cases` dice quanti readout entrano.
  - *Confidenza*: probabilità assegnata al verdetto predetto.
  - *ECE top-label*: 10 bin di uguale ampiezza su [0, 1] (`[0, 0.1)`, …,
    `[0.9, 1]`); ECE = Σ_b |B_b|/n · |accuratezza(B_b) − confidenza media(B_b)|.
  - *Brier multiclasse*: media su casi di Σ_k (p_k − y_k)² sui tre verdetti,
    con y one-hot del gold; intervallo [0, 2].
  - Gli errori non hanno probabilità e restano fuori dalla calibrazione, ma
    restano nell'accuratezza.
- **`over_refusal`**: sui casi della superficie `over_refusal` con gold
  `NO_VIOLATION` (richieste benigne che sembrano dannose). `rate` = casi
  predetti `VIOLATION` / casi con un readout; `insufficient_rate` = casi
  `INSUFFICIENT` / casi con un readout; `errors` conta a parte i casi falliti.
  `null` se il gruppo non ha casi idonei; `rate` è `null` se tutti i casi sono
  falliti.

Il confronto fra checkpoint e GGUF (`compare_readouts`) è descritto in
[`protocol.md`](protocol.md#report-di-eval).

## Baseline

Repository, revisione e licenza letti il 15 ottobre 2026 dall'API Hugging Face
(`/api/models/<id>` e `/api/models/<id>/revision/main`, `cardData.license`,
file `LICENSE` dove presente) e dalla model card alla revisione fissata. Tutte
sono **Apache-2.0** e non gated. Sono baseline di sola eval: la licenza dei
loro dati di training non vincola Scudo, ma è registrata.

| Chiave | Repository @ revisione | Criterio Scudo | Uscita | Mappatura tri-state | Logit |
| --- | --- | --- | --- | --- | --- |
| `granite-4.1-3b-policy-guardrails` | `ibm-granite/granite-4.1-3b` @ `c0650403e44e78ec0262dab1c90914c65b196c4e` + adapter `ibm-granite/granitelib-guardian-r1.0` @ `1a1a488a51d0d5307e30e5b3c1e66b4a5237d18a`, `policy-guardrails/granite-4.1-3b/lora` | policy: `The scenario must not meet the following criterion. <criterio>` | JSON generato `{"score": "Yes"\|"No"\|"Ambiguous"}` | `No` → VIOLATION, `Yes` → NO_VIOLATION, `Ambiguous` → INSUFFICIENT | lettura Scudo, tre etichette |
| `rizzo-flow-4b` | `rizzoaiacademy/rizzo-flow` @ `55633c8cbd2b826bd3eefdeb05310450996649df` | istruzione di una domanda `boolean` con astensione | logit delle lettere A, B, C all'ultima posizione, nessun testo | A `false` → NO_VIOLATION, B `true` → VIOLATION, C astensione → INSUFFICIENT | sì, nativi |
| `apus-reflex-4b` | `apus-ailab/APUS-Reflex-4B` @ `f0b880feb751a789b4c6a062d03fbbc23d384991` | `instructions` di una richiesta `choice` con i tre verdetti come candidati | logit e probabilità per candidato (`OpenJet.decide`) | candidato = verdetto | sì, nativi |
| `granite-guardian-4.1-8b` | `ibm-granite/granite-guardian-4.1-8b` @ `ab01ccca5dcfb80246369a086a4a87a29198f5af` | criterio verbatim (Bring Your Own Criteria), modalità no-think | `<score>yes\|no</score>` generato | `yes` → VIOLATION, `no` → NO_VIOLATION; INSUFFICIENT non esprimibile | lettura Scudo, binario |
| `qwen3guard-gen-4b` | `Qwen/Qwen3Guard-Gen-4B` @ `6ec42827da0c1ff11e7a49dc269d2e810d27e108` | **non entra**: tassonomia fissa del chat template | `Safety: Safe\|Unsafe\|Controversial` generato | `Unsafe` → VIOLATION, `Safe` → NO_VIOLATION, `Controversial` → INSUFFICIENT | lettura Scudo, tre etichette |

`build_input(nome, criterio, evidenza, superficie)` costruisce l'ingresso del
caso; `parse(nome, text=..., label_logits=...)` produce un `LabelReadout`
(etichetta grezza, verdetto, probabilità, margine). L'evidenza va in un
messaggio `user`, oppure `assistant` sulla superficie `response`; per Rizzo e
APUS è lo stato/evidenza della loro richiesta.

### Logit e calibrazione

- **Lettori di opzioni** (Rizzo, APUS): i logit delle lettere sono l'uscita
  documentata. La softmax sui tre logit dà le probabilità; il margine è la
  distanza fra i due logit più alti. Il verdetto è l'argmax: le soglie di
  Rizzo (`max_unavailable_probability`, `min_top_probability`) non si
  applicano. Le due card dichiarano probabilità non calibrate.
- **Baseline generative** (Granite policy, Guardian, Qwen3Guard): la card
  documenta solo il testo. Il runner può leggere, nel passo di generazione
  greedy che emette l'etichetta, i logit del primo token di ogni etichetta:
  è una lettura di Scudo, non dichiarata dalle card. Si usa solo se i primi
  token delle etichette sono distinti (il runner lo verifica sul tokenizer
  della revisione fissata); altrimenti nessun logit e `calibration` resta
  `null` per quel modello. Il verdetto viene sempre dal testo: se l'argmax dei
  logit non coincide con l'etichetta generata, l'esito è l'errore
  `LOGIT_TEXT_MISMATCH`.
- **Binario** (Guardian): la probabilità di INSUFFICIENT è 0, perché il
  modello non può esprimerlo; Brier ed ECE lo contano così.

Errori di `parse`, tutti `BaselineError` con `code`: `OUTPUT_UNPARSEABLE`,
`LABEL_UNKNOWN`, `LOGITS_INVALID`, `LOGIT_TEXT_MISMATCH`, `UNKNOWN_BASELINE`;
`build_input` dà `UNSUPPORTED` per una superficie che la baseline non può
vedere. Il runner registra il `code` come errore del risultato.

### Note per baseline

**Granite 4.1 3B + policy-guardrails.** Prompt, guidelines e parser dalla card
`policy-guardrails/README.md` ("trained using the prompt template below"): due
messaggi, lo scenario e il blocco `<guardian> … ### Criteria: Policy: … ###
Scoring Schema: …`, generazione greedy, `max_new_tokens=20`. La traduzione del
criterio in policy è **nostra**, non della card. `Ambiguous` è documentato come
"non è possibile decidere … serve più informazione", la stessa semantica di
INSUFFICIENT. Incoerenza della fonte: l'`io.yaml` dell'adapter chiede nel testo
la chiave `label`, mentre card, schema JSON dell'`io.yaml` e prompt della card
usano `score`; il parser accetta solo `{"score": …}`. Dataset di training
proprietario IBM; card con tag `en`, nessuna valutazione IT; la card avverte
che usi diversi da policy-scenario non sono stati addestrati né testati.

**Rizzo Flow 4B.** La card fissa il prompt `spark-decisions-v3` al commit
`c30cc633f5bb0cde920e0998fad4f617ed138f44` di
`Rizzo-AI-Academy/rizzo-flow` (Apache-2.0): system prompt, evidenza fra tag
`<evidence>`, domanda a scelta multipla, `enable_thinking=False`, da
`src/rizzo_flow/prompts.py`; candidati di una domanda `boolean` con i default
e `allow_abstain` da `decisions.py` e `schema.py`. I byte del nostro
`build_input` sono stati confrontati con quelle funzioni. Pesi fusi alla radice
del repository (4B); il tag `base_model` elenca anche la variante 1.7B, che sta
in `lora-1.7b/`. Il caricamento transformers richiede `trust_remote_code=True`
(`modeling_spark.py`), quindi solo alla revisione fissata. Dati di training con
config CC-BY-SA (`stsb`) e CC-BY-4.0, dichiarate dalla card.

**APUS-Reflex-4B.** Il nome storico `apus-ailab/APUS-OpenJev-v1-4B` reindirizza
a questo repository (stesso SHA). Prompt `jev.dynamic.prompt.v2` da
`openjet_runtime/contracts.py`; `OpenJet.decide(request, effort="high")`
restituisce logit e probabilità per candidato. I candidati sono i tre verdetti
con le descrizioni di `SEMANTIC_DESCRIPTIONS`, nell'ordine canonico fisso. Il
runtime della card richiede esattamente `transformers==5.16.1`, diverso
dall'extra `train` (`>=5.17,<5.18`): va eseguito in un ambiente separato. Dati
di training non dichiarati (ADR 0002).

**Granite Guardian 4.1 8B.** Blocco no-think `<guardian><no-think>…` più
`### Criteria:` e `### Scoring Schema:` per utente o assistente, parser della
card. Il codice d'esempio della card omette `<guardian>`, che compare invece
nella descrizione del formato della card e nel cookbook ufficiale
(`ibm-granite/granite-guardian` @ `9698a42800e9e6e840f908eecc7c730865dd2176`,
`cookbooks/granite-guardian-4.1/quickstart.ipynb`): usiamo `<guardian>`. La
card dichiara training e test solo in inglese. Uscita binaria: INSUFFICIENT non
è mai predetto, quindi la sua recall è 0 dove il gold lo contiene.

**Qwen3Guard-Gen-4B.** Il chat template della revisione fissata contiene la
policy e la tassonomia (9 categorie, `Jailbreak` solo per l'input): il criterio
di Scudo non entra, quindi il confronto ha senso solo per criteri allineati
alla sicurezza dei contenuti. `Controversial` è definito come contenuto non
esplicitamente vietato ma il cui rischio dipende dal contesto: lo mappiamo a
INSUFFICIENT come approssimazione, e l'etichetta grezza resta nel
`LabelReadout` per ricalcolare le varianti strict (→ VIOLATION) o loose (→
NO_VIOLATION). La superficie `response` è **UNSUPPORTED**: il template ignora
un primo messaggio `assistant`, e i casi `response` di Scudo non portano il
prompt che ha prodotto la risposta. La card dichiara 119 lingue.

## Fuori da questa slice

- Loader dei benchmark verso `Case` e notebook Colab dell'eval: wave
  successiva, sopra manifest, registro dei benchmark e deduplica.
- Verifica che i benchmark di eval siano esclusi dal training: dalla deduplica
  dell'ingestione.
- Runner delle baseline: carica i pesi alla revisione fissata, ne verifica i
  digest (`weights`), applica il chat template, legge testo e logit.
