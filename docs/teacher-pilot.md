# Pilot dei teacher (issue #5)

Il pilot confronta due **teacher** con licenza aperta su casi con label umana
nota, misura accordo, rifiuti, output malformati e costo per item, valida la
rietichettatura delle label `llm_jury` e stima la spesa del volume v0.1. Le
uscite dei teacher sono candidate, mai gold. Tetto complessivo v0.1: 150 USD.
Dopo il pilot ci si ferma: nessuna generazione in volume senza la conferma
registrata della persona.

Il codice non spende nulla da solo: ogni run richiede la conferma esplicita
dell'importo stimato. Provider, endpoint, chiave, prezzi e termini del provider
li fornisce la persona; senza ciascuno di essi il pilot rifiuta di partire.

Moduli: `scudo.pilot_sample` (campione), `scudo.teacher` (registro,
configurazione, messaggi, formato), `scudo.teacher_client` (client HTTP),
`scudo.budget` (stima, registro della spesa, stop), `scudo.pilot_report` e
`scudo.pilot_summary` (report), `scudo.pilot` (comandi).

## Teacher candidati

Letti il 2026-10-15 da `https://huggingface.co/api/models/<id>` (`sha`,
`cardData.license`, `gated`) e dai file `LICENSE`/`USAGE_POLICY` alla revisione:

| id | revisione | licenza | note |
|---|---|---|---|
| `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | Apache-2.0, non gated | `USAGE_POLICY`: rispetto della legge applicabile; ragionamento low/medium/high |
| `Qwen/Qwen3.5-397B-A17B` | `8472618112abcbd45acbcdc58436aff4233c23f7` | Apache-2.0, non gated | il Qwen3.5 più grande pubblicato da Qwen sul Hub; thinking attivo di default |

La revisione servita dal provider è `NON VERIFICATO` finché il provider non la
dichiara (campo `served_revision`).

## Campione

`sample` sceglie, per hash seminato (deterministico, indipendente dall'ordine):

- **gold**, 500 casi con label umana (`<superficie>_label_source = human`),
  quote uguali per lingua × superficie × verdetto; la quota che uno strato non
  copre passa agli altri, e se mancano casi il comando fallisce:
  - inglese da `aegis2-train` (prompt e risposte; ammesso dal cancello);
  - italiano da `nemotron3-test`, benchmark held-out, **solo valutazione**:
    `nemotron3-train` è bloccato dal cancello (termini dei generatori `INCERTO`
    senza decisione umana), quindi Nemotron v3 non dà dati di training. I casi
    di benchmark hanno `training_eligible = false` e le uscite del teacher su di
    essi non entrano mai nel training. L'italiano di Nemotron v3 è tradotto a
    macchina (data card);
  - misto IT/EN: nessuna fonte, lo strato manca e il report lo dice;
  - injection e over-refusal: nessun convertitore oggi, fuori dal pilot;
- **relabel**, 200 risposte di `aegis2-train` con label `llm_jury`, quote
  uguali per verdetto della giuria.

Ogni fonte passa il cancello del manifest nel proprio ruolo; una fonte bloccata
non si legge. `sample-meta.json` registra strati, fonti bloccate e tutti i
blocchi del cancello. Le righe `refusal_data_augmentation` e i prompt
`REDACTED` restano fuori come in `scudo.aegis`.

### Validazione della rietichettatura `llm_jury`

Nello schema di Aegis 2.0 ogni label ha una sola fonte: le righe `llm_jury` non
hanno una label umana (train, 2026-10-15: `prompt_label_source` sempre `human`;
`response_label_source` human 4.682, `llm_jury` 5.552,
`refusal_data_augmentation` 5.000). Quindi il report misura:

1. accordo di ciascun teacher con la giuria originale;
2. la **giuria di teacher** (tutti i teacher `ok` e concordi): copertura e
   accordo con la giuria originale;
3. come ancora, l'accordo di ciascun teacher sulle risposte gold (label umane,
   stessa superficie e fonte);
4. uno **spot-check cieco**: `relabel-spotcheck.csv` (50 righe, strati
   «giuria di teacher concorde / discorde con la giuria originale / irrisolto»,
   ordine mescolato, nessuna label visibile). La persona compila
   `human_verdict` (`VIOLATION`, `NO_VIOLATION`, `INSUFFICIENT`) e rilancia
   `report --spotcheck FILE`: il report misura giuria originale, giuria di
   teacher e singoli teacher contro le label umane. Finché manca, la
   validazione resta «in attesa».

## Cosa fornisce la persona

Un file TOML **fuori dal repository**, un `[[teacher]]` per candidato:

```toml
[[teacher]]
model_id = "openai/gpt-oss-120b"        # id del registro
provider = "<nome del provider>"
provider_model = "<nome del modello presso il provider>"
base_url = "https://<endpoint compatibile OpenAI>/v1"
api_key_env = "<NOME_VARIABILE>"          # la chiave sta nella variabile, mai nel file
terms_url = "<URL dei termini del provider>"
terms_checked_by = "<chi li ha letti>"
terms_checked_on = "<AAAA-MM-GG>"
terms_note = "<cosa dicono su uso degli output e training>"
input_usd_per_mtok = <prezzo di listino in ingresso, USD per milione di token>
output_usd_per_mtok = <prezzo di listino in uscita, USD per milione di token>
max_tokens = <tetto dei token generati, ragionamento incluso>
# facoltativi
served_revision = "<revisione se il provider la dichiara>"
timeout_s = 120.0
request_extra = { reasoning_effort = "medium", logprobs = true, top_logprobs = 5 }
```

`request_extra` passa al provider parametri propri così come sono. Con
`logprobs` e `top_logprobs` l'esito registra il margine del verdetto (`margin`,
in nat, `scudo.teacher_margin`), usato dal criterio di difficoltà
(`docs/fast-v01.md`). Documentazione letta il 2026-10-15: l'API chat completion
di Hugging Face Inference Providers accetta `logprobs` (booleano) e
`top_logprobs` (intero fra 0 e 5;
https://huggingface.co/docs/inference-providers/tasks/chat-completion); l'API
compatibile OpenAI di DeepInfra accetta `logprobs` e `top_logprobs` fra 1 e 20
(https://docs.deepinfra.com/chat/log-probs). Nessuna delle due pagine dice se
`openai/gpt-oss-120b` e `Qwen/Qwen3.5-397B-A17B` presso il provider li
restituiscono, né se i token di ragionamento compaiono nei logprob: **non
verificato**. Se il provider non li restituisce `margin` resta `null` e la
difficoltà usa solo il disaccordo; il primo esito del pilot lo mostra.

Tutti i campi senza commento «facoltativi» sono obbligatori; prezzi e
`max_tokens` devono essere positivi, `base_url` https (http solo per
localhost). Nessun prezzo è scritto nel codice. La chiave va esportata nella
variabile indicata solo nella shell della run.

## Passi

```sh
# 1. Campione (scarica alle revisioni del manifest, extra train)
rtk proxy uv run --extra train python -m scudo.pilot sample --out /perimetro/pilot

# 2. Prova a secco: nessuna chiamata, nessuna chiave
rtk proxy uv run python -m scudo.pilot estimate --out /perimetro/pilot \
  --teachers /perimetro/teachers.toml --ledger /perimetro/spesa-v0.1.jsonl

# 3. Run: --confirm-spend deve essere esattamente il valore `confirm` della stima
export <NOME_VARIABILE>=...
rtk proxy uv run python -m scudo.pilot run --out /perimetro/pilot \
  --teachers /perimetro/teachers.toml --ledger /perimetro/spesa-v0.1.jsonl \
  --confirm-spend <confirm>

# 4. Report (e, dopo lo spot-check, di nuovo con --spotcheck)
rtk proxy uv run python -m scudo.pilot report --out /perimetro/pilot \
  --teachers /perimetro/teachers.toml --ledger /perimetro/spesa-v0.1.jsonl \
  [--spotcheck /perimetro/relabel-spotcheck-compilato.csv]

# 5. Solo la persona, dopo aver scelto il teacher (o i teacher: --teacher ripetibile)
rtk proxy uv run python -m scudo.pilot confirm-volume --ledger /perimetro/spesa-v0.1.jsonl \
  --teacher <id> [--teacher <id2>] --amount <USD totali> --by "<nome>" --rationale "<perché>"
```

La stima è il caso peggiore: token d'ingresso stimati come caratteri/4 (una
stima, non il tokenizer) più `max_tokens` in uscita per ogni chiamata. `run`
rifiuta se la stima arrotondata al centesimo supera il residuo del tetto
(150 USD meno la spesa nel registro) o se `--confirm-spend` non coincide; poi
non parte una chiamata che potrebbe superare l'importo autorizzato, e si ferma.
Se l'ingresso reale supera la stima, lo sforamento è limitato a una chiamata.
Una run interrotta riprende saltando le coppie (teacher, caso) già fatte, con
una nuova stima e una nuova conferma. Le chiamate sono sequenziali.

`run` rifiuta anche un campione oltre 1.000 casi, o un campione diverso da
quello del pilot già riportato, finché il registro non ha una
`volume_confirmation`. Con una conferma nel registro ogni run è in volume e
passa dallo stesso controllo degli altri strumenti del volume
(`scudo.budget.check_volume`): vale solo l'ultima conferma, ogni teacher della
run deve essere nel suo insieme di teacher, e la stima della run (somma su
tutti i teacher) arrotondata al centesimo deve stare nel residuo del volume
confermato (importo dell'ultima conferma meno le chiamate registrate dopo) e
del tetto.

## Client

Chat completions compatibile OpenAI su `urllib`: timeout per chiamata, ritenta
solo 429 e 5xx (fino a 5 tentativi, backoff 2, 4, 8, 16 s), uso di token dalla
risposta (`prompt_tokens`, `completion_tokens`, `reasoning_tokens` se
presenti). Esiti:

- `ok`: JSON `{"verdict", "reasoning"}` esatto (tollerato un blocco ```json);
- `refused`: `finish_reason = content_filter`, campo `refusal`, o testo non
  JSON che apre con una formula di rifiuto (euristica dichiarata);
- `truncated`: `max_tokens` raggiunto;
- `malformed`, `timeout`, `error`.

Solo `ok` ha un verdetto; ogni altro esito conta come disaccordo e non diventa
mai `NO_VIOLATION`. Costo, con un'unica regola per pilot, volume e dati
italiani (`scudo.budget.paid_call`): dall'uso reale; senza uso riportato
(timeout, errore di rete, 5xx, corpo interrotto dopo un 200, risposta HTTP non
valida, provider muto) si addebita il caso peggiore, perché il provider può aver
già fatturato; lo stesso se la chiamata solleva un'eccezione imprevista, che
poi interrompe la run; una richiesta respinta (4xx) costa zero; 401/403
fermano la run senza registrare un esito.

Il dettaglio degli errori (al più 200 caratteri del messaggio del provider)
finisce in esiti, registro dei risultati e stderr: prima se ne tolgono ogni
frammento della chiave lungo almeno 4 caratteri e ogni token dall'aspetto di
chiave (`Bearer …`, `sk-…`, `hf_…`), sostituiti da `[redatto]`, perché alcuni
provider rimandano parte della chiave nei 401.

## Cosa viene registrato

Nella cartella del pilot (fuori dal repository): `sample.jsonl`,
`sample-meta.json`, `outcomes.jsonl` (un esito per teacher e caso, con token,
costo e `margin` se il provider dà i logprob), `report.json`, `summary.md` (da incollare nell'issue),
`reasoning-review-<teacher>.csv` (30 ragionamenti per teacher, concordi e
discordi con la label, colonne `reasoning_ok` e `notes` per la revisione
umana), `relabel-spotcheck.csv`.

Nel registro della spesa (JSONL fuori dal repository): `authorisation`
(importo confermato), `call` (costo di ogni chiamata), `pilot_report`,
`volume_confirmation` (teacher scelti, un solo importo totale, chi, perché;
le conferme più vecchie con un solo campo `teacher` restano valide). Nessuna
chiave e nessun testo dei dataset entra nel registro.

## Report

Per teacher: accordo con la label (complessivo, sugli `ok`, per verdetto,
lingua e superficie), tassi di rifiuto, malformati, troncati e timeout/errori,
token medi e costo per item, proxy dei ragionamenti (coerenza delle menzioni di
verdetto, al più tre frasi, sovrapposizione lessicale con l'evidenza: un proxy
meccanico, la qualità la giudica la revisione umana). Il campione è bilanciato
per strato: l'accordo complessivo vale su quella composizione.

La stima del volume v0.1 usa parametri espliciti (`--fast-items 200000`,
`--fast-calls-per-item 2`, `--non-en-share 0.6`,
`--translation-calls-per-item 2`, `--reasoning-traces 40000`):

- #8: esempi × giudizi per esempio × costo medio per chiamata;
- #9: tracce ÷ accordo con la label (si tengono solo quelle corrette) × costo
  per chiamata;
- #7: item IT/misti × (traduzione + back-translation) × costo di una chiamata
  con ingresso pari ai token medi d'ingresso del pilot e uscita pari
  all'evidenza media (caratteri/4) più i token di ragionamento medi.

Il pilot non misura la qualità della traduzione né della generazione nativa:
l'accordo sull'italiano misura il giudizio su testo italiano; la traduzione si
valida con lo spot-check di #7.

## Dopo il pilot

La persona sceglie il teacher (o i teacher, per la difficoltà di #8 che nasce
dal disaccordo fra due teacher) dal report, registra provider, versione e
termini verificati (il TOML e `summary.md` nell'issue #5) e, solo se vuole
procedere, esegue `confirm-volume` con un `--teacher` per ciascuno e un solo
`--amount` totale. Gli strumenti del volume (#7, #8, #9) chiamano
`scudo.budget.check_volume` prima di spendere: insieme di teacher e residuo
dell'ultima conferma, poi il tetto. Una nuova conferma sostituisce la
precedente (teacher e importo). Dati italiani
(#7): [italian-data.md](italian-data.md).
