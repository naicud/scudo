# Fast v0.1: mix e criterio di difficoltà (issue #8)

Termini come in [`CONTEXT.md`](../CONTEXT.md); formato in
[`protocol.md`](protocol.md). Fonte: `src/scudo/mix_pool.py`,
`src/scudo/mix_clean.py`, `src/scudo/mix.py`, `src/scudo/mix_cli.py`,
`src/scudo/difficulty.py`, `src/scudo/fast_notebook.py`; test in
`tests/test_mix.py`, `tests/test_mix_cli.py`, `tests/test_difficulty.py`,
`tests/test_fast_notebook.py`.

## Criterio di difficoltà

Il Fast valuta sullo stesso caso il criterio semantico e il **criterio di
difficoltà** (ADR 0001). Id, testo e versione sono quelli di open-guard 2ed1e92
(`open_guard.scoring.schema`: `ROUTING_DIFFICULTY = "routing.difficulty"`,
`ROUTING_DIFFICULTY_REF = "routing.difficulty/1"`), copiati verbatim in
`scudo.difficulty.CRITERION`:

> Is this case hard to judge reliably at a glance for a small classifier: subtle or indirect wording, mixed or unclear language, long or layered context, quotation or role-play that blurs intent, or evidence that pulls toward opposite verdicts? Answer VIOLATION when the case is hard, NO_VIOLATION when it is plainly easy, and INSUFFICIENT when the difficulty itself cannot be determined.

Come lo legge il backend nativo (open-guard 2ed1e92):

- `application/evaluation.py`: in modalità `auto` il Fast legge ogni criterio
  semantico e poi `ROUTING_DIFFICULTY` con lo stesso `snapshot.text`;
- `scoring/protocol.make_score_request` e `scoring/rendering.render_score_prompt`
  lo rendono con lo stesso renderer `decision-token` dei criteri semantici, con
  la permutazione fissa del backend (`VIOLATION`, `NO_VIOLATION`, `INSUFFICIENT`);
- `application/cascade.escalation_reason`: VIOLATION = difficile, NO_VIOLATION =
  facile, INSUFFICIENT o readout fallito = non valutabile, quindi difficile.

"Stesso prefisso" significa stessa evidenza: nel JSON del prompt il criterio
viene prima dell'evidenza, quindi a livello di token i due prompt condividono solo
`{"criterion": "`. La riga di difficoltà del mix ha la stessa evidenza della riga
semantica del caso e il completamento `VIOLATION` (HARD) o `NO_VIOLATION` (EASY).
Il golden in `tests/test_difficulty.py` fissa i byte prodotti dal renderer di
open-guard.

### Label del primo giro

Dagli esiti dei teacher nel formato del pilot (`outcomes.jsonl`). La giuria sono i
teacher presenti negli esiti:

- **HARD**: ogni teacher ha un esito `ok` e i verdetti discordano fra loro o con il
  gold (`disagree`), oppure concordano ma il margine di almeno un teacher è sotto
  la soglia (`low_margin`);
- **EASY**: ogni teacher ha un esito `ok` uguale al gold con margine almeno alla
  soglia (`agree`) o senza margine (`agree_margin_missing`);
- **nessuna riga**, contata per motivo: manca l'esito di un teacher
  (`no_teacher_record`) o un esito non è `ok` (`teacher_refused`,
  `teacher_malformed`, ...). Un guasto del teacher non è un segnale di difficoltà.

Il **margine** è il campo `margin` dell'esito: logprob del token che decide il
verdetto meno quello della migliore alternativa che inizia un verdetto diverso,
in nat (`scudo.teacher_margin`). C'è solo se la configurazione del teacher chiede
i logprob (`request_extra = { logprobs = true, top_logprobs = 5 }`, vedi
`docs/teacher-pilot.md`) e il provider li restituisce; senza, il margine è
assente, mai inventato, e conta solo il disaccordo (`agree_margin_missing` nel
manifest dice quanti casi). Se la migliore alternativa non è nel top-k il
margine è un limite inferiore. Soglia: `--margin-threshold` di `mix_cli build`,
default `ln 4 ≈ 1,386` (verdetto meno di 4 volte più probabile della migliore
alternativa): una convenzione dichiarata da tarare sugli esiti del pilot, non
una misura. Il manifest registra soglia (`difficulty_margin_threshold`) e righe
per motivo (`composition.difficulty.by_reason`).

I casi `needs_relabel` confermati sono concordi per costruzione (la conferma è
l'accordo unanime con la label) e diventano HARD solo per margine basso. Il mix
si rifiuta se mancano casi HARD o EASY.

## Categorie

Prima regola che vale (`scudo.mix.category`):

1. verdetto `INSUFFICIENT` -> `insufficient`;
2. superficie `over_refusal` -> `over_refusal`;
3. superficie `injection_direct`/`injection_indirect`, o criterio `injection.*`
   dei criteri di default -> `injection`;
4. criterio di sicurezza dei contenuti (`aegis.HARMFUL_REQUEST`,
   `aegis.HARMFUL_RESPONSE`, criteri di default `aegis.*`) -> `prompt_response`;
5. ogni altro criterio (CantTalkAboutThis, procedural, Open-Jev, ...) ->
   `custom`.

## Target e override

| Parametro | Default | Significato |
| --- | --- | --- |
| `--total` | 200.000 | righe semantiche |
| `--categories` | `prompt_response=45,injection=20,custom=15,over_refusal=10` | pesi delle categorie non INSUFFICIENT |
| `--insufficient-share` | 0,12 (`insufficient.DEFAULT_SHARE`) | quota INSUFFICIENT del mix (10–15% dell'issue) |
| `--languages` | `it=0.4,en=0.4,mixed=0.2` | lingua dentro ogni categoria non INSUFFICIENT |
| `--method-cap` | 0,4 | quota massima di ogni origine fra gli INSUFFICIENT |

I casi non INSUFFICIENT sono `round(total × (1 − share))`, divisi per categoria e
poi per lingua (resto più grande); dentro lo strato si sceglie per hash seminato
della chiave, quindi l'ordine dell'ingresso non conta. Uno strato con meno casi
del necessario è un errore che elenca tutte le mancanze: nessun riempimento
parziale. Oggi mancano l'italiano e il misto (#7) e l'over-refusal (nessuna fonte
di training ammessa): servono `--languages` e `--categories` espliciti. Ogni
valore diverso dal default è un override, rifiutato senza `--override-reason` e
registrato nel manifest del mix con default, valore e motivo.

**INSUFFICIENT.** Origini: i metodi di `scudo.insufficient` e `native:<voce>` per
gli INSUFFICIENT della fonte (procedural). Ogni origine si taglia a
`floor(method_cap × quota INSUFFICIENT)` casi prima di `mix_insufficient`; se il
pool tagliato non basta il mix fallisce. Il default 0,4 è una scelta nostra, non
un dato: `evidence_removed` (dialogo che finisce con l'assistente, una scorciatoia)
era l'87% del pool generato. Il manifest riporta pool, tetto, scelti e quota per
origine.

## Pulizia

- **Duplicati** dentro una fonte e fra fonti: stesso criterio e stessa evidenza
  normalizzata (`scudo.dedup.normalize`, hash esatto). Resta il caso con label
  gold, poi per voce e id. Il near-duplicate MinHash resta solo contro i
  benchmark (ingestione).
- **Conflitti**: stesso criterio ed evidenza, verdetti diversi (es. in-the-wild
  jailbreak e regular, jackhhao e in-the-wild regular): escono tutte le copie,
  contate per coppia di voci.
- **Rietichettatura**: un caso `needs_relabel` (giuria `llm_jury` di Aegis 2.0,
  Gretel, in-the-wild regular) entra solo se ogni teacher della giuria ha un esito
  `ok` uguale alla label. Una giuria discorde non cambia la label: esclude il caso.
- **INSUFFICIENT generati** dai casi puliti; un figlio con criterio ed evidenza
  già presenti fra i candidati esce.

## Ingressi

- uscite di `scudo.ingest build`: solo le fonti di `report.json`, solo se il
  report viene dal manifest attuale (stesso digest) e ogni fonte è ancora ammessa
  dal cancello; il numero di casi deve coincidere con il report;
- lotti italiani di #7 (`--italian DIR`, ripetibile): solo con lo spot-check
  umano accettato per il pool attuale (`scudo.mix_pool.batch_pool`, vedi
  [`italian-data.md`](italian-data.md#ingresso-nel-mix)), e solo se ogni testo è
  scritto da un teacher del registro con la sua voce del manifest ammessa dal
  cancello alla stessa revisione (`teacher-gpt-oss-120b`,
  `teacher-qwen3.5-397b-a17b`). Un caso tradotto resta della voce della sua
  fonte, che deve essere ammessa; un caso generato (IT o misto) è della voce del
  teacher che lo ha scritto. Origine sconosciuta, modello fuori dal registro o
  fonte diversa dalla voce sono errori. È l'unico mix: non c'è un mix per lingua a
  parte, e la lingua si applica dentro ogni categoria.

## Manifest del mix

`mix-manifest.json`, accanto a `mix.jsonl` e a `ingest-report.json` (copia del
report della build), fuori dal repository: SHA del repository (albero pulito,
altrimenti rifiuto), formato, criterio di difficoltà, seme, target, override e
motivo, giuria, digest di report, manifest dei dati, ogni JSONL di fonte, lotti
italiani (pool e spot-check) ed esiti; candidati per voce, scarti per motivo, conflitti, rietichettatura,
INSUFFICIENT per origine, composizione delle righe semantiche e di difficoltà per
categoria, superficie, lingua, verdetto e voce, righe e sha256 di `mix.jsonl`.

## Passi della persona

```sh
# 1. Build dei dati (cancello prima di ogni download)
rtk proxy uv run --extra train python -m scudo.ingest build --out /perimetro/ingest
rtk proxy uv run python -m scudo.ingest verify --report /perimetro/ingest/report.json

# 2. Campione da giudicare: candidati puliti e INSUFFICIENT generati
rtk proxy uv run python -m scudo.mix_cli judge-set --ingest /perimetro/ingest \
  [--italian /perimetro/it-batch] --out /perimetro/judge

# 3. Giudizio in volume (spesa): stima, conferma umana, run
rtk proxy uv run python -m scudo.pilot estimate --out /perimetro/judge \
  --teachers /perimetro/teachers.toml --ledger /perimetro/spesa-v0.1.jsonl
rtk proxy uv run python -m scudo.pilot confirm-volume --ledger /perimetro/spesa-v0.1.jsonl \
  --teacher <id1> --teacher <id2> --amount <USD totali> --by "<nome>" --rationale "<perché>"
rtk proxy uv run python -m scudo.mix_cli judge --out /perimetro/judge \
  --teachers /perimetro/teachers.toml --ledger /perimetro/spesa-v0.1.jsonl \
  --confirm-spend <confirm della stima>

# 4. Mix (da un commit pulito)
rtk proxy uv run python -m scudo.mix_cli build --ingest /perimetro/ingest [--italian /perimetro/it-batch] \
  --outcomes /perimetro/judge/outcomes.jsonl [--outcomes /perimetro/pilot/outcomes.jsonl] \
  --out /perimetro/mix [--languages it=0,en=1,mixed=0 --categories ... --override-reason "..."]

# 5. Upload privato: stampa revisione e sha256 del manifest del mix
rtk proxy uv run --extra train python -m scudo.mix_cli upload --out /perimetro/mix \
  --repo Naicud/scudo-fast-mix
```

6. Colab A100: `notebooks/fast_v01.ipynb`, con `mix_revision` e
   `mix_manifest_sha256` del passo 5 in `RUN`.

`judge` rifiuta senza una `volume_confirmation` nel registro; poi la run del
pilot applica `scudo.budget.check_volume`: vale solo l'ultima conferma, ogni
teacher configurato deve essere nel suo insieme (`--teacher` ripetibile), e la
stima della run, somma su tutti i teacher, deve stare nel residuo del suo unico
importo e del tetto. Infine chiede la conferma esatta della stima e si ferma
all'importo autorizzato. La difficoltà di #8 nasce dal disaccordo fra due
teacher: si confermano entrambi nella stessa conferma, con un importo che copre
la somma. Una nuova conferma sostituisce la precedente. Gli esiti del pilot
(stessa chiave `<voce>:<id>:<superficie>`) si possono riusare con `--outcomes`.

Limiti: il campione da giudicare contiene tutti i candidati, quindi la stima del
passo 3 va letta prima di confermare; i figli INSUFFICIENT non sono deduplicati
contro i benchmark (lo sono i loro genitori).

## Notebook

`python -m scudo.fast_notebook` rigenera `notebooks/fast_v01.ipynb`. Setup, LoRA,
addestramento con ripresa e fingerprint, salvataggio dell'adapter ed export GGUF
sono le celle condivise di `scudo.notebook_cells`, le stesse del tracer e
dell'Accurate. Il notebook scarica il manifest del mix e ne verifica lo sha256,
poi `mix.jsonl` e `ingest-report.json` con i digest del manifest
(`artifacts.load_verified`), verifica la deduplica (`verify_dedup`), carica la base
dopo `runner.fetch_verified`, scarta contate le righe oltre `max_seq_length`,
addestra insieme righe semantiche e di difficoltà, valuta un campione per
benchmark con l'harness di #4 (`benchmarks`, `runner`, `eval_report`) e aggiunge a
ogni report `difficulty_routing` (`scudo.difficulty.routing_metrics`):
`hard_recall_on_wrong` (quota degli errori del Fast mandati all'Accurate, la misura
che #12 deve migliorare), `hard_rate_on_correct`, `hard_rate` e i readout di
difficoltà, per lingua e superficie. RUN_LOG: SHA di scudo, revisione e digest del
mix, SHA della build del mix, override, pesi della base, adapter, GGUF, metriche.

NON VERIFICATO su Colab: memoria con `max_seq_length` 4096 e batch 4×4 su A100
40 GB, durata, loader Unsloth da cartella locale, export GGUF del 4B addestrato.
