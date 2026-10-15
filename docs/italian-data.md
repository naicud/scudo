# Dati italiani: traduzione, generazione nativa e misto IT/EN (issue #7)

Con il **teacher** scelto nel pilot ([teacher-pilot.md](teacher-pilot.md)) si
traducono in italiano casi inglesi con label gold, si generano casi nativi
italiani e misti IT/EN, si deduplica contro i benchmark, un madrelingua
controlla 300 item e solo allora il lotto entra nel mix 40/40/20 IT/EN/misto.
Ogni uscita del teacher è candidata: entra solo se la label coincide (accordo
di label) e se il lotto supera lo spot-check umano.

Il codice non spende nulla da solo e non compila nessuna decisione umana. Prima
di ogni chiamata servono: la conferma del volume registrata dalla persona con
lo stesso teacher (`python -m scudo.pilot confirm-volume`), una stima nel
residuo del volume confermato e del tetto v0.1 di 150 USD (condiviso col
pilot), e `--confirm-spend` uguale alla stima.

Moduli: `scudo.translate` (ammissibilità, traduzione, back-translation, accordo
di label, attribuzioni), `scudo.generate_it` (domini, criteri, generazione e
validazione separata), `scudo.italian_run` (piano, stima, run, deduplica),
`scudo.spotcheck_it` (campione ed esito umano), `scudo.language_mix` (mix),
`scudo.italian` (comandi). Il client è quello del pilot
(`scudo.teacher_client`), il tetto `scudo.budget`.

## Cosa si traduce

Solo casi che soddisfano tutte le condizioni (`scudo.translate.eligible`):

- lingua `en` e label gold: i casi `needs_relabel` (label `llm_jury`) restano
  fuori finché la giuria di teacher non li rietichetta;
- fonte di training del manifest (id del dataset e split) che passa il
  cancello: niente fonti gated, niente termini dei generatori `NO` o `INCERTO`
  senza decisione umana;
- licenza che ammette opere derivate e uso commerciale: Apache-2.0, MIT, CC0,
  CC-BY, ODC-BY. Per CC-BY e ODC-BY la modifica si attribuisce:
  `ATTRIBUTIONS-it.md` nel lotto riporta titolo, autori, URL, revisione,
  licenza e «evidenza tradotta automaticamente in italiano con `<teacher>` via
  `<provider>`»; ogni caso tradotto ha la stessa nota in provenienza.

Si traduce solo l'evidenza: il criterio resta il testo versionato d'origine.
Il caso tradotto tiene `source` della fonte, `source_id` con suffisso `#it`,
superficie e verdetto gold; in provenienza: voce del manifest, licenza, testo
originale, traduttore (modello, provider, modello presso il provider,
revisione servita), back-translation e punteggio.

Stato con il manifest del 2026-10-15: oggi produce casi solo `aegis2-train`
(CC-BY-4.0, ammessa; convertitore `scudo.aegis`), quindi è l'unica fonte
traducibile. Le altre fonti ammesse dal cancello non hanno ancora un
convertitore; quelle con termini `INCERTO` (`nemotron3-train`,
`deepset-prompt-injections`, `gandalf-ignore-instructions`,
`yanismiraoui-prompt-injections`) sono bloccate finché una persona non scrive
la decisione; `nemotron-sft-safety-v2` ha termini `NO` (italiano tradotto con
Riva-Translate, licenza che vieta di usarne gli output per altri modelli). `plan-meta.json` registra lo stato di ogni fonte incontrata e gli
scarti per motivo.

**Traduttore**: è il teacher del pilot; la run rifiuta un modello del registro
con licenza diversa da Apache-2.0 o MIT. Tower e NLLB (CC-BY-NC) sono esclusi.
I termini del provider sugli output li verifica la persona e li scrive nel TOML
(`terms_*`), come nel pilot.

## Controlli di accettazione

Traduzione, tre chiamate separate in ordine; la prima che fallisce chiude
l'item come rifiutato con il motivo:

1. traduzione EN→IT (JSON `{"translation": ...}`); rifiutata anche se, dopo la
   normalizzazione di `scudo.dedup`, è identica all'originale
   (`translation_unchanged`);
2. back-translation IT→EN e **similarità** con l'originale: F1 dei token sul
   multinsieme delle parole normalizzate con `scudo.dedup.normalize` (NFKC,
   casefold, punteggiatura in spazi). Sotto la soglia
   `back_translation_below_threshold`. Soglia: parametro `--bt-threshold` del
   piano, default 0,5, **non calibrata**: va rivista con l'esito dello
   spot-check (i punteggi restano in provenienza);
3. **accordo di label**: il teacher giudica il caso tradotto con il criterio
   d'origine (`scudo.teacher.messages`, la stessa chiamata del pilot, che non
   vede la label); entra solo se il verdetto è uguale al gold
   (`label_disagrees` altrimenti).

Generazione nativa e mista, due chiamate separate:

1. la chiamata di generazione riceve dominio, argomento, superficie, criterio,
   verdetto richiesto e lunghezza, e restituisce `{"evidence": ...}`;
2. una chiamata **distinta** giudica l'evidenza con il solo criterio
   (`scudo.teacher.messages`): non vede il verdetto richiesto né la richiesta
   di generazione. Il candidato entra solo se il verdetto coincide.

Il giudice è lo stesso modello del generatore in una chiamata nuova: è
indipendente la chiamata, non il modello.

Rifiuto, output malformato, troncamento, timeout o errore del teacher
diventano un rifiuto con motivo (`translation_refused`, `label_timeout`,
`generation_malformed`, ...), mai una label.

### Domini e criteri della generazione

| Dominio | Argomenti a rotazione | Criteri di default |
|---|---|---|
| `registro` | dialetto, burocratese, slang, chat, ironia | profanity, harassment, threat |
| `pa` | anagrafe, INPS, Agenzia delle Entrate, SPID/CIE, concorsi | fraud_deception, pii_privacy, high_risk_gov_decision |
| `banche` | conto, bonifici e IBAN, carte, mutui, home banking | fraud_deception, pii_privacy, criminal_planning |
| `sanita` | CUP e ASL, farmaci, fascicolo sanitario, pronto soccorso, salute mentale | unauthorized_advice, self_harm, pii_privacy |
| `hate_speech` | immigrazione, religione, orientamento e identità, disabilità, Nord/Sud | hate, harassment, threat |
| `misoginia` | lavoro, politica, social, sport, relazioni | hate, harassment |

Ogni dominio copre le cinque superfici v1 con target `VIOLATION` e
`NO_VIOLATION`; l'over-refusal solo `NO_VIOLATION`. Nessun target
`INSUFFICIENT`: dal testo libero non si costruisce (issue #6). Le superfici di
injection usano due criteri v0 scritti in `scudo.generate_it`
(`INJECTION_CRITERIA`), perché nessun criterio di default le copre. I misti
(`mixed`) chiedono italiano con inserti inglesi (code-switching). Fonte dei
casi generati: `scudo/teacher-generated`; `source_id` = primi 24 caratteri
dello SHA-256 del testo generato; il generatore è in provenienza.

## Spesa

Chiamate per item: 3 per una traduzione, 2 per una generazione. La stima è il
caso peggiore: ingresso in caratteri/4 più `max_tokens` in uscita per ogni
chiamata, e le uscite dei passi precedenti (traduzione, testo generato)
contano `max_tokens`. La stima del volume #7 del report del pilot conta 2
chiamate per item (traduzione e back-translation): l'accordo di label ne
aggiunge una.

`run` rifiuta, prima di qualsiasi chiamata, se manca la conferma del volume, se
il teacher non è quello confermato, se la stima arrotondata al centesimo supera
il residuo del volume confermato (importo dell'ultima `volume_confirmation`
meno le chiamate registrate dopo) o del tetto, o se `--confirm-spend` non
coincide. Poi un item parte solo se il suo caso peggiore sta nell'autorizzato e
ogni chiamata è controllata dallo stesso stop rigido del pilot. Ogni chiamata
va nel registro (`call`, `run_id` = `it-<sha del piano>`); 401/403 fermano la
run senza registrare l'item. Una run interrotta riprende saltando gli item
fatti, con nuova stima e nuova conferma.

## Deduplica

`dedup` scarica i benchmark del manifest alle revisioni fissate (extra
`train`) e usa lo stesso indice di `scudo.ingest` (esatta SHA-256 e
near-duplicate MinHash, soglia 0,8). Si confronta l'evidenza italiana e, per i
tradotti, anche l'originale inglese; tra i benchmark ci sono gli italiani
`xstest-it` e `nemotron3-test`. Si tolgono anche i duplicati esatti interni al
lotto. I benchmark che non passano il cancello (oggi `wildguardtest`, campi di
testo non verificati) non sono indicizzati e `dedup-report.json` li elenca in
`not_indexed`.

## Spot-check umano

`spotcheck-export` sceglie per hash seminato 300 item `it` dal pool deduplicato
(`--n`, minimo della specifica; fallisce se sono meno) e, **in più**, gli item
`mixed` nella stessa proporzione del pool, `ceil(n · misti / italiani)`: con un
pool 40/20 IT/misto sono 150 misti, 450 righe in tutto. I misti entrano nel mix
con il lotto, quindi passano dallo stesso madrelingua. Scrive `spotcheck.csv`
nel lotto, in UTF-8, con italiani e misti mescolati:

| colonna | contenuto |
|---|---|
| `item_id` | id opaco |
| `criterion` | criterio del caso |
| `evidence` | testo italiano |
| `verdict` | verdetto proposto |
| `judgement` | da compilare: `correct` o `incorrect` |
| `notes` | da compilare, facoltativo; utile per gli `incorrect` |

Il foglio non mostra fonte, modello, tipo (tradotto o generato) né punteggi.
Un item è `correct` solo se il testo è italiano naturale e comprensibile per un
madrelingua (per una traduzione: nessun errore di senso evidente) **e** il
verdetto proposto è giusto per il criterio. Altrimenti `incorrect`.

Il primo export di un pool si registra in `spotcheck-exports.jsonl` (seme,
righe per lingua, hash di pool e foglio, ora UTC). Rieseguire l'export con lo
stesso seme riscrive lo stesso foglio; un altro `--seed` o un altro `--n` sullo
stesso pool si rifiuta, così il campione non si sceglie rieseguendo l'export.
Solo un pool cambiato ammette un nuovo export.

`spotcheck-import --review FILE --by NOME` rifiuta un foglio con item mancanti,
estranei o senza `judgement` e un campione che non coincide con l'export
registrato (meta modificato, registro assente, item diversi da quelli che il
seme dà sul pool), poi registra in `spotcheck-outcomes.jsonl` chi, quando (UTC),
corretti, accuratezza complessiva e per lingua (`by_language`), seme e ora
dell'export, soglia, esito e gli item `incorrect` con le note. La soglia è 0,95
(specifica #7): `--threshold` la può alzare, mai abbassare. Il lotto è
accettato solo se **italiani e misti, ciascuno**, sono almeno alla soglia
(almeno 285 su 300 italiani e, con 150 misti, 143 su 150). L'esito è legato
all'hash del pool: se il pool cambia serve un nuovo campione. Un lotto respinto
resta respinto: niente nuovo campione e niente mix; si rifà un lotto in una
nuova cartella.

## Mix

`mix` unisce i casi del build (`--cases`, solo label gold di fonti ammesse dal
cancello) e i pool dei lotti (`--batch`, solo con spot-check accettato) e
sceglie `--n` casi con le quote `--shares` (default `it=0.4,en=0.4,mixed=0.2`,
resti maggiori), per hash seminato. Se una lingua non ha abbastanza casi il
comando fallisce elencando le carenze: niente riempimento con le altre lingue.
Scrive `mix.jsonl` (caso, `key`, `origin`, provenienza per i casi del teacher) e
`mix-meta.json` in `--dest`.

## Passi

Tutte le cartelle fuori dal repository.

```sh
# 0. Prerequisiti: build dei casi (#3) e conferma del volume dopo il pilot (#5)
rtk proxy uv run --extra train python -m scudo.ingest build --out /perimetro/build
rtk proxy uv run python -m scudo.pilot confirm-volume --ledger /perimetro/spesa-v0.1.jsonl \
  --teacher <id> --amount <USD> --by "<nome>" --rationale "<perché>"   # solo la persona

# 1. Piano: nessuna chiamata
rtk proxy uv run python -m scudo.italian plan --out /perimetro/it-lotto1 \
  --cases /perimetro/build/aegis2-train.jsonl --translate <N> --native <N> --mixed <N> \
  [--seed 0] [--bt-threshold 0.5]

# 2. Prova a secco
rtk proxy uv run python -m scudo.italian estimate --out /perimetro/it-lotto1 \
  --teachers /perimetro/teachers.toml --teacher <id> --ledger /perimetro/spesa-v0.1.jsonl

# 3. Run: --confirm-spend uguale al valore `confirm` della stima
export <NOME_VARIABILE>=...
rtk proxy uv run python -m scudo.italian run --out /perimetro/it-lotto1 \
  --teachers /perimetro/teachers.toml --teacher <id> --ledger /perimetro/spesa-v0.1.jsonl \
  --confirm-spend <confirm>

# 4. Deduplica contro i benchmark (scarica, extra train)
rtk proxy uv run --extra train python -m scudo.italian dedup --out /perimetro/it-lotto1

# 5. Campione per il madrelingua, poi l'esito
rtk proxy uv run python -m scudo.italian spotcheck-export --out /perimetro/it-lotto1
rtk proxy uv run python -m scudo.italian spotcheck-import --out /perimetro/it-lotto1 \
  --review /perimetro/spotcheck-compilato.csv --by "<madrelingua>"

# 6. Mix
rtk proxy uv run python -m scudo.italian mix --dest /perimetro/mix-v0.1 \
  --cases /perimetro/build/aegis2-train.jsonl --batch /perimetro/it-lotto1 --n <N>
```

Le quantità `--translate`, `--native`, `--mixed` e `--n` le decide la persona
dalla stima e dal residuo; per un mix 40/40/20 servono, dopo rifiuti e
deduplica, almeno 0,4·n casi italiani e 0,2·n misti.

## Cosa viene registrato

Nel lotto: `plan.jsonl`, `plan-meta.json`, `results.jsonl` (per item: esito,
motivo, caso, provenienza, chiamate con token e costo), `pool.jsonl`,
`dedup-report.json`, `ATTRIBUTIONS-it.md`, `spotcheck.csv`,
`spotcheck-meta.json`, `spotcheck-exports.jsonl`, `spotcheck-outcomes.jsonl`.
Nel registro della spesa:
`authorisation` e `call` della run. `run` stampa i conteggi di accettati per
origine e lingua e di rifiutati per tipo e motivo.

## Limiti

- La soglia di back-translation (0,5) è un punto di partenza non calibrato;
  la F1 dei token penalizza le parafrasi corrette.
- La lingua dell'uscita non si verifica in automatico oltre a
  `translation_unchanged`: la controlla lo spot-check.
- Il giudice dell'accordo di label è lo stesso modello che traduce o genera.
- I criteri di injection v0 e i prompt di generazione non sono ancora stati
  provati su un teacher reale: nessuna run reale è stata eseguita.
