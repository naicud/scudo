# Data card: `nvidia/Nemotron-Safety-Guard-Dataset-v3`

Una data card per ogni dataset che entra nel training o nell'eval. Un dataset
senza data card completa non entra nel manifest.

## Fonte

- Id e URL: `nvidia/Nemotron-Safety-Guard-Dataset-v3`,
  https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3
- Revisione (commit o versione) letta: `a3f7ecb3433d1933701a83f18de16c36934a7f51`
  (ultima modifica sul Hub: 2026-02-03; la stessa revisione è fissata in `REVISION`
  in `src/scudo/nemotron.py`)
- Data di lettura: 2026-10-15
- Config e split usati: config `default` (unica). File usati dal tracer:
  `it/train.jsonl`, `en/train.jsonl`, `it/test.jsonl`, `en/test.jsonl`. Lo split
  `validation` (`it/valid.jsonl`, `en/valid.jsonl`) non è usato dal tracer #2.
- Righe usate, per split e lingua:
  - Dataset totale: 514.617 righe (train 451.424, validation 27.791, test 35.402), come
    dichiarato nel README; la somma coincide con i conteggi per split.
  - Train, italiano: 37.534 righe disponibili; inglese: 40.007 righe disponibili. Il
    tracer #2 prende 1.000 casi per lingua, in modo deterministico (hash con seed).
    Una riga può dare un caso `input` e un caso `output`. Dimensione del pool dei casi
    dopo i filtri: non calcolata.
  - Test (held-out), italiano: 2.950 righe; inglese: 2.964 righe. Il tracer #2 prende
    200 casi per lingua, dopo i filtri e l'esclusione per id e testo esatto rispetto al
    campione di training. Conteggi dopo i filtri: non registrati.
  - Validation, italiano: 2.307 righe; inglese: 2.445 righe. Non usate.

## Licenza

- Licenza dichiarata (`cardData.license` o equivalente): `cc-by-4.0`
- Fonte della verifica (README, file LICENSE, paper): metadati API di Hugging Face
  (letti 2026-10-15) e sezione «License/Terms of Use» del README («CC-BY 4.0»). Il file
  LICENSE del repo non è stato letto.
- Gated: no (`gated: false` nei metadati API); termini aggiuntivi: nessuno dichiarato.
  Il README indica il dataset come «ready for commercial/non-commercial use».
- Uso commerciale e pesi derivati ammessi: SÌ, secondo il produttore. La voce sui
  generatori e sui teacher resta INCERTO (vedi sotto).
- Licenze delle fonti a monte, se il dataset ne aggrega altri:
  - Aegis 2.0 (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`): CC-BY-4.0, riletto il
    2026-10-15 da API e README alla revisione `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`
    (voce `aegis2-train` di `src/scudo/manifest.toml`).
  - Sottoinsieme `refusal_data_augmentation` (rifiuti generati con Gemma-2-27B in Aegis
    2.0): soggetto ai termini Gemma. Escluso dal tracer.
  - Pipeline CultureGuard: licenze dei modelli e del traduttore non verificate (vedi
    «Generatore e teacher»).

Ammesse: Apache-2.0, MIT, CC0, CC-BY, ODC-BY. Escluse: NC, CC-BY-SA, gated con
termini propri, dati generati o distillati da modelli con licenza che limita
uso o training.

## Generatore e teacher

- Testi scritti da: misto. Il README dichiara dati generati sinteticamente con la
  pipeline CultureGuard, che adatta culturalmente e traduce il dataset Aegis 2.0 di
  NVIDIA, e aggiunge prompt jailbreaking sintetici. Il modello che genera o adatta i
  testi non è dichiarato nel README. Secondo la nota di ricerca, i prompt di Aegis 2.0
  derivano da fonti umane (HH-RLHF, DAN, AART) e le risposte di Aegis 2.0 da
  Mistral-7B-v0.1.
- Label prodotte da: misto. Conteggi `prompt_label_source` (train): umani 333.026,
  `llm_jury` 118.398. Conteggi `response_label_source` (train): umani 51.340,
  `llm_jury` 219.122, `refusal_data_augmentation` 11.608. Il README non dichiara quali
  modelli compongono la giuria. Secondo la nota di ricerca, la pipeline CultureGuard usa
  come giuria di filtro Llama-Nemotron-Safety-Guard-V2, PolyGuard-Qwen,
  Llama-3.1-Nemotron-70B e Qwen3-235B (citazione [cg]); per le label di Aegis 2.0 la
  nota cita Mixtral-8x22B, Mistral-NeMo-12B e Gemma-2-27B. Che questa giuria abbia
  prodotto le label di Nemotron v3 è un'inferenza della nota, non verificata nel README.
- Teacher usati da noi (traduzione, generazione, ragionamenti), con licenza: nessuno nel
  tracer #2, che seleziona, filtra e mappa le righe senza generare, tradurre o
  ragionare. Licenza: non applicabile.
- Termini d'uso dei generatori compatibili con il training: INCERTO. Motivi:
  - Traduzione: la nota di ricerca indica Google Translate come traduttore del dataset;
    i termini di Google Translate per questo uso sono NON VERIFICATI.
  - Modelli Llama della giuria: la nota riporta i vincoli della licenza Llama 3 (divieto
    di usare gli output per migliorare altri LLM) e della licenza Llama 3.1 (obbligo di
    anteporre «Llama» ai modelli distribuiti addestrati con i suoi output). Non è
    verificato se e come si applichino a questo dataset.
  - Qwen3-235B e Llama-3.1-Nemotron-70B: licenze non verificate.
  - Sottoinsieme Gemma (rifiuti di Aegis 2.0): escluso dal tracer.

## Trasformazioni e filtri

- Mappatura delle label su `VIOLATION` / `NO_VIOLATION` / `INSUFFICIENT`: `unsafe` diventa
  `VIOLATION`, `safe` diventa `NO_VIOLATION`, per il prompt (superficie `input`) e per la
  risposta (superficie `output`). Una label diversa da queste è un errore (`ValueError`),
  non uno scarto silenzioso. Una risposta senza label non genera un caso `output`.
  Nessun caso è mappato su `INSUFFICIENT`.
- Costruzione dei casi `INSUFFICIENT`: nessuna, rinviata a issue #6. Il criterio chiede al
  giudice di rispondere INSUFFICIENT quando manca contesto essenziale, ma il dataset non
  ha una label corrispondente.
- Criteri semantici associati: `HARMFUL_REQUEST` (input) e `HARMFUL_RESPONSE` (output) in
  `src/scudo/aegis.py` (lo schema di Aegis 2.0, riesportati da `src/scudo/nemotron.py`), in
  inglese. Chiedono se il messaggio dell'utente, o la risposta
  dell'assistente, richiedono, promuovono o contengono contenuti vietati da una policy di
  content safety. Le categorie elencate sono violenza, minacce, armi, pianificazione
  criminale, frode, malware, sostanze controllate, autolesionismo o suicidio, contenuti
  sessuali (inclusi quelli che coinvolgono minori), odio, molestie, profanità, dati
  personali, manipolazione e altri contenuti vietati. Domande innocue, educazione,
  prevenzione e analisi non sono unsafe. Il criterio è allineato alla policy Aegis 2.0
  da cui il dataset deriva; la corrispondenza con le 12 categorie non è stata verificata
  voce per voce.
- Filtri (lingua, lunghezza, qualità, accordo con la label):
  - lingua `it` o `en`;
  - righe con `prompt` uguale a `"REDACTED"` escluse;
  - righe con `response_label_source` uguale a `refusal_data_augmentation` escluse per
    intero, prompt compreso;
  - caso `output` solo se `response` e `response_label` sono presenti;
  - nessun filtro su lunghezza, qualità o accordo tra label, e nessuna deduplica interna
    al training nel codice;
  - il campione è indipendente dall'ordine delle righe (ordinamento per hash SHA-256 di
    seed, id e superficie).
- Traduzione: traduttore, controllo a campione, back-translation:
  - Il tracer non traduce.
  - Traduttore del dataset: la nota di ricerca indica Google Translate; il README non lo
    dichiara.
  - Controllo a campione da madrelingua: non dichiarato.
  - Back-translation: la nota di ricerca cita per CultureGuard un filtro di coerenza con
    back-translation [cg]; non verificato nel README.
  - Per l'italiano, la nota segnala che il testo è tradotto a macchina, non nativo.
- Righe scartate e motivo:
  - Nel tracer, i conteggi per filtro non sono registrati.
  - Sull'intero dataset (train): 11.608 righe con `refusal_data_augmentation` e 10.728
    righe con `reconstruction_id_if_redacted` valorizzato. Non è verificato che la
    seconda cifra coincida con i prompt `REDACTED`.
  - Conteggi per italiano e inglese: non calcolati.

## Deduplicazione contro i benchmark

- Benchmark controllati: tutte le voci `benchmark` di `src/scudo/manifest.toml` (test di
  Aegis 2.0 e di Nemotron v3, XSTest, XSTest-IT, WildGuardTest, ToxicChat test, JBB-Behaviors,
  OR-Bench-hard-1k), con `python -m scudo.ingest build` (issue #3). Oggi la build è bloccata
  dal cancello: la voce `nemotron3-train` ha termini dei generatori INCERTO senza decisione
  umana e WildGuardTest (gated) non ha campi di testo verificati. Il tracer #2 non è
  ancora deduplicato contro i benchmark esterni.
- Metodo: nella build, SHA-256 del testo normalizzato e near-duplicate MinHash (shingle di 5
  caratteri, 128 bin, LSH 25×5, Jaccard stimata >= 0,8), vedi `docs/datasets.md`. Nel
  tracer #2 resta solo il controllo interno per id e testo esatto tra test held-out e i
  2.000 casi del campione di training passati a `heldout_cases`.
- Righe rimosse: nessun conteggio finché la build non passa il cancello. Le righe del test
  di Nemotron v3 hanno gli stessi id di quelle del test di Aegis 2.0 (letto da
  datasets-server): sono traduzioni delle stesse righe.

## Attribuzioni

Per CC-BY e ODC-BY: titolo, autori, URL, licenza e modifiche apportate, nel
formato richiesto dalla licenza. Le stesse attribuzioni vanno nella model card.

- Titolo: Nemotron-Safety-Guard-Dataset-v3 (già Nemotron-Content-Safety-Dataset-Multilingual-v1;
  `pretty_name` nei metadati: CultureGuard).
- Autori: NVIDIA Corporation (dataset owner, README). Pubblicazione di riferimento:
  Joshi, R., Paul, R., Singla, K., Kamath, A., Evans, M., Luna, K., Ghosh, S., Vaidya, U.,
  Long, E., Chauhan, S. S., et al., «CultureGuard: Towards Culturally-Aware Dataset and
  Guard Model for Multilingual Safety Applications», arXiv:2508.01710 (2025).
- URL: https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3
  (revisione `a3f7ecb3433d1933701a83f18de16c36934a7f51`).
- Licenza: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Modifiche apportate: selezione delle righe italiane e inglesi dai file train e test;
  esclusione delle righe `refusal_data_augmentation` e dei prompt `REDACTED`; mappatura
  delle label in `VIOLATION` / `NO_VIOLATION`; costruzione di casi con criterio testuale;
  campionamento deterministico; per la superficie `output`, evidenza costituita dalla sola
  risposta; held-out costruito dal test. Il testo di prompt e risposta non viene
  modificato, ma viene inserito in un template di prompt con criterio e opzioni
  (`training_row` in `src/scudo/protocol.py`).

## Rischi noti

- Contenuti dannosi: il dataset contiene per costruzione prompt e risposte non sicuri
  (armi, frode, malware, autolesionismo, contenuti sessuali, anche su minori). Non copiare
  esempi nel repository.
- Sbilanciamento di classe: nel train, 264.101 prompt unsafe su 451.424 (58,5%). Tra le
  risposte etichettate, 65.235 sono unsafe su 218.173 (29,9%); 152.304 risposte hanno
  label vuota. Il campione del tracer è bilanciato per lingua, non per label; la
  distribuzione safe/unsafe del campione non è stata calcolata.
- Sbilanciamento di lingua: italiano e inglese hanno circa 37.500 e 40.000 righe di train,
  su 451.424 totali (12 lingue). Le altre dieci lingue non sono valutate.
- Label rumorose: la quota di label prodotte da giurie LLM è alta: 26,2% dei prompt di train
  (118.398) e 77,7% delle risposte etichettate di train (219.122 su 282.070). Non esiste una
  misura di accuratezza delle label contro gold umani.
- Incoerenza nella documentazione: il README dice che i dati sono interamente sintetici e
  che gli annotatori non hanno curato dati nuovi, ma `prompt_label_source` = `human` vale
  333.026 righe nel train. Il README non chiarisce. Osservazione dai conteggi, non
  verificata riga per riga: il tag `jailbreaking` (118.398) coincide con il numero di prompt
  etichettati `llm_jury`.
- Traduzione automatica: l'italiano è tradotto a macchina (secondo la nota), non è nativo.
  Può perdere sfumature culturali e di registro, e un errore di traduzione può cambiare
  la label.
- Temi sensibili: il README cita il dataset Suicide Detection come fonte di prompt
  redatti. I prompt `REDACTED` sono esclusi dal tracer, ma non è verificato quali altri
  prompt derivino dalla stessa fonte.
- Criterio in inglese su testi in italiano: il giudizio dipende dalla lingua del criterio.
  Va tenuto presente nella lettura dei risultati.
- Held-out piccolo: 200 casi per lingua non bastano per stime precise.
- Deduplicazione incompleta: il meccanismo esiste (issue #3) ma la build è bloccata dal
  cancello finché i termini dei generatori restano INCERTO, quindi il rischio di
  contaminazione resta.
