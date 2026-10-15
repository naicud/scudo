---
status: accepted
---

> Origine: ADR accettato nel repository privato `open-guard`, copiato qui senza
> modifiche al testo. Riferimenti a issue, tag, lane e percorsi sono di quel
> repository.

# Fast tipizzato, Accurate ragionante, escalation decisa dal Fast

Il sensore ha due tier con ruoli diversi, non due taglie dello stesso readout.
Il **Fast** è un modello specializzato in decisioni tipizzate letto sugli score
delle opzioni, senza generazione. L'**Accurate** è un LLM generale piccolo, come
Qwen o Bonsai, che genera un ragionamento e poi espone lo score delle opzioni sul
verdetto tri-state, così resta calibrabile come il Fast. La modalità `auto` è una
cascade: il Fast valuta sullo stesso prefisso il criterio semantico e un
**criterio di difficoltà**, e passa il caso all'Accurate quando il caso è
difficile, quando il verdetto è `INSUFFICIENT` o quando il margine è sotto
soglia. Policy, gateway, review e audit restano invariati: vedono soltanto il
giudizio tri-state.

Regole dell'escalation:

- una `VIOLATION` del Fast con margine sopra soglia e caso non difficile produce
  BLOCK senza escalation: l'escalation serve contro l'incertezza, e un falso
  blocco finisce in review;
- la policy può marcare una regola semantica come **regola con Accurate
  obbligatorio**; la policy di default lo fa per le regole della superficie
  `tool`, perché producono effetti;
- se l'Accurate non risponde (timeout, sovraccarico, errore), l'esito è REVIEW;
  il verdetto del Fast non lo sostituisce mai.

Fast e Accurate restano caricati entrambi sul profilo Mac 24 GB. Il serving usa
un solo runtime, il backend nativo GGUF; il backend Ollama esce dal prodotto.
MLX/Unsloth servono soltanto al training: un Fast addestrato da noi passa dalla
lane `train-smoke`, ripuntata su un backbone 2–4B, e si esporta in GGUF prima del
serving.

## Considered Options

- **Stesso readout a due taglie** (piano V5 precedente): entrambi i tier letti
  sugli score delle opzioni, l'Accurate era solo un modello più grande. Scartato
  perché non usa la capacità di ragionamento che giustifica l'Accurate.
- **Router come terzo modello** prima dei due sensori. Scartato: un terzo
  artefatto da addestrare, calibrare e tenere in memoria sul Mac 24 GB, mentre il
  criterio di difficoltà sul prefisso condiviso del Fast costa quasi nulla.
- **Router a sole soglie** su margine ed entropia. Resta come segnale insieme al
  criterio di difficoltà, non da solo: un errore confidente del Fast non verrebbe
  mai instradato.
- **Ollama per l'Accurate**. Scartato: non espone in modo affidabile gli score
  delle opzioni dopo il ragionamento e aggiunge un secondo runtime da misurare.
- **Lane Clef** (testa di classificazione con LoRA) per il fine-tune. Scartata:
  il backend nativo non serve quella testa, mentre `train-smoke` legge gli stessi
  token del serving.

## Consequences

- Fast e Accurate si scelgono solo fra pesi con licenza aperta senza limiti d'uso
  commerciale (Apache-2.0, MIT o equivalenti). Restano fuori modelli con soglie di
  fatturato o di utenti, o con termini d'uso del produttore (LFM, Gemma, Llama,
  NVIDIA Open Model License).
- Il Fast si sceglie fra checkpoint pronti per decisioni tipizzate (Rizzo Flow,
  Granite con policy-guardrails e alternative) e un fine-tune nostro con Unsloth
  su backbone 2–4B; SmolLM2-135M resta solo per gli smoke meccanici.
- Rimossi dal repository, recuperabili dal tag `pre-retarget-2026-10-09`: lane
  Clef ed esperimenti nativi, `mlx_research`, Ollama, Garak, campagne, reload
  nativo, acceptance Langfuse, dossier di delivery e blueprint storici.
- La valutazione usa un solo harness (corpus IT/EN e Promptfoo) con un report per
  candidato; i gate restano soltanto per la promozione in produzione.
