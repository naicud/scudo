# Scudo

Scudo addestra e valuta sensori semantici generalisti che giudicano un testo
rispetto a un criterio. Il giudizio semantico e la decisione di policy sono
concetti distinti: Scudo produce solo il primo. Glossario derivato da quello
del progetto del gateway, ristretto ai termini che servono qui.

## Language

### Sensori

**Criterio semantico**:
Una domanda versionata che ammette VIOLATION, NO_VIOLATION o INSUFFICIENT sulla
base delle evidenze disponibili.
_Avoid_: permesso del modello, sicurezza percentuale.

**Verdetto tri-state**:
L'uscita del sensore per un criterio: `VIOLATION`, `NO_VIOLATION` o
`INSUFFICIENT` quando il testo non contiene l'evidenza che il criterio
richiede. Non è un'azione né un permesso.
_Avoid_: safe/unsafe, ALLOW/BLOCK, punteggio di rischio.

**Sensore semantico**:
Il produttore del giudizio per un criterio; non è un'autorità che concede permessi.
_Avoid_: decisore autorizzativo, policy neurale.

**Fast**:
Il tier del sensore tipizzato: un modello specializzato in decisioni tipizzate che
dà il giudizio leggendo gli score delle opzioni, senza generare testo.
_Avoid_: modello piccolo, classifier generico.

**Accurate**:
Il tier del sensore ragionante: un LLM generale che produce un ragionamento prima
del giudizio tri-state e ne espone lo score delle opzioni.
_Avoid_: modello grande, giudice, ground truth del Fast.

**Modello generalista**:
Il Fast o l'Accurate pubblico, addestrato a giudicare criteri semantici liberi,
anche mai visti, su testo IT/EN/misto.
_Avoid_: modello a tassonomia fissa, classificatore di categorie.

**Criteri di default**:
L'insieme versionato di criteri semantici pubblicati insieme al generalista,
che esprime come criteri le categorie di rischio classiche.
_Avoid_: categorie del modello, tassonomia cablata.

**Adapter di specializzazione**:
Un adattamento del generalista a un dominio o ai criteri di un cliente. Non
cambia il tier né il contratto tri-state.
_Avoid_: modello verticale, fork del generalista.

**Criterio di difficoltà**:
Il criterio valutato dal Fast insieme a quello semantico, che stabilisce se il
caso richiede l'Accurate.
_Avoid_: confidenza, router neurale separato.

**Escalation**:
Il passaggio di un caso dal Fast all'Accurate, deciso dal consumatore sul
criterio di difficoltà, sul verdetto e sul margine. Se l'Accurate non risponde,
il verdetto del Fast non lo sostituisce.
_Avoid_: retry, seconda opinione facoltativa.

### Dati e valutazione

**Teacher**:
Il modello con licenza aperta che traduce, genera casi e scrive ragionamenti per
il training. Le sue uscite sono candidate, non gold: entrano solo se coincidono
con la label o superano la revisione.
_Avoid_: ground truth, giudice, oracolo.

**Gold set**:
L'insieme di casi con label verificata da revisione umana, tenuto fuori dal
training e deduplicato contro di esso, su cui si misurano generalisti e
baseline.
_Avoid_: split di validazione, output del teacher.

**Baseline**:
Un guard pronto di terzi valutato sullo stesso harness del generalista come
termine di confronto, non come punto di partenza.
_Avoid_: candidato Fast, modello di riferimento gold.

**Data card**:
La scheda di un dataset con fonte, licenza, generatore, filtri, deduplicazione
e attribuzioni. Senza data card un dataset non entra nel manifest.
_Avoid_: README del dataset, nota informale.

**Manifest**:
L'elenco versionato dei dataset o degli artefatti pubblicati, con licenza,
revisione e digest. Il gateway consuma i GGUF tramite il manifest dei modelli.
_Avoid_: lista di link, ultima versione disponibile.
