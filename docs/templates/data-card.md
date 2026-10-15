# Data card: `<owner/dataset>`

Una data card per ogni dataset che entra nel training o nell'eval. Un dataset
senza data card completa non entra nel manifest.

## Fonte

- Id e URL:
- Revisione (commit o versione) letta:
- Data di lettura:
- Config e split usati:
- Righe usate, per split e lingua:

## Licenza

- Licenza dichiarata (`cardData.license` o equivalente):
- Fonte della verifica (README, file LICENSE, paper):
- Gated: sì / no; termini aggiuntivi:
- Uso commerciale e pesi derivati ammessi: SÌ / NO
- Licenze delle fonti a monte, se il dataset ne aggrega altri:

Ammesse: Apache-2.0, MIT, CC0, CC-BY, ODC-BY. Escluse: NC, CC-BY-SA, gated con
termini propri, dati generati o distillati da modelli con licenza che limita
uso o training.

## Generatore e teacher

- Testi scritti da: umani / modello `<id>` / misto
- Label prodotte da: umani / modello `<id>` / regola
- Teacher usati da noi (traduzione, generazione, ragionamenti), con licenza:
- Termini d'uso dei generatori compatibili con il training: SÌ / NO / INCERTO

## Trasformazioni e filtri

- Mappatura delle label su `VIOLATION` / `NO_VIOLATION` / `INSUFFICIENT`:
- Costruzione dei casi `INSUFFICIENT`:
- Criteri semantici associati:
- Filtri (lingua, lunghezza, qualità, accordo con la label):
- Traduzione: traduttore, controllo a campione, back-translation:
- Righe scartate e motivo:

## Deduplicazione contro i benchmark

- Benchmark controllati (es. XSTest, XSTest-IT, WildGuardTest, HarmBench,
  ToxicChat, gold set):
- Metodo: hash esatto / near-duplicate (soglia):
- Righe rimosse:

## Attribuzioni

Per CC-BY e ODC-BY: titolo, autori, URL, licenza e modifiche apportate, nel
formato richiesto dalla licenza. Le stesse attribuzioni vanno nella model card.

## Rischi noti

Contenuti dannosi presenti, sbilanciamenti di lingua o classe, label rumorose.
