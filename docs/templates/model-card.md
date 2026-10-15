---
license: apache-2.0
language:
  - it
  - en
base_model: Qwen/Qwen3.5-4B
datasets:
  - <owner/dataset>
pipeline_tag: text-generation
tags:
  - guardrail
  - safety
  - scudo
---

# `naicud/scudo-<tier>-<size>`

Sostituire i segnaposto e allineare `base_model` al tier: `Qwen/Qwen3.5-4B` per
il Fast, `Qwen/Qwen3.5-9B` per l'Accurate. `datasets` elenca ogni dataset del
manifest.

## Uso previsto

Sensore semantico generalista IT/EN: dato un testo e un criterio semantico in
linguaggio naturale, restituisce un verdetto tri-state. Superfici della v1:
prompt utente, risposta del modello, prompt injection diretta e indiretta,
over-refusal.

## Fuori perimetro

- decisioni di policy, permessi o autorizzazioni: le prende il gateway;
- esecuzione di tool e valutazione di azioni di agenti (v2);
- rilevamento di PII (v2);
- input non testuali e lingue diverse da IT/EN.

## Contratto di uscita

- Verdetti: `VIOLATION`, `NO_VIOLATION`, `INSUFFICIENT`.
- Lettura: score delle opzioni sul verdetto (Fast senza generazione; Accurate
  dopo il ragionamento).
- Formato del prompt e template:
- Un output malformato, un timeout o un errore non equivalgono a
  `NO_VIOLATION`: il consumatore li tratta come assenza di giudizio.

## Valutazione

Stesso harness e stessi set per il modello e per ogni baseline. Riportare
anche i casi in cui il modello perde.

| Set | Lingua | Metrica | Questo modello | Baseline 1 | Baseline 2 |
| --- | --- | --- | --- | --- | --- |
| | | accuratezza / F1 per classe | | | |
| | | calibrazione (ECE, Brier) | | | |
| | | over-refusal | | | |

- Revisione del modello, hash dei pesi e quantizzazione:
- Casi persi contro le baseline e analisi:

## Limiti

Lingue e registri poco coperti, dipendenza da dati tradotti, classi rare,
sensibilità alla formulazione del criterio, stabilità tra quantizzazioni.

## Ricetta di training

- Base e revisione:
- Metodo (LoRA / QLoRA, rank, target), iperparametri:
- Hardware e durata (Colab A100):
- Versioni: Unsloth, torch, transformers, trl, peft:
- Export GGUF: versione di llama.cpp, quantizzazioni, digest:

## Attribuzioni

Attribuzioni richieste dalle licenze dei dataset (CC-BY, ODC-BY), riprese dalle
data card.

## Licenza

Apache-2.0.
