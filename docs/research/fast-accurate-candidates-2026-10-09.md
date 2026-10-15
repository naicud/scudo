# Candidati Fast e Accurate con licenza aperta

> Origine: nota di ricerca del repository privato `open-guard`, copiata qui.
> I numeri sono dichiarati dalle fonti, non misure di Scudo.

Ricerca del **9 ottobre 2026**, a monte dello screen dei candidati. Base:
[ADR 0001](../adr/0001-tier-tipizzato-ragionante-con-escalation.md) e glossario
`CONTEXT.md`. Il documento non contiene benchmark del progetto: ogni numero è
**dichiarato dalla fonte** e va riprodotto sul nostro corpus IT/EN prima di
qualunque scelta.

## Metodo e convenzioni

Licenza, parametri, lingue e presenza di GGUF sono stati letti il 9 ottobre 2026
dall'API Hugging Face (`cardData.license`, `safetensors.total`, `gguf.total`,
tag di lingua) e dal README grezzo del modello. I parametri sono il totale
`safetensors.total` (può includere torre visiva o embedding). Le metriche sono
quelle pubblicate dagli autori, con il loro hardware e i loro dataset; non sono
confrontabili tra famiglie. **NON VERIFICATO** indica un'affermazione che non ho
potuto confermare su fonte primaria: non va usata come presupposto.

## Scopo e vincolo di licenza

Scegliere i candidati per il [Fast][adr] (decisione tipizzata letta sugli score
delle opzioni, senza generazione) e l'Accurate (LLM piccolo che ragiona e poi
espone lo score del verdetto tri-state). L'ADR ammette solo pesi con licenza
aperta senza limiti d'uso commerciale (Apache-2.0, MIT o equivalenti): restano
fuori soglie di fatturato o utenti e termini d'uso del produttore. Tutti i
candidati sotto risultano Apache-2.0 sul tag di licenza della repository.
Questo non sostituisce la verifica legale dei dati di training: la model card di
Rizzo Flow segnala nel training set configurazioni CC-BY-SA e CC-BY-4.0
([Rizzo Flow][rizzo], sezione *Licenses of the training data*).

## Candidati Fast

Colonne: id; parametri; licenza; criteri personalizzati; uscita e lettura degli
score; italiano; GGUF; numeri dichiarati; rischi.

| Candidato | Param. | Licenza | Criteri custom | Uscita e lettura | Italiano | GGUF | Numeri dichiarati | Rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rizzoaiacademy/rizzo-flow` (LoRA fusa su `XHToken/Spark-X2.5-4B`) [rizzo] | 4,11 B | Apache-2.0 | sì, domande tipizzate (`boolean`, `choice`, `score`) | softmax sui logit delle lettere di risposta in un forward pass, nessun testo generato | tag `en`; nessuna valutazione IT nella card; la nota «English is strongest» è NON VERIFICATO | ufficiale: Q8_0 4,4 GB, Q4_K_M 2,6 GB, BF16 8,2 GB; safetensors per MLX | typed-decisions 0,648 (base 0,574; Jev 1.13.0 0,727 da dataset card di terzi); ECE 0,112 | probabilità non calibrate (la card lo dichiara); prompt fisso `spark-decisions-v3`; testato su llama.cpp b11081 e GPU NVIDIA, Mac non misurato |
| `ibm-granite/granite-4.1-3b` + adapter *policy-guardrails* in `ibm-granite/granitelib-guardian-r1.0` [gpol][g41] | 3,40 B base + LoRA r16 | Apache-2.0 (base e adapter) | sì: policy + scenario | JSON `{"score":"Yes"\|"No"\|"Ambiguous"}` generato; `Yes` = conforme, quindi `No` ≈ VIOLATION. Lo score sul token dopo prefisso JSON forzato è **una nostra ipotesi**, non dichiarata | base: IT dichiarato [g41]; adapter: tag `en`, IT non valutato | base: `ibm-granite/granite-4.1-3b-GGUF` ufficiale; uso del LoRA su llama.cpp NON VERIFICATO | split OOD per policy, balanced accuracy 0,937 (LoRA), 0,948 (aLoRA), 0,825 (zero-shot), dataset proprietario IBM | dataset non pubblico; esempi con runtime Mellea o PEFT; target `Ambiguous` ≠ `INSUFFICIENT` da dimostrare |
| `jaredpalmer/kev-4b` [kev] | NON VERIFICATO (`safetensors` assente nell'API) | Apache-2.0 | sì (testa a puntatore sulle opzioni) | prefill-only, nessuna generazione | solo inglese | `ggml-org/kev-4b-GGUF` esiste | NON VERIFICATO | testa a puntatore: compatibilità col backend GGUF nativo da provare |
| `Mapika/decider-4b` [decider] | 4,21 B | Apache-2.0 | sì | LM che non genera testo, legge stato e domande tipizzate | «English is the main language» | `Mapika/decider-4b-GGUF` esiste | Decision Index 40,70 (riportato da [d1]) | inglese |
| `apus-ailab/APUS-OpenJev-v1-4B` [apus] | 4,54 B | Apache-2.0 | giudizi su principi in linguaggio naturale | decisioni tipizzate, probabilità | tag `en`, `zh` | GGUF e MLX ufficiali | Frozen80 66/80 (BF16), 67/80 (Q8_0) | non italiano; panel piccolo (80 item) |
| `ibm-granite/granite-guardian-3.2-3b-a800m` | 3,30 B (MoE) | Apache-2.0 | NON VERIFICATO (card non riletta) | NON VERIFICATO | NON VERIFICATO | NON VERIFICATO | n/d | generazione precedente, scartato dalla shortlist |
| `convaiinnovations/laya-multilingual` | 0,32 B | Apache-2.0 | NON VERIFICATO | NON VERIFICATO | molte lingue nei tag, incluso `it`; valutazione IT NON VERIFICATO | NON VERIFICATO | MASSIVE 0,366 solo dal rapporto grezzo non ricontrollato | debole, fuori shortlist |
| `tasksource/ModernBERT-large-nli` | 0,40 B | Apache-2.0 | ipotesi NLI | entailment/neutral/contradiction | tag `en` | n/a (encoder) | NON VERIFICATO | solo baseline di confronto, non è un Fast tipizzato |

`ZefanCai/Open-Jev` è un **dataset** (training), non un modello pronto; i modelli
`ZefanCai/Open-Jev-2B` (base `Qwen/Qwen3.5-2B`, Apache-2.0) richiedono un loader
dedicato [openjev] e non sono servibili dal backend GGUF nativo. Il dataset
Open-Jev (sintetico, `en`/`zh`/`tr`, licenza CC0 dichiarata da verificare) è un
candidato per il fine-tune Unsloth del Fast. Tra i dataset:
`tasksource/procedural-typed-decisions` (training) e `Praveenrajus/jev-bench`
(benchmark) esistono e rispondono 200.

## Candidati Accurate

| Candidato | Param. | Licenza | Criteri custom | Uscita | Italiano | GGUF | Numeri dichiarati | Rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` [q35] | 9,65 B | Apache-2.0 | prompt libero | testo con `<think>` di default; verdetto e score da progettare | «201 lingue e dialetti» (dichiarato, non valutato per il nostro task) | solo non ufficiale: `unsloth/Qwen3.5-9B-GGUF` | 262.144 token di contesto; metriche di task nella card | modello generale: tri-state e readout da calibrare |
| `Qwen/Qwen3.5-4B` | 4,66 B | Apache-2.0 | prompt libero | come sopra | come sopra | solo non ufficiale: `unsloth/Qwen3.5-4B-GGUF` | n/d | come sopra, capacità di ragionamento minore |
| `ibm-granite/granite-guardian-4.1-8b` [gg41] | 8,38 B | Apache-2.0 | sì, *Bring Your Own Criteria* | `<score>yes\|no</score>`, hybrid thinking (`<think>`/`<no-think>`); binario, manca `INSUFFICIENT` | tag `en`; la base Granite 4.1 dichiara IT [g41], il Guardian non lo valuta | `ibm-granite/granite-guardian-4.1-8b-GGUF` ufficiale | IFEval multi-constraint BAcc 0,844 no-think (0,458 precedente) | uscita binaria: `INSUFFICIENT` va derivato dal margine |
| `prism-ml/Bonsai-8B-gguf` [bon8] | 8,19 B (Qwen3-8B dense) | Apache-2.0 | prompt libero (modello generale) | testo | non dichiarato | ufficiale, Q1_0, 1,15 GB | 70,5 di media su 6 categorie (autori) | richiede il fork llama.cpp di PrismML; 1-bit: qualità sul tri-state da misurare |
| `prism-ml/Ternary-Bonsai-2-27B-gguf` [bon27] | 27,36 B (base `Qwen/Qwen3.8-27B`) | Apache-2.0 | prompt libero | testo | non dichiarato | ufficiale, PTQ1_0 5,95 GB, PQ2_0 7,21 GB | math 96,57, coding 89,42 (autori) | fork llama.cpp/MLX di PrismML; 27 B supera lo scopo di un Accurate «piccolo» |
| `openai/gpt-oss-safeguard-20b` [safeg] | 21,5 B (3,6 B attivi) | Apache-2.0 | sì, policy nel prompt | ragionamento poi verdetto, solo formato *harmony* | non dichiarato | `unsloth/gpt-oss-safeguard-20b-GGUF` non ufficiale | la card dichiara 16 GB di VRAM | tenere Fast e Accurate sul Mac 24 GB è NON VERIFICATO; probabile candidato di confronto, non di default |

## Esclusi per licenza o fit

| Modello | Motivo, verificato sulla fonte |
| --- | --- |
| `LiquidAI/d1-3B` [d1] | Licenza LFM 1.0 (`license: other`). Il testo definisce una soglia («Threshold») di 10 milioni di USD di ricavi annui. Escluso da ADR 0001, benché sia il miglior punteggio Decision Index dichiarato (48,57) e abbia GGUF ufficiale e IT dichiarato. |
| `google/shieldgemma-2b` | Licenza `gemma`, repository gated. |
| `meta-llama/Llama-Guard-4-12B`, `meta-llama/Llama-Prompt-Guard-2-86M` | Licenza `llama4` (`other`), gated. |
| `nvidia/Nemotron-3.5-Content-Safety` | `openmdw-license-agreement` (`other`), architettura `gemma3`: termini del produttore. |
| `nvidia/Nemotron-Content-Safety-Reasoning-4B` | `nvidia-open-model-license`, base `google/gemma-3-4b-it`. |
| `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` | `nvidia-open-model-license`, base Llama 3.1. |
| `Qwen/Qwen3Guard-Gen-4B` (Apache-2.0) | Licenza idonea; la tassonomia fissa a 9 categorie è NON VERIFICATO sulla card (dal rapporto grezzo). Scartato come sensore di criteri personalizzati. |
| `allenai/wildguard`, `tomg-group-umd/DynaGuard-4B` | Apache-2.0, tag solo inglese. Tassonomia fissa e uscita binaria PASS/FAIL sono NON VERIFICATO. DynaGuard è un possibile confronto per criteri dinamici. |

Cloudflare «Clef» (9B/27B) citato nel rapporto grezzo non è stato valutato.

## Shortlist

**Fast**

1. **Granite 4.1 3B + policy-guardrails** (Apache-2.0 verificata su base e
   adapter): unico con output già tri-state e criteri in linguaggio naturale; IT
   dichiarato solo sulla base. Da provare: LoRA su GGUF, lettura dello score.
2. **Rizzo Flow 4B** (Apache-2.0): è il caso d'uso del Fast (score su lettere,
   zero generazione, GGUF e MLX ufficiali). Da provare: IT, calibrazione,
   astensione, esecuzione su Metal.
3. **APUS-OpenJev-v1-4B** (Apache-2.0, GGUF e MLX ufficiali): Fast della
   famiglia Jev con decisioni tipizzate. Lingue dichiarate `en`/`zh`: entra
   nello screen perché il report per candidato misura l'accuratezza IT, che
   decide se resta. Da provare: lettura degli score sul backend nativo.
4. **Kev-4B** e **Decider-4B** (Apache-2.0, GGUF presenti, solo inglese):
   riserve per la stessa famiglia di readout; non entrano nello screen senza
   evidenza IT.

**Accurate**

1. **Qwen3.5 4B e 9B** (Apache-2.0, 201 lingue dichiarate): baseline ragionanti;
   GGUF non ufficiali, quindi da pinzare per hash.
2. **Granite Guardian 4.1 8B** (Apache-2.0, GGUF ufficiale, modalità think/no-think,
   BYOC): giudice con criteri propri, ma binario.
3. **Bonsai** (Apache-2.0): 8B 1-bit come candidato di footprint; il 27B ternario
   come confronto di qualità, entrambi con fork llama.cpp.
4. **gpt-oss-safeguard-20b** (Apache-2.0): solo se la memoria lo permette con il
   Fast residente. 21,5 B in safetensors; la card dichiara 16 GB di VRAM, non la
   compatibilità con 24 GB condivisi e un secondo modello.

## Rischi aperti

- **Italiano**: nessun candidato Fast ha una valutazione italiana pubblicata nelle
  fonti lette. Granite base e Qwen3.5 lo dichiarano, non lo misurano sul task.
- **Readout tri-state**: `Ambiguous` (Granite) e l'astensione (Rizzo) non sono
  `INSUFFICIENT` finché non sono mappati e calibrati sul nostro corpus.
- **Runtime unico GGUF**: ADR 0001 prevede il solo backend nativo. LoRA Granite su
  GGUF, fork PrismML e architettura `spark2_5` di Rizzo sono da provare sulla
  versione di llama.cpp scelta; Rizzo è legato a b11081 dalla sua card.
- **Memoria Mac 24 GB**: i GGUF dichiarati (Rizzo Q8 4,4 GB; Bonsai-8B 1,15 GB;
  Qwen3.5-9B quantizzato) sono dimensioni dei file, non picchi con contesto e KV.
- **Provenienza**: dataset di training proprietario (Granite), benchmark con gold da
  teacher ~4B (Rizzo, per la sua card), GGUF di terzi (Qwen, safeguard).
- **Numeri**: metriche diverse e dataset diversi; nessun confronto diretto tra Rizzo
  0,648, Granite 0,937 e Decision Index 48,57.

## Cosa deve misurare lo screen

Sullo stesso corpus IT/EN/misto e lo stesso harness, per candidato e quantizzazione:
accuratezza e F1 per `VIOLATION`/`NO_VIOLATION`/`INSUFFICIENT`, per lingua;
calibrazione (ECE, Brier) e margine sulla soglia; qualità del criterio di
difficoltà (recall dei casi che richiedono l'Accurate); latenza p50/p95 e memoria
di picco su Mac 24 GB con Fast e Accurate caricati insieme; stabilità tra
quantizzazioni; comportamento su timeout e output malformato (esito REVIEW, mai
ALLOW). Per l'Accurate: accuratezza dopo il ragionamento, lettura dello score sul
verdetto e costo in token. Ogni risultato riporta SHA, hash dei pesi, versione di
llama.cpp e il tipo di prova (locale o live).

## Fonti

Letti il 9 ottobre 2026. Gli id esatti sono nelle tabelle.

[adr]: ../adr/0001-tier-tipizzato-ragionante-con-escalation.md
[rizzo]: https://huggingface.co/rizzoaiacademy/rizzo-flow
[gpol]: https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/blob/main/policy-guardrails/README.md
[g41]: https://huggingface.co/ibm-granite/granite-4.1-3b
[gg41]: https://huggingface.co/ibm-granite/granite-guardian-4.1-8b
[kev]: https://huggingface.co/jaredpalmer/kev-4b
[decider]: https://huggingface.co/Mapika/decider-4b
[apus]: https://huggingface.co/apus-ailab/APUS-OpenJev-v1-4B
[openjev]: https://huggingface.co/ZefanCai/Open-Jev-2B
[q35]: https://huggingface.co/Qwen/Qwen3.5-9B
[bon8]: https://huggingface.co/prism-ml/Bonsai-8B-gguf
[bon27]: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
[safeg]: https://huggingface.co/openai/gpt-oss-safeguard-20b
[d1]: https://huggingface.co/LiquidAI/d1-3B
