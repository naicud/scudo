# Modelli base per Fast e Accurate

Ricerca del **15 ottobre 2026**. Domanda: da quali pesi base partire per il
[Fast](../../CONTEXT.md) (decision-token letto sui logit di `A`/`B`/`C`, nessuna
generazione; oggi `Qwen/Qwen3.5-4B`) e per l'Accurate (ragionamento, poi logit
delle opzioni; oggi `Qwen/Qwen3.5-9B`), e come deciderlo con numeri nostri.
Aggiorna lo screen del 9 ottobre
([fast-accurate-candidates-2026-10-09.md](fast-accurate-candidates-2026-10-09.md)),
che confrontava checkpoint pronti; qui si confrontano **basi da addestrare** con
la ricetta del tracer (#2).

## Metodo e convenzioni

- Metadati letti il 15 ottobre da `https://huggingface.co/api/models/<id>/revision/main`
  (`sha`, `createdAt`, `cardData.license`, `safetensors.total`, `config.architectures`,
  `gated`), dal README grezzo e dal file `LICENSE` alla stessa revisione. I
  parametri sono `safetensors.total` (includono torre visiva ed embedding).
- Nomi dei moduli LoRA: chiavi dell'indice dei pesi (`model.safetensors.index.json`)
  o dell'header safetensors, alla revisione indicata.
- llama.cpp: sorgente letto con `gh api` a `master` = `620d2225cc436ab0b001e3200e590e3b286ef9ab`
  (2026-10-15) e alla revisione fissata dal backend nativo di open-guard
  `631109b34da437a3c4a5ebd75091d677671392e3` (2026-10-02,
  `src/open_guard/native/CMakeLists.txt`, `OG_LLAMA_REVISION`).
- Unsloth: documentazione ufficiale via Context7 (`/websites/unsloth_ai`).
- Tokenizer: scaricato solo `tokenizer.json` alla revisione indicata e provato
  con `tokenizers` (script nella sezione *Tokenizer*).
- **Ogni numero di benchmark è dichiarato dal produttore** nella sua model card,
  con il suo harness e la sua modalità (spesso *thinking*): non sono confrontabili
  fra famiglie e non sono misure di Scudo. **NON VERIFICATO** segna ciò che non ho
  confermato su fonte primaria.

## «Qwen 3.8» esiste?

Sì, ma **non in taglie 1–14B**. Dall'elenco dei modelli dell'organizzazione `Qwen`
(`/api/models?author=Qwen&sort=createdAt`), i pesi Qwen3.8 ufficiali sono:

| Repo | Revisione | Creato su HF | Parametri | Architettura | Licenza |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 2026-08-05 | 27,78 B (densa, VL) | `Qwen3_5ForConditionalGeneration` (`qwen3_5`) | `apache-2.0`, `LICENSE` = testo Apache 2.0 |
| `Qwen/Qwen3.8-27B-FP8` | — | 2026-08-13 | come sopra, FP8 | — | — |
| `Qwen/Qwen3.8-2.4T-A95B` (+ `-FP8`) | `207bd685a7e3696cfaff12ded7c6a7ea0f88c996` | 2026-08-08 | 2.446 B, MoE A95B | `Qwen3_5MoeForCausalLM` | `other` / `qwen3.8-max` |
| `Qwen/Qwen3.8-Flash-Next` (+ `-FP8`) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | 2026-08-24 | 180 B | `Qwen4ExpForConditionalGeneration` (`qwen4_exp`) | `other` / `qwen-community-1.0` |

Dalla card di Qwen3.8-27B: «Built on the architectural foundation of Qwen3.5»;
64 layer con layout `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`;
contesto 262.144 nativo, estendibile a 1.000.000; thinking attivo di default.
Il blog `qwen.ai/blog?id=qwen3.8` citato nella card è reso in JavaScript e non
l'ho potuto leggere: eventuali annunci di taglie piccole sono **NON VERIFICATO**.

- **Qwen3.7**: nessun peso aperto nell'organizzazione `Qwen`. «Qwen3.7-Plus»
  compare solo come colonna di confronto nella tabella della card di Qwen3.8-27B
  (modello ospitato).
- **Qwen3.6**: `Qwen/Qwen3.6-27B` (2026-04-21) e `Qwen/Qwen3.6-35B-A3B`
  (2026-04-15), Apache-2.0; nessuna taglia piccola.
- Le taglie Qwen ufficiali più recenti nella fascia 1–14B restano quindi
  **Qwen3.5** 0.8B/2B/4B/9B (febbraio 2026).
- `empero-ai/Qwen3.8-9B-Distill` e `-4B-Distill` (tag Apache-2.0) **non sono
  Qwen**: sono SFT di Qwen3.5-9B/4B su «~70,000 curated teacher traces from our
  internal Qwen3.8 distillation datasets», teacher «Qwen3.8 2.4T A95B». Esclusi:
  dati interni non dichiarati e teacher con licenza Qwen3.8-Max (sotto).

Per Scudo, Qwen3.8-27B non entra nel profilo Mac 24 GB insieme al Fast: il GGUF
`ggml-org/Qwen3.8-27B-GGUF` (`71bc7b62…`) pesa 18,97 GB in Q4_K_M e 28,60 GB in
Q8_0, più 4,48 GB del Fast Q8_0. È invece un **teacher** idoneo (Apache-2.0,
nessuna soglia) per tracce di ragionamento e casi IT, accanto a quelli del pilot.

## Novità che cambiano lo screen

1. **Gemma 4 è Apache-2.0.** I repository `google/gemma-4-*` hanno
   `license: apache-2.0`, non sono gated, e `license_link`
   (`https://ai.google.dev/gemma/docs/gemma_4_license`) reindirizza a
   `https://ai.google.dev/gemma/apache_2`, intitolata «Apache License 2.0», con il
   testo della licenza. L'ADR 0001 esclude i «termini d'uso del produttore (… Gemma
   …)»: valeva per i Gemma Terms of Use delle generazioni 1–3. Sul criterio
   dell'ADR (Apache-2.0, nessun limite d'uso) Gemma 4 è idoneo; il testo dell'ADR
   nomina però la famiglia, quindi **serve una conferma esplicita del
   proprietario** prima di usarlo come base pubblicata.
2. **Mistral Shieldstral-1.0-3B** (`mistralai/Shieldstral-1.0-3B`, 2026-07-16,
   Apache-2.0, base `Ministral-3-3B-Base-2512`): «policy-adaptive multimodal safety
   classifier», criteri in linguaggio naturale, «single yes/no forward pass»,
   italiano fra le 12 lingue. È la prova più vicina al nostro Fast che una base
   Ministral 3 da 3B regge il compito. Resta una **baseline** (ADR 0002): la card
   dice solo «synthetic and public safety data», senza elenco dei dataset (il
   report arXiv 2607.25857 non l'ho letto: NON VERIFICATO).
3. **IBM Granite 4.2** (3B, 8B, 30B; card: «Release Date August 25, 2026»),
   trasformatore denso `GraniteForCausalLM` con ragionamento nativo, italiano fra
   le lingue testate.
4. Il backend nativo di open-guard compila llama.cpp **solo CPU**: nel
   `CMakeLists.txt` `GGML_METAL`, `GGML_ACCELERATE` e `GGML_BLAS` sono forzati a
   `OFF`. La latenza della generazione dell'Accurate sul Mac va misurata in
   queste condizioni; per il Fast (solo prefill) pesa meno.

## Tabella dei candidati

Licenza = `cardData.license` alla revisione. «llama.cpp» = architettura
registrata nel converter a `master` e presente in `src/llama-arch.cpp` alla
revisione di open-guard. GGUF = dimensione del file di terzi o ufficiale indicato
(solo pesi del modello testuale, senza `mmproj`).

| Candidato (revisione) | Licenza | Param. | Contesto | Arch. / llama.cpp | Unsloth | Italiano (dichiarato) | GGUF Q8_0 / Q4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) | apache-2.0 | 4,66 B | 262K | `qwen35` ibrido Gated DeltaNet; `conversion/qwen.py:665` | sì (in uso, #2) | MMMLU 76,1, MMLU-ProX 71,5, INCLUDE 71,0 [q35-9b] | 4,48 / 2,74 (Q4_K_M) [ug4] |
| `Qwen/Qwen3.5-2B` (`15852e8c16360a2fea060d615a32b45270f8a8fc`) | apache-2.0 | 2,27 B | 262K | come sopra | sì | MMMLU 63,1, MMLU-ProX 52,3, INCLUDE 55,4 (thinking) [q35-2b] | 2,01 / 1,28 |
| `Qwen/Qwen3.5-9B` (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`) | apache-2.0 | 9,65 B | 262K | come sopra | sì (in uso, #9) | MMMLU 81,2, MMLU-ProX 76,3, INCLUDE 75,6 [q35-9b] | 9,53 / 5,68 |
| `Qwen/Qwen3.8-27B` (`1d4bf0f2…`) | apache-2.0 | 27,78 B | 262K | `qwen35`, stesso converter | sì, QLoRA 4-bit [u-q38] | nessuna tabella multilingue nella card | 28,60 / 18,97 |
| `google/gemma-4-E4B-it` (`ee0ef6023621cff504d758262d4e04895a5af4a2`) | apache-2.0 | 8,00 B (4,5 B «effective» + embedding per-layer) | 128K | `gemma4`; `conversion/gemma.py:632` | sì, LoRA E4B 17 GB VRAM [u-g4] | MMMLU 76,6 [g4] | 8,03 / 4,59 (Q4_0) [gg-e4b] |
| `google/gemma-4-12B-it` (`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`) | apache-2.0 | 11,96 B | 256K | `gemma4` unified; `conversion/gemma.py:934` | sì [u-g4] | MMMLU 83,4, MMLU Pro 77,2 [g4] | 12,67 / 7,22 (Q4_0) |
| `google/gemma-4-26B-A4B-it` (`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`) | apache-2.0 | 25,8 B (3,8 B attivi) | 256K | `gemma4` MoE | sì, LoRA > 40 GB [u-g4] | MMMLU 86,3 [g4] | 26,86 / 14,62 (Q4_0) |
| `mistralai/Ministral-3-3B-Base-2512` (`6f9c4b12a95b139af68670a6713616b757923735`) | apache-2.0 | 3,85 B (BF16) | 256K | `mistral3`; `conversion/mistral3.py:14` | sì [u-m3] | Multilingual MMLU 0,652 (base) [m3] | Instruct: 3,65 / 2,15 [mg3] |
| `mistralai/Ministral-3-8B-Reasoning-2512` (`81eaece1948f3875421d9a45bc55487d10e2d894`) | apache-2.0 | 8,92 B (BF16) | 256K | `mistral3` | sì [u-m3] | Multilingual MMLU 0,706 (base 8B) [m3] | Instruct: 9,03 / 5,20 |
| `ibm-granite/granite-4.2-3b` (`e459acceac81e5fe67c07d9cfc72329a332e7eb1`) | apache-2.0 | 3,66 B | 128K | `granite` denso GQA; `conversion/granite.py:17` | 4.1 sì [u-gr]; 4.2 stessa arch., NON VERIFICATO | IT «tested»; MMLU-ProX lite 27,78 [g42] | 3,89 / 2,24 [gg42] |
| `ibm-granite/granite-4.2-8b` (`f8de16cdcdbc6c779ca517604e050d82cc119e44`) | apache-2.0 | 8,79 B | 128K (512K esteso) | `granite` | come sopra | IT «tested»; MMLU-ProX lite 61,06, MMLU-Pro 74,04 [g42] | 9,35 / 5,35 |
| `openai/gpt-oss-20b` (`6cee5e81…`) | apache-2.0 (+ usage policy: solo «comply with all applicable law») | 20,9 B (MoE) | — | `gpt-oss`; `conversion/gpt_oss.py:13` | sì (noto, non riletto) | nessuna valutazione IT nella card | MXFP4 12,11 |
| `microsoft/Phi-4-mini-instruct` (`cfbefacb…`) | mit | 3,84 B | — | `phi3` | — | 24 lingue nei tag | non letto |
| `HuggingFaceTB/SmolLM3-3B` (`a07cc9a0…`) | apache-2.0 | 3,08 B | 64K (128K YaRN) | `smollm3` | — | IT supportato; Global MMLU 53,5 vs 65,1 di Qwen3-4B, no-think [smol3] | non letto |
| `allenai/Olmo-3-7B-Instruct` (`6e5971d9…`) | apache-2.0 | 7,30 B | — | `olmo2` | — | tag solo `en` | non letto |
| `sapienzanlp/Minerva-7B-instruct-v1.0` (`d1fc0f0e…`) | apache-2.0 | 7,40 B | — | `llama` (Mistral) | — | pre-training ~metà italiano; in Evalita-LLM | non letto |
| `utter-project/EuroLLM-9B-Instruct-2512` (`def82454…`) | apache-2.0 | 9,15 B | — | `llama` | — | IT fra 35 lingue | non letto |
| `AlmawaveLabs/Velvet-14B` / `-2B` | apache-2.0, gated (`auto`, solo privacy policy) | 14,08 / 2,22 B | — | `llama` (Mistral) | — | IT/EN; card non leggibile senza accettare il gate: NON VERIFICATO | non letto |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` / `-14B`, `DeepSeek-R1-0528-Qwen3-8B` | mit (basi Qwen2.5/Qwen3 Apache-2.0) | 7,6 / 14,8 / 8,2 B | — | `qwen2` / `qwen3` | — | nessuna IT | non letto |

Valutazioni italiane indipendenti: il dataset dei risultati di Evalita-LLM
(`evalitahf/evalita_llm_results`, ultimo aggiornamento 2025-12-07) contiene
Qwen2.5, Qwen3-Next, Gemma 2/3/3n, Granite 3.1, Phi-3/4, Minerva, Velvet, ma
**nessuno dei candidati 2026** sopra. Per i candidati principali l'unica
evidenza italiana sono i benchmark multilingue dei produttori (MMMLU,
MMLU-ProX, INCLUDE, Global MMLU), che coprono l'italiano fra molte lingue ma non
lo riportano separatamente, salvo SmolLM3. Per questo la decisione passa dal
bake-off.

Guard pubblicati sulle basi (rilevanza per il compito, numeri dei produttori):

- Qwen3.5-4B → `h2oai/h2o-lightning-4b` (decisioni tipizzate, un forward pass;
  la card dichiara il #1 della JevBench Composite open-weights dal 7 al 15
  ottobre, 72,5) e `microsoft/FrogNano-4B-2609` (agente di codice, non rilevante).
- Qwen3.5-9B → `Cloudflare/clef-flash` (testa congiunta separata: non servibile
  dal backend GGUF, come la lane Clef scartata dall'ADR 0001).
- Ministral-3-3B-Base → Shieldstral-1.0-3B: F1 prompt WildGuardTest 88,1,
  Aegis v2 86,2, PolyGuard (multilingue) 84,6; risposta Aegis v2 87,2 [shield].
- Granite 4.1-8B → `ibm-granite/granite-guardian-4.1-8b` (screen del 9 ottobre).
- Gemma 4: nessun guard ufficiale trovato; solo fine-tune di terzi.

## Tokenizer

Script: `tokenizer.json` alla revisione fissata, `tokenizers.Tokenizer`,
`encode(..., add_special_tokens=False)` come il notebook e il bridge nativo
(«No BOS/EOS, special parsing», `bridge.c:329`). Prompt di prova: JSON del
decision-token con evidenza italiana e accenti, poi `\nDecision:\n`. Controlli:
`A`/`B`/`C` token singoli e distinti; `enc(prompt + lettera) == enc(prompt) + [lettera]`;
per il formato reasoned `prompt + "Reasoning:\n"` e `...Answer:\n + lettera`
come concatenazioni esatte.

| Modello (revisione) | A / B / C | Tutti i controlli |
| --- | --- | --- |
| Qwen3.5-4B, 2B, 9B (revisioni sopra) | 32 / 33 / 34 | sì |
| Qwen3.8-27B (`1d4bf0f2…`) | 32 / 33 / 34 | sì |
| gemma-4-E4B-it, gemma-4-12B-it | 236776 / 236799 / 236780 | sì |
| granite-4.2-3b, granite-4.2-8b | 32 / 33 / 34 | sì |
| Ministral-3-3B-Base, 3B-Instruct, 8B-Instruct | 1065 / 1066 / 1067 | sì |

Rischio Gemma: i Gemma sono addestrati con BOS e il protocollo non lo mette. La
LoRA vede lo stesso prompt in training e serving, quindi è coerente, ma l'effetto
sulla qualità va misurato; aggiungere il BOS cambierebbe i byte del formato
(nuova `FORMAT_VERSION` e modifica del bridge), fuori da questa nota.

## Memoria sul Mac 24 GB

Somma dei file GGUF (solo pesi; KV, contesto e processo esclusi). Il budget
indicativo per i due modelli è ~16 GB, lasciando ~8 GB a macOS, gateway e KV:
**è una nostra ipotesi da misurare**, non un dato.

| Fast + Accurate | Q8_0 + Q8_0 | Q8_0 + Q4 |
| --- | --- | --- |
| Qwen3.5-4B + Qwen3.5-9B | 4,48 + 9,53 = 14,0 GB | 4,48 + 5,68 = 10,2 GB |
| Qwen3.5-4B + gemma-4-12B-it | 4,48 + 12,67 = 17,2 GB | 4,48 + 7,22 = 11,7 GB |
| Qwen3.5-4B + granite-4.2-8b | 4,48 + 9,35 = 13,8 GB | 4,48 + 5,35 = 9,8 GB |
| gemma-4-E4B-it + gemma-4-12B-it | 8,03 + 12,67 = 20,7 GB | 8,03 + 7,22 = 15,3 GB |
| Ministral-3-3B + Qwen3.5-9B | 3,65 + 9,53 = 13,2 GB | 3,65 + 5,68 = 9,3 GB |
| Qwen3.5-4B + gemma-4-26B-A4B-it | — | 4,48 + 14,62 = 19,1 GB |
| Qwen3.5-4B + Qwen3.8-27B | — | 4,48 + 18,97 = 23,5 GB (fuori) |
| Qwen3.5-4B + Qwen3.6-35B-A3B | — | 4,48 + 22,13 = 26,6 GB (fuori) |

I GGUF Ministral sono quelli Instruct ufficiali; un nostro export dalla Base ha la
stessa architettura e dimensione attesa (NON VERIFICATO). Il GGUF del tracer
Qwen3.5-4B Q8_0 misurato in #2 è 4.610.580.000 byte.

## Esclusi

### Per licenza (clausola citata)

| Modello | Clausola |
| --- | --- |
| `Qwen/Qwen3.8-Flash-Next` | Qwen Community License 1.0, §1: «If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 … monthly revenue, respective model name must be prominently displayed …»; §2: «If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen …». |
| `Qwen/Qwen3.8-2.4T-A95B` | Qwen3.8-Max License, §2: «… and the aggregate revenue of the licensee and its affiliates exceeds US$50,000,000 … the licensee shall obtain a separate license from Qwen …» (più la stessa §1). |
| `empero-ai/Qwen3.8-9B-Distill`, `-4B-Distill` | Tag Apache-2.0, ma addestrati su «internal Qwen3.8 distillation datasets» con teacher Qwen3.8 2.4T A95B (licenza sopra): AGENTS.md esclude dati distillati da modelli la cui licenza limita l'uso, e i dati non sono dichiarati. |
| `swiss-ai/Apertus-v1.5-8B` | Gated con termini propri: «I confirm I have read and agree to the Apache 2․0 license and the Acceptable Use Policy (AUP)» (`extra_gated_prompt`). |
| `LiquidAI/LFM2.5-*` (es. `LFM2.5-2.6B`, `LFM2.5-24B-A2B`) | LFM License: «"Threshold" shall mean annual revenue of 10 million United States dollars ($10,000,000) or more»; «Any Commercial Use … by a Legal Entity that exceeds the Threshold is not licensed under this Agreement.» |
| Llama 3.x (es. `meta-llama/Llama-3.2-3B-Instruct`) e derivati (LLaMAntino) | Llama 3.2 License §2: «If … the monthly active users … is greater than 700 million monthly active users … you must request a license from Meta». |
| NVIDIA Nemotron (NVIDIA Open Model License) | «If You bypass, disable, reduce the efficacy of, or circumvent any … Guardrail … your rights under this Agreement will automatically terminate» e «Use of the Models … must be consistent with NVIDIA's Trustworthy AI terms». |
| `amazon/ALoDLM-8B` | `cc-by-nc-4.0` (non commerciale). |

### Per idoneità tecnica o capacità

- `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.6-35B-A3B`: fuori budget di memoria con il
  Fast (tabella sopra). Qwen3.8-27B resta candidato teacher.
- `Cloudflare/clef`, `clef-flash`: testa congiunta separata, non servibile dal
  backend GGUF nativo.
- `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (MIT, base Qwen3.5-9B): SFT agentico su
  «MiMo-generated data»; la card dà solo la ripartizione dei token per dominio,
  non i dataset (ADR 0002: base di terzi solo con dati dichiarati).
- `h2oai/h2o-lightning-4b`, Shieldstral, Rizzo Flow, Granite Guardian: baseline,
  non basi (ADR 0002).
- SmolLM3-3B: la sua card lo pone sotto Qwen3-4B (generazione precedente) su
  Global MMLU; Qwen3.5-4B lo domina.
- Olmo 3 7B: tag solo `en`. Phi-4-mini (febbraio 2025) e distill DeepSeek R1
  (Qwen2.5/Qwen3): generazioni precedenti, nessuna evidenza IT che li porti
  sopra Qwen3.5; restano fuori senza bake-off.
- Minerva-7B, EuroLLM-9B-2512, Velvet: le uniche basi con forte focus italiano
  (Minerva ed EuroLLM Apache-2.0 senza gate), ma di generazione o capacità
  generale inferiore, senza ragionamento nativo; Velvet ha la card dietro gate
  (NON VERIFICATO). Riserve se il bake-off mostra un divario IT/EN che le basi
  multilingui non chiudono.
- `openbmb/MiniCPM5-2B`: tag `en`/`zh`. `Aleph-Alpha/Kolibri-1`: 78 B, `de`/`en`.
- `gpt-oss-20b`: licenza idonea, entra in memoria (12,11 GB MXFP4), ma formato
  harmony e nessuna evidenza IT; resta un confronto, come
  `gpt-oss-safeguard-20b` nello screen del 9 ottobre.

## Shortlist

### Fast

1. **`Qwen/Qwen3.5-4B`** (controllo): miglior evidenza multilingue dichiarata fra
   le basi 3–4B (MMMLU 76,1, INCLUDE 71,0), tracer già misurato (0,8775
   complessiva, IT 0,875, EN 0,880, ECE 0,034 su 400 casi held-out, #2), tokenizer e
   GGUF già verificati, e un modello di decisioni tipizzate di terzi costruito sopra
   (h2o-lightning-4b). Costo: layer linear-attention lenti senza `causal-conv1d`.
2. **`google/gemma-4-E4B-it`**: MMMLU dichiarato 76,6, pari a Qwen3.5-4B; attenzione
   standard (kernel più semplici su CPU e in training); Apache-2.0 da confermare
   rispetto al testo dell'ADR. Costi: 8,03 GB in Q8_0 per gli embedding per-layer
   (quasi il doppio di Qwen3.5-4B) e prompt senza BOS.
3. **`mistralai/Ministral-3-3B-Base-2512`**: la stessa base di Shieldstral, guard
   a criteri liberi con un solo forward pass e risultati multilingue dichiarati
   forti; il più leggero (3,65 GB in Q8_0). Base BF16 ufficiale (l'Instruct
   ufficiale è FP8).

Riserve a basso costo: `Qwen/Qwen3.5-2B` (latenza e memoria, ma MMMLU 63,1) e
`ibm-granite/granite-4.2-3b` (denso semplice, ma MMLU-ProX lite 27,78 nella sua
stessa card).

### Accurate

1. **`Qwen/Qwen3.5-9B`** (controllo): miglior multilingue dichiarato fra le basi
   che entrano nel budget (MMMLU 81,2, INCLUDE 75,6), frame reasoned già verificato
   su 43.281 tracce, 9,53 GB in Q8_0.
2. **`google/gemma-4-12B-it`**: MMMLU 83,4 e MMLU Pro 77,2 dichiarati, i più alti
   fra i candidati che entrano nel budget; 17,2 GB con il Fast in Q8_0, 11,7 GB
   con l'Accurate in Q4_0. Costo: 12 B densi in generazione su CPU, più lenti del 9B.
3. **`ibm-granite/granite-4.2-8b`**: ragionamento nativo, italiano testato,
   trasformatore denso GQA (il più semplice da servire), linea Granite Guardian;
   multilingue dichiarato sotto Qwen3.5-9B (MMLU-ProX lite 61,06).

Riserve: `mistralai/Ministral-3-8B-Reasoning-2512` (Apache-2.0, BF16) e
`google/gemma-4-26B-A4B-it`, che con 3,8 B attivi genera più in fretta su CPU ma
sta a 19,1 GB col Fast e richiede più di 40 GB per la LoRA (A100 80 GB).

## Bake-off

Obiettivo: scegliere con numeri nostri, a parità di ricetta, dati, seme e harness.
Ogni run è una prova **local** (checkpoint in training); il vincitore passa poi
dalla prova live nel backend nativo.

### Fase F: Fast

Per ogni candidato, il notebook del tracer (`src/scudo/notebook.py`, celle di
`notebook_cells.py`) con **la stessa ricetta**: 2.000 casi Nemotron v3 IT/EN
(`train_per_language` 1000, held-out 200 per lingua), seme 3407, LoRA r16/α16,
lr 2e-4, batch 8 × grad_accum 2, 1 epoca = 125 step, `max_seq_length` 2048,
prompt senza chat template. Poi `notebooks/eval.ipynb` con `EVAL["fast"]` puntato
alla base e all'adapter del candidato, `per_benchmark` 500, seme 3407: stessi
casi per ogni modello sui benchmark del registro (Aegis 2.0, Nemotron v3 IT/EN,
XSTest, XSTest-IT, ToxicChat, JBB, OR-Bench-hard, CantTalkAboutThis, Open-Jev).

Campi che cambiano per candidato (tutto il resto identico):

| Candidato | `base_model` / `base_revision` | `checkpoint_repo` (privato, nuovo) | `target_modules` (dai pesi alla revisione) | Loader |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B (controllo, rieseguito) | `Qwen/Qwen3.5-4B` / `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `Naicud/scudo-bakeoff-qwen35-4b` | 32 layer: 24 `linear_attn` (`in_proj_qkv`, `in_proj_z`, `out_proj`; `in_proj_a/b` esclusi come oggi) + 8 `self_attn` (`q/k/v/o_proj`); MLP `gate/up/down_proj` | `FastLanguageModel`, `text_only=True` (come oggi) |
| gemma-4-E4B-it | `google/gemma-4-E4B-it` / `ee0ef6023621cff504d758262d4e04895a5af4a2` | `Naicud/scudo-bakeoff-gemma4-e4b` | 42 layer `model.language_model.layers.N`: `q/k/v/o_proj`, `gate/up/down_proj` (`per_layer_projection`, `per_layer_input_gate` esclusi) | la doc Unsloth usa `FastModel` per Gemma 4; `text_only` su Gemma 4 NON VERIFICATO |
| Ministral-3-3B-Base | `mistralai/Ministral-3-3B-Base-2512` / `6f9c4b12a95b139af68670a6713616b757923735` | `Naicud/scudo-bakeoff-ministral3-3b` | 26 layer `language_model.model.layers.N`: `q/k/v/o_proj`, `gate/up/down_proj` | `FastLanguageModel` su `Mistral3ForConditionalGeneration` (torre Pixtral): `text_only` NON VERIFICATO |
| (riserva) Qwen3.5-2B | `Qwen/Qwen3.5-2B` / `15852e8c16360a2fea060d615a32b45270f8a8fc` | `Naicud/scudo-bakeoff-qwen35-2b` | 24 layer: 18 `linear_attn` + 6 `self_attn`, stessi nomi del 4B | come Qwen3.5-4B |
| (riserva) granite-4.2-3b | `ibm-granite/granite-4.2-3b` / `e459acceac81e5fe67c07d9cfc72329a332e7eb1` | `Naicud/scudo-bakeoff-granite42-3b` | 40 layer `model.layers.N`: `q/k/v/o_proj`, `gate/up/down_proj` | `FastLanguageModel`, solo testo |

Il `checkpoint_repo` nuovo è obbligatorio: la cella di training rifiuta un
checkpoint con un'impronta di run diversa. Le lettere si verificano già nella
cella `model` (asserzioni su token singoli e concatenazione).

Tempo A100 (**stima, non misurata**: il RUN_LOG di #2 non registra la durata):
per un 3–4B denso, ~1 M token di training in 125 step richiedono pochi minuti
di calcolo; con download della base, export GGUF q8_0 e eval held-out il notebook
del tracer dovrebbe stare in 30–45 minuti, l'eval di 9 benchmark × 500 casi
(un forward per caso) in 10–20 minuti. Qwen3.5 sarà il più lento per il ripiego
PyTorch dei layer linear-attention, Gemma 4 E4B per gli 8 B di parametri totali.
Per 3 candidati più il controllo: ~4–5 ore di A100.

### Fase A: Accurate

Notebook dell'Accurate (`src/scudo/accurate_notebook.py`, #9), stessa ricetta per
tutti (tracce Nemotron Content Safety Reasoning alla revisione fissata, LoRA bf16
r16/α16, lr 2e-4, batch 4 × grad_accum 4, 1 epoca, `max_new_tokens` 512,
`eval_per_benchmark` 100, seme 3407). Campi per candidato:

| Candidato | `base_model` / `base_revision` / `base_license` | `target_modules` |
| --- | --- | --- |
| Qwen3.5-9B (controllo) | `Qwen/Qwen3.5-9B` / `c202236235762e1c871ad0ccb60c8ee5ba337b9a` / `apache-2.0` | come oggi (32 layer: 24 `linear_attn` + 8 `self_attn`, MLP) |
| gemma-4-12B-it | `google/gemma-4-12B-it` / `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` / `apache-2.0` | 48 layer: `q/k/o_proj` ovunque, `v_proj` in 40 layer su 48 (assente negli altri 8), `gate/up/down_proj` |
| granite-4.2-8b | `ibm-granite/granite-4.2-8b` / `f8de16cdcdbc6c779ca517604e050d82cc119e44` / `apache-2.0` | 40 layer `model.layers.N`: `q/k/v/o_proj`, `gate/up/down_proj` |

Il 12B in LoRA bf16 sta su A100 80 GB (tabella Unsloth: 11 B → 29 GB, 14 B →
33 GB); su 40 GB serve batch più piccolo con lo stesso batch effettivo. Stima:
2–4 ore per candidato, dominate dalla generazione dell'eval.

### Regola di decisione

Sullo stesso campione di casi per ogni modello:

1. **Cancelli** (uno fallito esclude): zero errori di readout; per l'Accurate
   errori `REASONING_*` sotto l'1%; export GGUF q8_0 caricabile dal backend nativo
   alla revisione `631109b3…` con readout coerenti (`compare_readouts` sopra la
   tolleranza da fissare in #2, > 0,125); confermato l'uso di Gemma 4 rispetto
   all'ADR 0001, se il candidato è Gemma.
2. **Metrica primaria**: macro-F1 su `VIOLATION`/`NO_VIOLATION` (gli
   `INSUFFICIENT` entrano quando #6 li fornisce), media sui benchmark separata per
   IT ed EN, poi media delle due lingue, così l'italiano pesa quanto l'inglese con
   meno benchmark.
3. **Vincoli secondari**: tasso di over-refusal su XSTest, XSTest-IT e
   OR-Bench-hard non peggiore del controllo di oltre 3 punti; ECE non peggiore di
   oltre 0,02; divario IT–EN della macro-F1 non oltre 3 punti.
4. **Sostituzione del controllo** solo se la metrica primaria migliora di almeno
   2 punti **e** l'intervallo bootstrap al 95% della differenza appaiata (stessi
   casi) esclude lo zero. Il rumore di readout misurato in #2 (un verdetto su
   392) è sotto questa soglia. I primi due rifanno il training con un secondo
   seme prima della scelta.
5. **Mac**: a parità entro la soglia, decide il profilo Mac misurato nel backend
   nativo (CPU, Metal spento): latenza p50/p95 del Fast su 500 prompt, del
   Accurate con budget di 512 token, RSS di picco con Fast e Accurate caricati
   insieme. Un Accurate che non resta nel budget di memoria o che rende l'escalation
   inutilizzabile in latenza esce anche se vince in qualità.

Ogni risultato riporta SHA di Scudo, revisione e digest dei pesi, revisione di
llama.cpp e tipo di prova; un numero senza queste voci non entra nel confronto.

### Cosa cambia nel codice (minimo)

Oggi la base è scritta in quattro punti: `RUN` di `notebook.py` e
`fast_notebook.py`, `FAST_BASE_REVISION`/`EVAL["fast"]` di `eval_notebook.py`,
le costanti di `accurate_notebook.py`, e la lista `target_modules` delle celle
`LORA` (`notebook_cells.py`, `accurate_cells.py`), con `text_only=True` nelle
celle di caricamento. Per il bake-off bastano:

- due campi in più in `RUN` (e in `EVAL["fast"]`): `lora_target_modules` e
  `text_only`, letti dalle celle `LORA` e `model` al posto dei valori scritti;
  entrano nell'impronta del run, quindi un checkpoint non si riprende con un'altra
  base;
- la scelta del loader (`FastLanguageModel` o `FastModel`) se Gemma 4 lo richiede,
  come terzo campo;
- i valori della tabella sopra, impostati nel notebook al momento del run (o con
  un argomento del generatore), senza un registro di basi finché il bake-off non
  sceglie.

Nessun cambiamento al protocollo, al runner o al report: `letter_ids`,
`fetch_verified` e `build_report` sono già indipendenti dalla base. Utile ma non
necessario: registrare `train_runtime` delle metriche di `trainer.train()` nel
RUN_LOG per sostituire le stime di tempo con misure.

## Rischi aperti

- Nessuna valutazione italiana indipendente dei candidati 2026: le uniche cifre
  sono dei produttori, su benchmark multilingui aggregati.
- Gemma 4 contro il testo dell'ADR 0001; BOS assente nel protocollo.
- Loader Unsloth con `text_only` su Gemma 4 e Ministral 3: NON VERIFICATO, la
  cella fallisce in modo esplicito se non regge.
- Backend nativo solo CPU: la latenza dell'Accurate su Mac potrebbe decidere più
  della qualità.
- Shieldstral: dati di training non elencati nella card; il report tecnico non è
  stato letto.

## Fonti

Lette il 15 ottobre 2026.

- [q35-9b] https://huggingface.co/Qwen/Qwen3.5-9B (tabella *Language*, sezione *Multilingualism*; colonne Qwen3.5-9B e Qwen3.5-4B)
- [q35-2b] https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md (*Multilingualism (Thinking)*)
- https://huggingface.co/Qwen/Qwen3.8-27B, https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE, https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/LICENSE
- https://huggingface.co/empero-ai/Qwen3.8-9B-Distill
- [g4] https://huggingface.co/google/gemma-4-E4B-it (tabella dei benchmark, colonne 31B/26B A4B/12B/E4B/E2B); https://ai.google.dev/gemma/docs/gemma_4_license → https://ai.google.dev/gemma/apache_2
- [gg-e4b] https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF, https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF, https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF
- [m3] https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (tabelle *Benchmark Results*); [mg3] https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF, https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF
- [shield] https://huggingface.co/mistralai/Shieldstral-1.0-3B
- [g42] https://huggingface.co/ibm-granite/granite-4.2-8b (tabella dei benchmark 3B/8B/30B); [gg42] https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF, https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF
- [ug4] https://huggingface.co/unsloth/Qwen3.5-4B-GGUF, https://huggingface.co/unsloth/Qwen3.5-9B-GGUF, https://huggingface.co/unsloth/Qwen3.5-2B-GGUF, https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF, https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF, https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
- [smol3] https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://huggingface.co/h2oai/h2o-lightning-4b, https://huggingface.co/Cloudflare/clef-flash, https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
- https://huggingface.co/datasets/evalitahf/evalita_llm_results
- llama.cpp: `conversion/qwen.py`, `gemma.py`, `granite.py`, `mistral3.py`, `phi.py`, `olmo.py`, `llama.py`, `gpt_oss.py` a `620d2225cc436ab0b001e3200e590e3b286ef9ab`; `src/llama-arch.cpp` a `631109b34da437a3c4a5ebd75091d677671392e3`
- Unsloth (Context7, `/websites/unsloth_ai`): [u-g4] https://unsloth.ai/docs/models/gemma-4/train; [u-q38] https://unsloth.ai/docs/models/qwen3.8/train; [u-m3] https://unsloth.ai/docs/models/tutorials/ministral-3; [u-gr] https://unsloth.ai/docs/models/ibm-granite-4.1; requisiti VRAM https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- Licenze escluse: https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/LICENSE, https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE, https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/, `extra_gated_prompt` di https://huggingface.co/swiss-ai/Apertus-v1.5-8B
