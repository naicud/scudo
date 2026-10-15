# Dataset pubblici per il fine-tune di Fast e Accurate

> Origine: nota di ricerca del repository privato `open-guard`, copiata qui.
> I numeri sono dichiarati dalle fonti, non misure di Scudo.

Ricerca del **15 ottobre 2026**. Inventario dei dataset pubblici (Hugging Face Hub
e Kaggle) utilizzabili per addestrare il Fast tipizzato (2–4 B) e l'Accurate
ragionante (~4–9 B) previsti dall'[ADR 0001][adr], che giudicano un testo
rispetto a un criterio in linguaggio naturale e restituiscono `VIOLATION`,
`NO_VIOLATION` o `INSUFFICIENT`. Lingue di interesse: italiano, inglese e
misto IT/EN. I pesi risultanti sono destinati a pubblicazione **Apache-2.0**:
la licenza dei dati e i termini dei modelli che li hanno generati sono il
criterio principale. Il documento **non** contiene misure del progetto né
pareri legali: le colonne di compatibilità sono una prima lettura da far
validare ai responsabili Legal/Risk prima di qualsiasi training.

## Metodo e convenzioni

Per ogni dataset Hugging Face sono stati letti il 15 ottobre 2026: l'API
`https://huggingface.co/api/datasets/<id>` (`cardData.license`,
`license_name`, `gated`, tag `language:`), il README grezzo e, dove
disponibile, l'endpoint `datasets-server /size` per il numero di righe per
split. Per i dataset gated il README è stato letto dalla pagina pubblica del
dataset. Il conteggio delle righe italiane, quando riportato, viene
dall'endpoint `datasets-server /statistics` sulla colonna di lingua. Per
Kaggle è stata usata l'API pubblica `api/v1/datasets/view` (campo licenza);
le pagine e le regole delle competizioni richiedono login e restano
**NON VERIFICATO**. I modelli usati per generare o etichettare i dati sono
quelli dichiarati nel README o nel paper collegato (arXiv HTML).

- **NON VERIFICATO**: affermazione non confermata su fonte primaria; non va
  usata come presupposto.
- **Righe**: somma o elenco degli split pubblicati; `n/d` se l'endpoint non
  risponde (tipicamente dataset gated).
- **Comm. + pesi**: `SÌ` se licenza dei dati e termini dei generatori
  dichiarati non pongono limiti noti a uso commerciale e pubblicazione di
  pesi derivati; `NO` se c'è una clausola esplicita contraria (NC, divieto di
  trasferimento, licenza di valutazione); `INCERTO` con il motivo.
- **Superfici**: (a) prompt dannosi / tassonomie; (b) risposte dannose o non
  conformi; (c) prompt injection / jailbreak, diretti e indiretti; (d) criteri
  personalizzati (policy + testo → conforme o no); (e) over-refusal /
  borderline benigni; (f) sicurezza agentica / tool call; (g) PII / data leak;
  (h) italiano. `R` indica tracce di ragionamento utili all'Accurate.

## Vincoli dei generatori che ricorrono

La licenza del dataset (CC-BY, ODC-BY, MIT) copre il contributo degli autori,
non i termini d'uso dei modelli che hanno prodotto testi o etichette. Tre
clausole ricorrono nelle tabelle:

- **OpenAI** (GPT-3.5/4/4o/4.1): i termini d'uso vietano di «Use Output to
  develop models that compete with OpenAI» ([copia del 2024-12-11][oaitou];
  la pagina ufficiale [openai.com][oai] risponde 403 agli strumenti usati, il
  testo corrente è **NON VERIFICATO**). Un guardrail pubblicato come modello
  aperto ricade potenzialmente nel divieto: dati con testi o label GPT sono
  marcati `INCERTO`.
- **Gemma**: i [Gemma Terms of Use][gemma] definiscono «Model Derivatives»
  anche i modelli addestrati con «synthetic data Outputs by Gemma» per farli
  funzionare in modo simile a Gemma; tali derivati restano soggetti ai
  termini Gemma.
- **Llama 3 / 3.1**: la [licenza Llama 3][llama3] vieta di usare output per
  migliorare altri LLM; la [licenza Llama 3.1][llama31] impone di anteporre
  «Llama» al nome di qualunque modello distribuito addestrato con i suoi
  output.
- **Traduttori NC**: `Unbabel/TowerInstruct-7B-v0.2` e
  `facebook/nllb-200-3.3B` hanno `license: cc-by-nc-4.0` sull'API HF
  ([tower], [nllb]); le traduzioni prodotte con essi ereditano un rischio NC
  **NON VERIFICATO** sul piano legale.

## Dataset di training

### (a) Prompt dannosi e tassonomie, (b) risposte

| Dataset | Righe | Lingue | Etichette | Sintetico / generatori | Licenza | Gated | Comm. + pesi | Sup. | Note e rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` [aeg2] | 30.007 train, 1.445 val, 1.964 test | en | `prompt_label`/`response_label` safe/unsafe; 12 categorie + 9 sottocategorie | ibrido: prompt umani (HH-RLHF, DAN, AART), risposte Mistral-7B-v0.1; label risposta da giuria Mixtral-8x22B, Mistral-NeMo-12B, Gemma-2-27B; 5.000 rifiuti generati con Gemma-2-27B | CC-BY-4.0 | no | SÌ per prompt/label umane; INCERTO per il sottoinsieme rifiuti Gemma (termini Gemma) | a, b | campioni Suicide Detection redatti e non distribuiti; test da tenere held-out |
| `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` [aeg1] | 10.798 train, 1.199 test | en | safe/unsafe + 13 categorie (NON VERIFICATO il numero esatto) | prompt HH-RLHF, risposte Mistral-7B-v0.1, label umane | CC-BY-4.0 | no | SÌ | a, b | sovrapposto ad Aegis 2.0: deduplicare |
| `nvidia/Nemotron-Safety-Guard-Dataset-v3` [nsg3] | 451.424 train, 27.791 val, 35.402 test; **it: 37.534 train, 2.950 test** | en, ar, de, es, fr, hi, **it**, ja, ko, nl, th, zh | schema Aegis 2.0 (id mappati) + `language` | adattamento culturale e traduzione di Aegis 2.0 con pipeline CultureGuard: traduzione Google Translate, giuria di filtro Llama-Nemotron-Safety-Guard-V2, PolyGuard-Qwen, Llama-3.1-Nemotron-70B, Qwen3-235B [cg] | CC-BY-4.0 | no | SÌ dichiarato («ready for commercial/non-commercial use»); i vincoli Gemma del sottoinsieme rifiuti di Aegis 2.0 restano da tracciare | a, b, h | miglior fonte italiana con licenza commerciale; italiano tradotto a macchina, non nativo |
| `nvidia/Nemotron-SFT-Safety-v2` [sft2] | ~135.000 (dichiarato); **it: 14.459** | en, de, es, fr, ja, **it**, zh | prompt + risposta sicura (con ragionamento), sottoinsiemi per fonte; label guard per riga NON VERIFICATO | ibrido; traduzione con Riva-Translate-4B (`nvidia-community-model-license`) | CC-BY-4.0 (+ Apache-2.0, MIT) | no | SÌ dichiarato («ready for commercial or non-commercial uses») | a, c, e, h | dati SFT di allineamento, non di classificazione: servono come prompt IT, le label vanno ricostruite; include `sdg_oversafety` e `sdg_latent_injection` |
| `nvidia/Nemotron-SFT-Safety-v1` [sft1] | 45.145 | en | come v2 | come v2 | CC-BY-4.0 (+ Apache, MIT) | no | SÌ dichiarato | a, e | sottoinsieme inglese della v2 |
| `allenai/wildguardmix` [wgm] | WildGuardTrain 86.759; WildGuardTest 1.725 | en | prompt harm, response harm, response refusal | 87% sintetico; prompt e risposte da GPT-4 e altri LLM; label GPT-4 con audit umano [wg] | ODC-BY + accettazione AI2 Responsible Use Guidelines | sì (auto) | INCERTO: testi e label GPT-4 (termini OpenAI) | a, b, c, e | WildGuardTest è benchmark: held-out |
| `allenai/wildjailbreak` [wjb] | n/d (gated); split adversarial harmful 82.728 dichiarato | en | vanilla/adversarial × harmful/benign | GPT-4, GPT-3.5, Mixtral-8x7B [wtm] | ODC-BY + AI2 RUG | sì (auto) | INCERTO: GPT | a, c, e | contrasto adversarial-benign utile all'over-refusal |
| `ToxicityPrompts/PolyGuardMix` [pgm] | 1.910.372 | 17 lingue incl. **it** (righe it NON VERIFICATO) | prompt harm, response harm, refusal, categorie | traduzioni WildGuardMix con TowerInstruct-7B e NLLB; label GPT-4o + Llama-Guard-3-8B [pg] | CC-BY-4.0 | no | INCERTO/NO: label GPT-4o, origine GPT-4, traduttori CC-BY-NC; il paper CultureGuard lo definisce «non-commercially usable» [cg] | a, b, c, h | licenza del repository più permissiva delle fonti |
| `ToxicityPrompts/mWildGuardMix-train-tower-nllb-v2.1` [mwgm] | 1.474.903 | multilingue (tag assenti) | come WildGuardMix | traduzioni Tower/NLLB | ODC-BY | no | INCERTO/NO: traduttori NC e origine GPT-4 | a, b, h | — |
| `OpenSafetyLab/Salad-Data` [salad] | base 21.318; attack 5.000; defense 200; MCQ 3.840 | en | tassonomia a 3 livelli (6/16/66) | 15.433 domande da GPT-3.5 fine-tunato; il resto da HH, AdvBench, ToxicChat, DAN, ecc. | Apache-2.0 | no | INCERTO: GPT-3.5 e sottoinsieme ToxicChat (CC-BY-NC) | a, c | — |
| `Anthropic/hh-rlhf` [hh] | 160.800 train, 8.552 test | en | preferenze; red-team con rating (NON VERIFICATO il formato del sottoinsieme red-team) | prompt umani, risposte di modelli Anthropic | MIT | no | SÌ (licenza MIT del rilascio) | a | non è un dataset di classificazione: va rietichettato |
| `CohereForAI/aya_redteaming` [aya] | n/d (endpoint 404) | ar, en, fil, fr, hi, ru, sr, es | 9 categorie, globale/locale | umano, annotatori professionisti | Apache-2.0 | no | SÌ («any purpose, whether academic or commercial») | a | niente italiano; ottimo seme da tradurre |
| `gretelai/gretel-safety-alignment-en-v1` [gsa] | 5.997 train, 1.183 test, 1.181 val | en | `risk_category`, sottocategoria; prompt/response/safe_response | Gretel Navigator con Granite-3.0-8B e Qwen2.5-7B (Apache-2.0) | Apache-2.0 nel README (`cardData.license` vuoto) | no | SÌ | a, b | — |
| `ibm-research/AttaQ` [attaq] | 1.402 | en | 7 categorie | HH-RLHF + generazione LLM (modello NON VERIFICATO) | MIT | no | INCERTO: generatore non dichiarato | a | — |
| `LibrAI/do-not-answer` [dna] | 939 | en | 5 aree di rischio; risposte di 6 modelli con harmful e action category | risposte GPT-4, ChatGPT, Claude, LLaMA-2, Vicuna, ChatGLM2 | Apache-2.0 | no | SÌ per i prompt; INCERTO per le risposte (termini dei fornitori) | a, b | piccolo; più adatto alla valutazione |
| `KoalaAI/Text-Moderation-Multilingual` [koala] | 1.459.350 train, 162.150 val | 17 lingue incl. **it** | multi-label S/H/V/HR/SH/S3/H2 | aggregato di fonti, incluso il dataset di valutazione OpenAI | Apache-2.0 (tag) | no | INCERTO: il README rimanda alle licenze delle fonti | a, h | `enguard/multi-lingual-prompt-moderation` [enguard] ha gli stessi conteggi: probabile copia |
| `google/civil_comments` [civil] | 1.804.874 train, 97.320 val, 97.320 test | en | punteggi di tossicità per attributo | commenti umani, annotazione crowd | CC0-1.0 | no | SÌ | a | tossicità, non policy LLM |
| `google/jigsaw_toxicity_pred` [jig] | n/d (download manuale) | en | 6 label di tossicità | commenti Wikipedia | CC0 «with the underlying comment text being governed by Wikipedia's CC-SA-3.0» | no | INCERTO: share-alike sul testo | a | — |
| `textdetox/multilingual_toxicity_dataset` [tdx] | 5.000 per lingua; **it: 5.000** | 15 lingue incl. **it** | toxic 0/1 | IT da AMI, HODI e Jigsaw Multilingual (Kaggle) | openrail++ | no | INCERTO: AMI è CC-BY-NC-SA, Jigsaw ha regole di competizione | a, h | licenza a valle incoerente con le fonti italiane |

### (c) Prompt injection e jailbreak

| Dataset | Righe | Lingue | Etichette | Sintetico / generatori | Licenza | Gated | Comm. + pesi | Sup. | Note e rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `deepset/prompt-injections` [dps] | 546 train, 116 test | non dichiarate | `label` 0/1 | NON VERIFICATO | Apache-2.0 | no | SÌ | c | README vuoto |
| `rikka-snow/prompt-injection-multilingual` [rikka] | 6.006 train, 1.276 test | 11 lingue incl. **it** | `label` 0/1 | NON VERIFICATO (le prime righe coincidono nel tema con deepset) | MIT | no | INCERTO: provenienza e traduttore non dichiarati | c, h | — |
| `yanismiraoui/prompt_injections` [yanis] | 1.034 | en, fr, de, es, pt, **it**, ro | solo positivi (una colonna) | NON VERIFICATO | Apache-2.0 | no | SÌ | c, h | servono negativi da altre fonti |
| `jackhhao/jailbreak-classification` [jack] | 1.044 train, 262 test | en | `jailbreak`/`benign` | jailbreak da `verazuo/jailbreak_llms`; benigni da GPTeacher (GPT-4) | Apache-2.0 | no | SÌ per i jailbreak; INCERTO per i benigni GPT-4 | c | — |
| `TrustAIRLab/in-the-wild-jailbreak-prompts` [twj] | jailbreak 666 + 1.405; regular 5.721 + 13.735 | en | jailbreak/regular per snapshot | raccolti da Reddit, Discord, FlowGPT | MIT | no | SÌ (MIT), con rischio di contenuti utente e dati personali | c | — |
| `Lakera/mosscap_prompt_injection` [moss] | 223.533 train, 27.683 val, 27.729 test | non dichiarate | solo attacchi dei giocatori | umano (gioco) | MIT | no | SÌ | c | positivi soltanto |
| `Lakera/gandalf_ignore_instructions` [gandalf] | 777 train, 111 val, 112 test | en | solo attacchi | umano (gioco) | MIT | no | SÌ | c | — |
| `microsoft/llmail-inject-challenge` [llmail] | Phase1 370.724, Phase2 90.916 | non dichiarate | flag `defense.undetected`, `exfil.sent`, sottoinsieme etichettato | attacchi umani contro un assistente email (GPT-4o mini e altri come vittime) | MIT | no | SÌ | c, f | iniezione **indiretta** realistica; utile anche per il caso negativo |
| `hackaprompt/hackaprompt-dataset` [hap] | n/d (gated) | en | livello, successo, completamento atteso | prompt umani; completion di text-davinci-003, FlanT5-XXL, gpt-3.5-turbo | MIT | sì (auto) | SÌ per i prompt; INCERTO per le completion GPT | c | — |
| `reshabhs/SPML_Chatbot_Prompt_Injection` [spml] | 16.012 | en | system prompt + user prompt, label injection e grado | system prompt estesi con GPT-4 | MIT | no | INCERTO: GPT-4 | c, d | coppia system prompt / attacco vicina al formato policy + testo |
| `neuralchemy/Prompt-injection-dataset` [neur] | core 6.274; full 15.919 (somma split) | en | `label` 0/1, 29 categorie, severità | aggregato: fonti Apache, CC-BY, MIT e «WildGuard / JudgeComparison — Research» | Apache-2.0 (tag) | no | INCERTO: parte delle fonti è solo «Research» | c | — |
| `xTRam1/safe-guard-prompt-injection` [xtram] | 8.236 train, 2.060 test | non dichiarate | attacco/sicuro | GPT-3.5-turbo | assente | no | NO: nessuna licenza; INCERTO anche per GPT-3.5 | c | — |
| `jayavibhav/prompt-injection` [jaya] | 261.738 train, 65.416 test | non dichiarate | NON VERIFICATO | NON VERIFICATO | assente | no | NO: nessuna licenza | c | — |
| `Necent/llm-jailbreak-prompt-injection-dataset` [necent] | n/d (gated) | 27 lingue incl. **it** | NON VERIFICATO | NON VERIFICATO | MIT | sì (auto) | INCERTO: README non leggibile senza accesso | c, h | fonte principale di Nailyk14 |
| `Nailyk14/prompt-safety-multilingual` [nailyk] | 1.115.571 train, 47.784 val, 47.790 test | 50+ lingue incl. **it** | schema WildGuard + label fine (benign, injection, jailbreak, hijack, exfiltration) | aggregato: Necent (MIT), 40.898 righe Qwen2.5-7B (Apache-2.0), contesti BIPIA generati con GPT-4o | `other` | no | INCERTO: licenza non dichiarata e righe GPT-4o | c, f, h | il README riconosce che la mappatura WildGuard è derivata, non annotata |
| `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT` [bipia] | n/d (gated) | non dichiarate | NON VERIFICATO | GPT (dal nome; NON VERIFICATO) | CC-BY-SA-4.0 | sì (auto) | INCERTO: share-alike e GPT | c | — |
| Kaggle `mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd` [mpdd] | 39.234 dichiarate | en | malicious/benign 50/50 | aggregato di dataset Kaggle | CC0 (dichiarato dal caricatore) | no | INCERTO: le fonti aggregate hanno licenze proprie | c | — |

### (d) Criteri personalizzati e decisioni tipizzate

| Dataset | Righe | Lingue | Etichette | Sintetico / generatori | Licenza | Gated | Comm. + pesi | Sup. | Note e rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tasksource/procedural-typed-decisions` [ptd] | `all`: 360.000 train, 18.000 val, 18.000 test | en | domande tipizzate `choice`, `noul`, `score` su uno stato | generazione procedurale deterministica, nessun LLM dichiarato | Apache-2.0 | no | SÌ | d | formato vicino al Fast; non è dominio safety |
| `ZefanCai/Open-Jev` [oj] | config `release-v2-redistributable`: 79.116 train, 4.672 cal., 3.723 val, 10.356 test, 15.701 OOD; altre 11 config | en, zh, tr | `choice`/`noul`/`score` con distribuzioni target | «mostly synthetic»; il modello teacher non è dichiarato (NON VERIFICATO); `metadata_json` può contenere label teacher privilegiate | CC0-1.0 per i contenuti generati; codice MIT; testi upstream esclusi | no | SÌ per le config redistributable; INCERTO per le descrizioni TypeSafe senza licenza verificata | d | test e OOD hanno label pubbliche: dichiarare l'uso |
| `ZefanCai/Open-Jev-v1.1` [oj11] | 147.139 train, 26.675 cal., 25.413 val, 43.125 test, 84.267 OOD | en, zh | come Open-Jev | sintetico + WANLI (CC-BY-4.0) | `source-dependent-cc0-1.0-and-cc-by-4.0` | no | SÌ con attribuzione WANLI | d | Wikispeedia escluso per licenza non verificata |
| `montehoover/DynaBench` (alias `tomg-group-umd/DynaBench`) [dynab] | DynaBenchTrain 61.496; DynaBenchSafetyMix 70.313 train + 7.813 val; DynaBench test 543 | en | PASS/FAIL per policy con regole multiple; tracce di spiegazione | dialoghi GPT-4o-mini e GPT-4.1-mini; label e tracce GPT-4o e Gemini-2.0-Flash; regole espanse con GPT-4o, Gemini-2.0-Flash, Claude Sonnet 3.5 [dg] | MIT | no | INCERTO per Train (OpenAI/Google); NO/INCERTO per SafetyMix, che include BeaverTails e ToxicChat (CC-BY-NC) | d, R | DynaBench test (543, verificato da 3 annotatori) va held-out |
| `nvidia/CantTalkAboutThis-Topic-Control-Dataset` [ctat] | 1.073 train, 20 test (1.080 dialoghi) | en | dialoghi con turni distrattori fuori tema, 9 domini | Mixtral-8x7B-Instruct (Apache-2.0), versione «commercially friendly» | CC-BY-4.0 | no | SÌ | d | policy di sistema + turni: formato BYOC |
| `nvidia/Nemotron-Content-Safety-Reasoning-Dataset` [ncsr] | 28.000 (Aegis 2.0) + 8.712 (topic control) | en | label originali + tracce di ragionamento | tracce DeepSeek-R1-0528, Qwen3-32B, gpt-oss-120b; filtrate su accordo con la ground truth | CC-BY-4.0 | no | SÌ dichiarato («ready for commercial use»); eredita il caveat Gemma di Aegis 2.0 | a, b, d, R | candidato principale per l'Accurate |
| `Virtue-AI-HUB/PolyGuard` [vpg] | 100K–1M (tag); config per dominio (code, cyber, education, …) | en | safe/unsafe per policy di dominio (NON VERIFICATO) | NON VERIFICATO (README senza descrizione) | CC-BY-4.0 | no | INCERTO: generatori non dichiarati | d | omonimo di PolyGuard (ToxicityPrompts), progetto diverso |

### (e) Over-refusal e (f) agenti

| Dataset | Righe | Lingue | Etichette | Sintetico / generatori | Licenza | Gated | Comm. + pesi | Sup. | Note e rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bench-llm/or-bench` [orb] | 80k: 80.359; hard-1k: 1.319; toxic: 655 | en | prompt «seemingly toxic» + categoria | riscrittura con Mixtral-8x7B; moderazione con GPT-4-turbo, Llama-3-70b, Gemini-1.5-pro [orbp] | CC-BY-4.0 | no | INCERTO: label da modelli con termini restrittivi | e | hard-1k e toxic da tenere held-out |
| `allenai/coconot` [coco] | original 11.477 train + 1.001 test; contrast 379 test; pref 927 | en | categoria di non conformità; risposte | risposte GPT-4 (`gpt-4-1106-preview`) [cocop] | `cardData.license` assente; il README cita sia AI2 ImpACT-LR sia ODC-BY | no | INCERTO: licenza contraddittoria e GPT-4 | e | contrast set è test |
| `mii-llm/refusal-ita` [miiref] | 781 | **it** | system/prompt/output | NON VERIFICATO (README senza descrizione) | Apache-2.0 | no | INCERTO: generatore ignoto | e, h | — |
| `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` [nipi] | 1.272 | en | ambiente Gym con verifier deterministico | sintetico, attaccante Nemotron Super 3, difensore Nemotron Nano 3 (`nvidia-nemotron-open-model-license`) | CC-BY-4.0 | no | SÌ dichiarato («ready for commercial or non-commercial uses»); termini sugli output Nemotron NON VERIFICATO | c, f | formato RL: va convertito in esempi di classificazione |

### (g) PII e fuga di dati

| Dataset | Righe | Lingue | Etichette | Sintetico / generatori | Licenza | Gated | Comm. + pesi | Sup. | Note e rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nvidia/Nemotron-PII` [npii] | 100.000 (50k train, 50k test) | en | span di 55+ categorie PII/PHI | NeMo Data Designer (LLM sottostante NON VERIFICATO) | CC-BY-4.0 | no | SÌ dichiarato | g | span NER: va riformulato come giudizio su policy |
| `gretelai/synthetic_pii_finance_multilingual` [gpii] | 50.346 train, 5.594 test | en, fr, de, nl, es, **it** | span di 29 tipi PII | Gretel Navigator: Mistral-7B fine-tunato, Mixtral-8x7B | Apache-2.0 | no | SÌ | g, h | dominio finanziario |
| `rizzoaiacademy/rizzo-pii-it-dataset` [rpii] | 7.000 validation (datasets-server); pool più ampio nel README | **it** | BIO, 22 tag (CF, P.IVA, catasto) | `synthetic/*` scritti da un LLM non nominato (NON VERIFICATO) con valori iniettati da codice | `mixed-see-card`: `synthetic/*` MIT; altre parti con licenze a monte | no | SÌ solo per `synthetic/*`; INCERTO per le parti Ai4Privacy/DeepMount | g, h | — |
| `ai4privacy/open-pii-masking-500k-ai4privacy` [ai5] | 464.150 train, 116.077 val | en, fr, de, **it**, es, nl, hi, te | span PII | generato con Llama 3.1 e 3.3 | `other` (`license_name: cc-by-4.0`) + licenza Llama | no | NO per un modello Apache con nome libero: la licenza Llama 3.1 impone «Llama» nel nome | g, h | — |
| `ai4privacy/pii-masking-400k` / `-300k` [ai4] | 325.517 + 81.379 / 177.677 + 47.728 | en, fr, de, **it**, es, nl | span PII | «proprietary algorithms» | `other` (`license.md`) | no | NO: «Commercial entities should contact us» | g | — |
| `DeepMount00/pii-masking-ita` [dmpii] | n/d (gated) | it (dal nome) | NON VERIFICATO | NON VERIFICATO | assente | sì (auto) | INCERTO: nessuna licenza dichiarata | g, h | — |

### (h) Italiano e multilingue con italiano, non classificati sopra

| Dataset | Righe | Lingue | Etichette | Sintetico / generatori | Licenza | Gated | Comm. + pesi | Sup. | Note e rischi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `evalitahf/hatespeech_detection` (HaSpeeDe2) [haspe] | dev 6.839; test 1.763 (tweet 1.263, news 500) | **it** | hate speech, stereotipi | tweet e titoli umani, pseudonimizzati | CC-BY-NC-SA-4.0 | no | NO: NC | a, h | utile solo come valutazione interna |
| `RiTA-nlp/ami_2020` [ami] | n/d (gated) | **it** | misoginia, aggressività (NON VERIFICATO) | umano | CC-BY-NC-SA-4.0 | sì (auto) | NO: NC | a, h | — |
| `felfri/M-ALERT` [malert] | 14.763 per lingua | en, de, es, fr, **it** | categoria ALERT | traduzioni MT-OPUS, Google Translate, Tower-instruct-mistral | CC-BY-NC-SA-4.0 | no | NO: NC | a, h | — |
| `heretic-org/Multilingual-Harmless-Harmful` [heretic] | 416 harmful + 416 harmless per lingua | 9 lingue incl. **it** | harmful/harmless | traduzione (strumento NON VERIFICATO); IT controllato a campione dall'autore | CC-BY-SA-4.0 | no | INCERTO: share-alike | a, e, h | coppie semanticamente vicine, utili come hard negative |
| `iNLP-Lab/multilingual-safety` [inlp] | 1.000 per lingua | 12 lingue incl. **it** | prompt dannoso + rifiuto | tradotto con Gemini-2.0-Flash | CC-BY-NC-4.0 | no | NO: NC | a, h | — |
| `hasankursun/multilingual-safety-classification-dataset` [hasan] | 212.594 | multilingue incl. **it** | NON VERIFICATO | NON VERIFICATO | CC-BY-NC-SA-4.0 | no | NO: NC | a, h | — |
| `YATAV-ENT/aegis-multilingual-guard-dataset-v0.1-samples` [yatav] | 1.200 (anteprima) | 25 lingue incl. **it** | attack/benign, tecnica, dominio | `source: llm_gen` (modello NON VERIFICATO) | `aegis-responsible-ai-use-v0.1` | no | NO/INCERTO: licenza proprietaria, dataset completo su richiesta | a, c, h | non collegato a NVIDIA Aegis |
| `MilaNLProc/honest` [honest] | 810–840 template per lingua | en, **it**, fr, pt, ro, es | template per completamenti offensivi | template scritti da madrelingua | MIT | no | SÌ | h | misura bias di completamento, non guard |
| Kaggle `wajidhassanmoosa/multilingual-hatespeech-dataset` [kmh] | n/d | 13 lingue incl. **it** | hate 0/1 | fonti NON VERIFICATO | ODC-By (dichiarata dal caricatore) | no | INCERTO: fonti non documentate | a, h | — |
| Kaggle `julian3833/jigsaw-multilingual-toxic-comment-classification` [kjm] | n/d | en + lingue della competizione (presenza di it in validation/test NON VERIFICATO) | toxic | ricaricamento dei dati della competizione | CC0 (dichiarato dal caricatore) | no | INCERTO: le regole della competizione originale non sono verificabili senza login | a, h | licenza del mirror diversa da quella della fonte |
| Kaggle `miklgr500/jigsaw-train-multilingual-coments-google-api` [kjt] | n/d | pt, tr, ru, **it**, es, fr | toxic (dal train Jigsaw) | traduzione Google API (libreria `translators`) | CC0 (dichiarato dal caricatore) | no | INCERTO: testo Wikipedia CC-BY-SA e output di Google Translate | a, h | — |

### Tracce di ragionamento per l'Accurate

| Dataset | Righe | Generatori delle tracce | Licenza | Comm. + pesi | Note |
| --- | --- | --- | --- | --- | --- |
| `nvidia/Nemotron-Content-Safety-Reasoning-Dataset` [ncsr] | 36.712 | DeepSeek-R1-0528, Qwen3-32B, gpt-oss-120b | CC-BY-4.0 | SÌ dichiarato | tracce filtrate sull'accordo con la label; solo inglese |
| `yueliu1999/GuardReasonerTrain` [grt] | WildGuardTrainR 86.759; AegisTrainR 10.798; BeaverTailsTrainR 27.186; ToxicChatTrainR 2.801 | GPT-4o, con la ground truth in input [grp] | MIT (tag) | NO/INCERTO: tracce GPT-4o; i sottoinsiemi BeaverTails e ToxicChat derivano da fonti CC-BY-NC-4.0 | esempio di licenza a valle più permissiva delle fonti |
| `montehoover/DynaBench` [dynab] | vedi (d) | GPT-4o, Gemini-2.0-Flash; integra anche le tracce GuardReasoner [dg] | MIT | INCERTO | unico set di policy multiregola con tracce |
| `nvidia/Nemotron-SFT-Safety-v2` [sft2] | vedi (a) | NON VERIFICATO (il README descrive la generazione a due fasi senza nominare i modelli nelle righe lette) | CC-BY-4.0 | SÌ dichiarato | contiene italiano |

## Benchmark solo test (held-out)

Questi set vanno esclusi dal training e da ogni selezione di checkpoint
diversa da quella dichiarata; molti modelli guard pubblici li citano come
valutazione, quindi la contaminazione va controllata anche nei dataset
aggregati sopra (es. Nailyk14 include XSTest come hard negative).

| Benchmark | Righe | Lingue | Licenza | Gated | Sup. | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `Paul/XSTest` [xst], `walledai/XSTest` [wxst] | 450 (250 safe, 200 unsafe) | en | CC-BY-4.0 | no / sì | e | alcuni esempi scritti con aiuto GPT-4 |
| `allenai/xstest-response` [xstr] | n/d | en | ODC-BY | sì | b, e | — |
| `saiteki-kai/XSTest-IT` [xstit] | 3.150 (450 `it`; 7 valori di `lang`); 1.750 safe, 1.400 unsafe | **it**, scn, fur, lmo, lij, vec | CC-BY-4.0 | no | e, h | unico over-refusal italiano trovato; traduttore NON VERIFICATO |
| WildGuardTest in `allenai/wildguardmix` [wgm] | 1.725 | en | ODC-BY | sì | a, b, c, e | — |
| `ToxicityPrompts/PolyGuardPrompts` [pgp] | 29.325; **it: 1.725** | 17 lingue | CC-BY-4.0 | no | a, b, h | test tradotto con GPT-4o [pg] |
| `ToxicityPrompts/RTP-LX` [rtplx] | 30.253; **it: 1.113** | 28 locale | ODC-BY | no | a, h | tossicità di completamento |
| `ToxicityPrompts/PolygloToxicityPrompts` [ptp] | n/d (endpoint 501) | 17 lingue incl. **it** | CC-BY-4.0 | no | a, h | — |
| `ToxicityPrompts/XSafety` [xsafe] | 27.999 | 10 lingue, senza it | ODC-BY | no | a | — |
| `DAMO-NLP-SG/MultiJail` [mj] | 315 prompt × 10 lingue | en, zh, **it**, vi, ar, ko, th, bn, sw, jv | MIT | no | a, c, h | traduzioni umane |
| `Paul/hatecheck-italian` [hci] | 3.690 | **it** | CC-BY-4.0 | no | a, h | suite funzionale (MHC) con gold ed etichette annotatori |
| `Paul/hatecheck` [hc] | 3.728 | en | CC-BY-4.0 | no | a | — |
| `lmsys/toxic-chat` [tc] | 5.082 train + 5.083 test (0124) | en | CC-BY-NC-4.0 | no | a, c | NC: solo valutazione interna |
| `JailbreakBench/JBB-Behaviors` [jbb] | 100 harmful, 100 benign, 300 judge | en | MIT | no | a, c | — |
| `walledai/HarmBench` [hb] | n/d | en | MIT | sì | a | — |
| `walledai/AdvBench` [advb] | n/d | en | MIT | sì | a | — |
| `Bertievidgen/SimpleSafetyTests` [sst] | 100 | en | CC-BY-2.0 | no | a | — |
| `Babelscape/ALERT` [alert] | 14.763 + 30.968 adversarial | en | CC-BY-NC-SA-4.0 | no | a, c | anche `Babelscape/ALERT_DPO` [alertdpo] è NC-SA |
| `sorry-bench/sorry-bench-202503` [sorry] | n/d | en, zh, fr, ml, mr, ta | `sorry-bench` (vieta il trasferimento a terzi) | sì | a, e | — |
| `Qwen/Qwen3GuardTest` [q3gt] | 1.059 thinking, 813 response_loc, 569 thinking_loc | en (NON VERIFICATO) | CC-BY-NC-4.0 | no | b, R | sicurezza delle tracce di ragionamento |
| `montehoover/DynaBench` config DynaBench [dynab] | 543 | en | MIT | no | d | policy fuori tassonomia safety |
| `Praveenrajus/jev-bench` [jevb] | molti config (ARC, Banking77, civil_comments, …) | en | `mixed-see-manifest` | no | d | ricomprende test pubblici noti: controllare overlap |
| `apus-ailab/APUS-OpenJev-Eval-Frozen80` [f80] | 80 | en | nessuna licenza unica («no single blanket dataset license») | no | d | il README lo dichiara «private» per revisione dei diritti |
| `normster/RuLES` [rules] | n/d | en | Apache-2.0 | no | d | regole dello sviluppatore vs jailbreak |
| `ai-safety-institute/AgentHarm` [ah] | 176 + 176 + 44 test_public; 32 + 32 + 8 val | en | MIT con clausola aggiuntiva (solo scopi di sicurezza) | no | f | — |
| `thu-coai/Agent-SafetyBench` [asb] | 2.000 casi, 349 ambienti (paper [asbp]) | en | MIT | no | f | casi generati con GPT-4o |
| `nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0` [aiq] | safety 2.800 + 2.800, security 2.600 + 2.596 trace (con/senza difesa) | en | NVIDIA Evaluation Dataset License | no | c, f | licenza di sola valutazione |
| `qualifire/prompt-injections-benchmark` [qpi] (= `rogue-security/...`) | n/d | n/d | CC-BY-NC-4.0 | sì | c | — |

## Lacune italiane

1. **Nessun set nativo italiano con label guard e licenza commerciale.** Le
   fonti native (HaSpeeDe2, AMI 2020, M-ALERT) sono CC-BY-NC-SA. L'italiano
   utilizzabile viene da traduzioni automatiche NVIDIA: Nemotron-Safety-Guard
   v3 (37.534 righe train, Google Translate) e Nemotron-SFT-Safety-v2 (14.459,
   Riva-Translate). I testi italiani di PolyGuardMix passano per traduttori e
   label con vincoli NC/OpenAI.
2. **Criteri personalizzati solo in inglese** (o en/zh/tr): DynaBench, Open-Jev,
   procedural-typed-decisions e CantTalkAboutThis non hanno italiano.
3. **Iniezione indiretta e agenti in italiano assenti**: in italiano ci sono
   solo prompt injection diretti e piccoli (yanismiraoui, 1.034 solo positivi;
   rikka-snow, provenienza ignota).
4. **Over-refusal italiano solo come test** (XSTest-IT) e un piccolo SFT
   (refusal-ita, 781, generatore ignoto).
5. **Testo misto IT/EN**: nessun dataset trovato con code-switching
   etichettato.

Candidati da tradurre (licenza che ammette derivati e uso commerciale):
`tasksource/procedural-typed-decisions` (Apache-2.0, generatore
deterministico: si può rigenerare direttamente in italiano invece di
tradurre), `nvidia/CantTalkAboutThis-Topic-Control-Dataset` e
`nvidia/Nemotron-Content-Safety-Reasoning-Dataset` (CC-BY-4.0, criteri e
tracce per l'Accurate), `microsoft/llmail-inject-challenge` e
`Lakera/mosscap_prompt_injection` (MIT, iniezioni), i config redistributable
di `ZefanCai/Open-Jev` (CC0), `CohereForAI/aya_redteaming` (Apache-2.0,
umano), `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` (CC-BY-4.0)
e `gretelai/gretel-safety-alignment-en-v1` (Apache-2.0). Il traduttore deve
avere a sua volta termini compatibili: Tower e NLLB sono CC-BY-NC; un
traduttore Apache-2.0 è preferibile, i termini di Google Translate per questo
uso sono **NON VERIFICATO**. Le traduzioni vanno controllate a campione da
madrelingua e le label ri-validate (la pipeline CultureGuard usa il filtro di
consistenza per back-translation [cg]).

## Shortlist per licenza compatibile con pesi Apache-2.0

Solo dataset con `Comm. + pesi` = SÌ (o SÌ dichiarato dal produttore). Ogni
voce resta da confermare legalmente prima del training.

- **(a) Prompt dannosi**: Aegis 2.0 e 1.0 (escludendo o tracciando i rifiuti
  Gemma-2-27B), Nemotron-Safety-Guard-v3, Nemotron-SFT-Safety-v1/v2,
  `CohereForAI/aya_redteaming`, `gretelai/gretel-safety-alignment-en-v1`,
  `google/civil_comments` (tossicità).
- **(b) Risposte**: Aegis 2.0 (`response_label`), Nemotron-Safety-Guard-v3,
  `gretelai/gretel-safety-alignment-en-v1`.
- **(c) Injection / jailbreak**: `deepset/prompt-injections`,
  `microsoft/llmail-inject-challenge` (indiretta),
  `Lakera/mosscap_prompt_injection`, `Lakera/gandalf_ignore_instructions`,
  `TrustAIRLab/in-the-wild-jailbreak-prompts`, `yanismiraoui/prompt_injections`
  (IT), parte jailbreak di `jackhhao/jailbreak-classification`.
- **(d) Criteri personalizzati**: `tasksource/procedural-typed-decisions`,
  `ZefanCai/Open-Jev` (config redistributable), `ZefanCai/Open-Jev-v1.1`,
  `nvidia/CantTalkAboutThis-Topic-Control-Dataset`.
- **(e) Over-refusal**: nessun set dedicato pulito; usare `sdg_oversafety` di
  Nemotron-SFT-Safety-v2 e le coppie harmless di Aegis 2.0. OR-Bench, CoCoNot
  e WildJailbreak restano INCERTO.
- **(f) Agenti**: `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1`,
  `microsoft/llmail-inject-challenge`.
- **(g) PII**: `nvidia/Nemotron-PII`,
  `gretelai/synthetic_pii_finance_multilingual` (IT),
  `rizzoaiacademy/rizzo-pii-it-dataset` solo `synthetic/*` (IT).
- **(h) Italiano**: Nemotron-Safety-Guard-v3 (it), Nemotron-SFT-Safety-v2
  (it), `yanismiraoui/prompt_injections`, `gretelai/synthetic_pii_finance_multilingual`.
- **Ragionamento (Accurate)**: `nvidia/Nemotron-Content-Safety-Reasoning-Dataset`.

## Rischi aperti

- **Termini dei generatori**: i set più completi per guardrail (WildGuardMix,
  WildJailbreak, PolyGuardMix, GuardReasonerTrain, DynaBench, CoCoNot,
  OR-Bench) hanno testi o label prodotti da GPT-4/4o o Gemini. La licenza del
  repository non neutralizza i termini d'uso del generatore.
- **Licenze a valle più permissive delle fonti**: GuardReasonerTrain (MIT) e
  DynaBenchSafetyMix (MIT) includono BeaverTails e ToxicChat (CC-BY-NC-4.0);
  textdetox (openrail++) include AMI (CC-BY-NC-SA); KoalaAI/enguard e
  neuralchemy aggregano fonti con licenze diverse; i mirror Kaggle dichiarano
  CC0 su dati di competizione; ai4privacy open-500k è CC-BY ma vincolato alla
  licenza Llama 3.1.
- **Italiano tradotto**: la copertura IT con licenza commerciale dipende da
  traduzioni automatiche di dati inglesi; i fenomeni nativi italiani (hate
  speech, misoginia, registro colloquiale) sono coperti solo da dataset NC.
- **Contaminazione**: XSTest, ToxicChat, HarmBench, WildGuardTest e
  BeaverTails test compaiono dentro aggregati di training; serve una
  deduplicazione per hash e near-duplicate contro tutti i benchmark sopra.
- **Contenuti**: quasi tutti i set contengono testo estremamente dannoso
  (anche CBRN e autolesionismo); restano nel perimetro autorizzato e non nel
  repository, come richiesto da AGENTS.md.
- **Mappatura sul tri-state**: nessun dataset ha `INSUFFICIENT`. Va costruito
  (es. stato privo dell'evidenza richiesta dal criterio) e misurato; le label
  binarie non lo forniscono.

## Fonti

Letti il 15 ottobre 2026. Per i dataset HF la fonte è la pagina del dataset;
licenza, gating e tag sono stati letti dall'API corrispondente.

[adr]: ../adr/0001-tier-tipizzato-ragionante-con-escalation.md
[oai]: https://openai.com/policies/terms-of-use/
[oaitou]: https://open.windriver.com/info/uni-license-list/licenses/openai-tou-20241211.html
[gemma]: https://ai.google.dev/gemma/terms
[llama3]: https://github.com/meta-llama/llama3/blob/main/LICENSE
[llama31]: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
[tower]: https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2
[nllb]: https://huggingface.co/facebook/nllb-200-3.3B
[cg]: https://arxiv.org/abs/2508.01710
[wg]: https://arxiv.org/abs/2406.18495
[wtm]: https://arxiv.org/abs/2406.18510
[pg]: https://arxiv.org/abs/2504.04377
[dg]: https://arxiv.org/abs/2509.02563
[grp]: https://arxiv.org/abs/2501.18492
[orbp]: https://arxiv.org/abs/2405.20947
[asbp]: https://arxiv.org/abs/2412.14470
[cocop]: https://arxiv.org/abs/2407.12043
[aeg2]: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
[aeg1]: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
[nsg3]: https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3
[sft2]: https://huggingface.co/datasets/nvidia/Nemotron-SFT-Safety-v2
[sft1]: https://huggingface.co/datasets/nvidia/Nemotron-SFT-Safety-v1
[wgm]: https://huggingface.co/datasets/allenai/wildguardmix
[wjb]: https://huggingface.co/datasets/allenai/wildjailbreak
[pgm]: https://huggingface.co/datasets/ToxicityPrompts/PolyGuardMix
[pgp]: https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts
[mwgm]: https://huggingface.co/datasets/ToxicityPrompts/mWildGuardMix-train-tower-nllb-v2.1
[salad]: https://huggingface.co/datasets/OpenSafetyLab/Salad-Data
[hh]: https://huggingface.co/datasets/Anthropic/hh-rlhf
[aya]: https://huggingface.co/datasets/CohereForAI/aya_redteaming
[gsa]: https://huggingface.co/datasets/gretelai/gretel-safety-alignment-en-v1
[attaq]: https://huggingface.co/datasets/ibm-research/AttaQ
[dna]: https://huggingface.co/datasets/LibrAI/do-not-answer
[koala]: https://huggingface.co/datasets/KoalaAI/Text-Moderation-Multilingual
[enguard]: https://huggingface.co/datasets/enguard/multi-lingual-prompt-moderation
[civil]: https://huggingface.co/datasets/google/civil_comments
[jig]: https://huggingface.co/datasets/google/jigsaw_toxicity_pred
[tdx]: https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset
[dps]: https://huggingface.co/datasets/deepset/prompt-injections
[rikka]: https://huggingface.co/datasets/rikka-snow/prompt-injection-multilingual
[yanis]: https://huggingface.co/datasets/yanismiraoui/prompt_injections
[jack]: https://huggingface.co/datasets/jackhhao/jailbreak-classification
[twj]: https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts
[moss]: https://huggingface.co/datasets/Lakera/mosscap_prompt_injection
[gandalf]: https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions
[llmail]: https://huggingface.co/datasets/microsoft/llmail-inject-challenge
[hap]: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
[spml]: https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection
[neur]: https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset
[xtram]: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
[jaya]: https://huggingface.co/datasets/jayavibhav/prompt-injection
[necent]: https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset
[nailyk]: https://huggingface.co/datasets/Nailyk14/prompt-safety-multilingual
[bipia]: https://huggingface.co/datasets/MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT
[mpdd]: https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd
[ptd]: https://huggingface.co/datasets/tasksource/procedural-typed-decisions
[oj]: https://huggingface.co/datasets/ZefanCai/Open-Jev
[oj11]: https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1
[dynab]: https://huggingface.co/datasets/montehoover/DynaBench
[ctat]: https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset
[ncsr]: https://huggingface.co/datasets/nvidia/Nemotron-Content-Safety-Reasoning-Dataset
[vpg]: https://huggingface.co/datasets/Virtue-AI-HUB/PolyGuard
[orb]: https://huggingface.co/datasets/bench-llm/or-bench
[coco]: https://huggingface.co/datasets/allenai/coconot
[miiref]: https://huggingface.co/datasets/mii-llm/refusal-ita
[nipi]: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1
[npii]: https://huggingface.co/datasets/nvidia/Nemotron-PII
[gpii]: https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual
[rpii]: https://huggingface.co/datasets/rizzoaiacademy/rizzo-pii-it-dataset
[ai5]: https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy
[ai4]: https://huggingface.co/datasets/ai4privacy/pii-masking-400k
[dmpii]: https://huggingface.co/datasets/DeepMount00/pii-masking-ita
[haspe]: https://huggingface.co/datasets/evalitahf/hatespeech_detection
[ami]: https://huggingface.co/datasets/RiTA-nlp/ami_2020
[malert]: https://huggingface.co/datasets/felfri/M-ALERT
[heretic]: https://huggingface.co/datasets/heretic-org/Multilingual-Harmless-Harmful
[inlp]: https://huggingface.co/datasets/iNLP-Lab/multilingual-safety
[hasan]: https://huggingface.co/datasets/hasankursun/multilingual-safety-classification-dataset
[yatav]: https://huggingface.co/datasets/YATAV-ENT/aegis-multilingual-guard-dataset-v0.1-samples
[honest]: https://huggingface.co/datasets/MilaNLProc/honest
[kmh]: https://www.kaggle.com/datasets/wajidhassanmoosa/multilingual-hatespeech-dataset
[kjm]: https://www.kaggle.com/datasets/julian3833/jigsaw-multilingual-toxic-comment-classification
[kjt]: https://www.kaggle.com/datasets/miklgr500/jigsaw-train-multilingual-coments-google-api
[grt]: https://huggingface.co/datasets/yueliu1999/GuardReasonerTrain
[xst]: https://huggingface.co/datasets/Paul/XSTest
[wxst]: https://huggingface.co/datasets/walledai/XSTest
[xstr]: https://huggingface.co/datasets/allenai/xstest-response
[xstit]: https://huggingface.co/datasets/saiteki-kai/XSTest-IT
[rtplx]: https://huggingface.co/datasets/ToxicityPrompts/RTP-LX
[ptp]: https://huggingface.co/datasets/ToxicityPrompts/PolygloToxicityPrompts
[xsafe]: https://huggingface.co/datasets/ToxicityPrompts/XSafety
[mj]: https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail
[hci]: https://huggingface.co/datasets/Paul/hatecheck-italian
[hc]: https://huggingface.co/datasets/Paul/hatecheck
[tc]: https://huggingface.co/datasets/lmsys/toxic-chat
[jbb]: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
[hb]: https://huggingface.co/datasets/walledai/HarmBench
[advb]: https://huggingface.co/datasets/walledai/AdvBench
[sst]: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests
[alert]: https://huggingface.co/datasets/Babelscape/ALERT
[alertdpo]: https://huggingface.co/datasets/Babelscape/ALERT_DPO
[sorry]: https://huggingface.co/datasets/sorry-bench/sorry-bench-202503
[q3gt]: https://huggingface.co/datasets/Qwen/Qwen3GuardTest
[jevb]: https://huggingface.co/datasets/Praveenrajus/jev-bench
[f80]: https://huggingface.co/datasets/apus-ailab/APUS-OpenJev-Eval-Frozen80
[rules]: https://huggingface.co/datasets/normster/RuLES
[ah]: https://huggingface.co/datasets/ai-safety-institute/AgentHarm
[asb]: https://huggingface.co/datasets/thu-coai/Agent-SafetyBench
[aiq]: https://huggingface.co/datasets/nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0
[qpi]: https://huggingface.co/datasets/qualifire/prompt-injections-benchmark
