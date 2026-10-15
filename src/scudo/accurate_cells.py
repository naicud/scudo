"""Celle proprie del notebook Colab dell'Accurate v0.1 (issue #9), come stringhe Python.

Le genera `scudo.accurate_notebook`; i segnaposto `@...@` li riempie il generatore
con le revisioni fissate nel repository. Setup, addestramento ed export GGUF
sono le celle del tracer (`scudo.notebook`), non copie.
"""

INTRO = """\
# Scudo - Accurate v0.1 (issue #9)

SFT di `Qwen/Qwen3.5-9B` (Apache-2.0, revisione fissata) sui ragionamenti di
`nvidia/Nemotron-Content-Safety-Reasoning-Dataset`, tenuti solo se il loro
verdetto finale coincide con la label, piu' i file del teacher (#7) se presenti.
Formato `scudo.reasoned-decision/1`: prompt decision-token, poi `Reasoning:`,
un ragionamento limitato chiuso da `</reasoning>`, poi `Answer:` e i logit di
A, B, C (vedi `docs/protocol.md`). L'eval usa lo stesso harness delle baseline
(`scudo.benchmarks`, `scudo.runner`) con il passo ragionato di
`scudo.reasoned_runner`, su un campione per benchmark del registro.

Prerequisiti:

- runtime Colab con GPU A100 (40 GB);
- Colab Secrets `HF_TOKEN` (scrittura su `Naicud/scudo-accurate-checkpoints`,
  lettura dei dataset e dei benchmark del manifest, compresi quelli gated) e
  `GH_TOKEN` (lettura del repository privato `naicud/scudo`);
- dopo una disconnessione basta rieseguire tutte le celle: l'addestramento
  riprende da `last-checkpoint` sul checkpoint repo, solo con la stessa
  configurazione e gli stessi dati.

Attenzione: la documentazione Unsloth sconsiglia QLoRA 4-bit su Qwen3.5
("higher than normal quantization differences"). Il ticket chiede QLoRA;
`RUN["load_in_4bit"] = False` passa a LoRA bf16 (circa 22 GB per il 9B secondo
Unsloth). La scelta e' della persona che lancia il run.

Questo notebook e' una prova locale (checkpoint in training), non la prova live
nel backend nativo."""

CONFIG = """\
CACHE_DIR = "/content/reasoning-data/hf-cache"  # la stessa cache della build dei dati
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"  # prima di qualsiasi import di huggingface_hub

from scudo import reasoning_data
from scudo.reasoned import MAX_NEW_TOKENS, REASONED_FORMAT_VERSION

FORMAT_VERSION = REASONED_FORMAT_VERSION  # nome letto dalla cella di training del tracer
RUN = {
    "base_model": "@BASE_MODEL@",
    "base_revision": "@BASE_REVISION@",
    "base_license": "@BASE_LICENSE@",  # cardData.license e LICENSE alla revisione fissata
    "format": REASONED_FORMAT_VERSION,
    "max_new_tokens": MAX_NEW_TOKENS,
    "dataset": reasoning_data.DATASET,
    "dataset_revision": "@NCSR_REVISION@",  # fissato nel notebook, non letto a runtime
    "extra": [],  # JSONL di ReasoningRecord del teacher (#7), nel perimetro autorizzato
    "eval_per_benchmark": 100,  # ogni caso costa una generazione: campione deterministico
    "seed": 3407,
    "checkpoint_repo": "Naicud/scudo-accurate-checkpoints",
    "save_steps": 100,
    "max_seq_length": 2048,
    "load_in_4bit": True,  # ticket #9; vedi l'avvertenza Unsloth nell'introduzione
    "learning_rate": 2e-4,
    "batch_size": 4,
    "grad_accum": 4,
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 16,
}
assert RUN["dataset_revision"] == reasoning_data.REVISION
SEED = RUN["seed"]
CKPT = RUN["checkpoint_repo"]
RUN_LOG["base"] = {k: RUN[k] for k in ("base_model", "base_revision", "base_license")}
RUN_LOG["format"] = RUN["format"]
RUN_LOG["quantization"] = "qlora-4bit" if RUN["load_in_4bit"] else "lora-bf16"
print(json.dumps(RUN, indent=2))"""

DATA = """\
from collections import Counter
from pathlib import Path

from scudo.artifacts import file_sha256
from scudo.ingest import verify_dedup

# Build riproducibile fuori dal clone: cancello del manifest, digest del file,
# filtro sull'accordo con la label, deduplica contro tutti i benchmark.
# NON VERIFICATO su Colab: l'indice dei benchmark scarica tutti i benchmark del
# manifest; se uno non e' accessibile con HF_TOKEN la build si ferma qui.
DATA_DIR = Path("/content/reasoning-data")
command = [sys.executable, "-m", "scudo.reasoning_data", "build", "--out", str(DATA_DIR),
           "--seed", str(SEED)]
for path in RUN["extra"]:
    command += ["--extra", path]
subprocess.run(command, check=True, cwd=REPO_DIR)
data_report = json.loads((DATA_DIR / "reasoning-report.json").read_text(encoding="utf-8"))
DEDUP = verify_dedup(data_report)  # PASS solo se ha indicizzato tutto il registro
with open(DATA_DIR / "reasoning.jsonl", encoding="utf-8") as handle:
    reasoning_rows = [json.loads(line) for line in handle if line.strip()]
RUN["data_sha256"] = file_sha256(DATA_DIR / "reasoning.jsonl")  # entra nel fingerprint
RUN_LOG["data"] = {"sha256": RUN["data_sha256"], "rows": len(reasoning_rows),
                   "report": data_report}
RUN_LOG["dedup_verification"] = DEDUP

for name, stats in data_report["sources"].items():
    print(name, {k: stats[k] for k in ("kept", "disagree", "removed_exact", "removed_near")})
print("righe di training", len(reasoning_rows))
print("  per generatore", dict(Counter(r["generator"] for r in reasoning_rows)))
print("  per verdetto  ", dict(Counter(r["verdict"] for r in reasoning_rows)))
print("deduplica", DEDUP["status"], DEDUP["problems"])"""

MODEL = """\
from datasets import Dataset
from unsloth import FastLanguageModel

from scudo.reasoned import ANSWER_PREFIX, REASONING_PREFIX, REASONING_STOP
from scudo.runner import NO_SUBFOLDERS, ROOT_FILES, fetch_verified, letter_ids

# Ogni file della base verificato alla revisione fissata prima del loader.
# text_only: carica solo il modello testuale (parametro di Unsloth 2026.10.5).
base_root, BASE_WEIGHTS = fetch_verified(RUN["base_model"], RUN["base_revision"],
                                         ROOT_FILES, NO_SUBFOLDERS)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(base_root),
    use_exact_model_name=True,  # niente rimappatura su un mirror pre-quantizzato
    max_seq_length=RUN["max_seq_length"],
    load_in_4bit=RUN["load_in_4bit"],
    load_in_16bit=not RUN["load_in_4bit"],
    text_only=True,
)
tok = getattr(tokenizer, "tokenizer", tokenizer)


def enc(text):
    return tok.encode(text, add_special_tokens=False)


# Nessun chat template: testo grezzo, come nel backend nativo.
LETTER_IDS = letter_ids(tok)
ANSWER_IDS = enc(ANSWER_PREFIX)

# Righe con i confini di token del frame nativo e dentro budget e contesto;
# le altre escono contate (troncarle insegnerebbe ragionamenti senza risposta).
kept, dropped = [], Counter()
for row in reasoning_rows:
    prompt, decision_prompt = row["prompt"], row["prompt"][:-len(REASONING_PREFIX)]
    prompt_ids = enc(prompt)
    through_stop = enc(prompt + row["reasoning"] + REASONING_STOP)
    full = enc(prompt + row["completion"])
    if prompt_ids != enc(decision_prompt) + enc(REASONING_PREFIX):
        dropped["confine prompt/prefisso"] += 1
    elif through_stop[:len(prompt_ids)] != prompt_ids:
        dropped["confine prefisso/ragionamento"] += 1
    elif full != through_stop + ANSWER_IDS + [LETTER_IDS[row["completion"][-1]]]:
        dropped["confine stop/risposta"] += 1
    elif len(through_stop) - len(prompt_ids) > RUN["max_new_tokens"]:
        dropped["ragionamento oltre il budget"] += 1
    elif len(full) + 1 > RUN["max_seq_length"]:  # +1: EOS aggiunto da TRL
        dropped["oltre max_seq_length"] += 1
    else:
        kept.append({"prompt": prompt, "completion": row["completion"]})
RUN_LOG["train_rows"] = {"kept": len(kept), "dropped": dict(dropped)}
print("righe di training tenute", len(kept), "scartate", dict(dropped))
assert kept, "nessuna riga di training"
dataset = Dataset.from_list(kept)"""

LORA = """\
# Qwen3.5-9B alla revisione fissata (config.json e indice dei pesi): 32 layer,
# 8 full_attention (q/k/v/o_proj) e 24 linear_attention (in_proj_qkv, in_proj_z,
# out_proj); i nomi della torre visiva (qkv, proj, linear_fc*) non coincidono.
model = FastLanguageModel.get_peft_model(
    model,
    r=RUN["lora_r"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=RUN["lora_alpha"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
model.print_trainable_parameters()"""
