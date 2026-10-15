"""Genera il notebook Colab del tracer veloce (issue #2).

Il notebook è un artefatto generato: le celle vivono qui come stringhe Python e
`python -m scudo.notebook` riscrive `notebooks/fast_tracer.ipynb`. Solo standard
library; le dipendenze pesanti vengono importate dentro le celle, su Colab.
"""

import json
from pathlib import Path

from scudo import nemotron

NOTEBOOK_PATH = "notebooks/fast_tracer.ipynb"

INTRO = """\
# Scudo - tracer veloce (issue #2)

LoRA veloce di `Qwen/Qwen3.5-4B` su 2.000 casi IT/EN di Nemotron Safety Guard v3,
nel formato decision-token di open-guard: il prompt termina con `Decision:` e la
risposta e' una sola lettera (A, B o C) che mappa a VIOLATION, NO_VIOLATION o
INSUFFICIENT con un ordine delle opzioni diverso per ogni caso.

Prerequisiti:

- runtime Colab con GPU A100 (40 GB);
- Colab Secrets `HF_TOKEN` (scrittura su `Naicud/scudo-checkpoints`) e `GH_TOKEN`
  (lettura del repository privato `naicud/scudo`), con accesso da notebook attivo;
- dopo una disconnessione basta rieseguire tutte le celle: l'addestramento
  riprende da `last-checkpoint` sul checkpoint repo.

Questo notebook e' una prova locale (checkpoint in training), non la prova live
nel backend nativo."""

SETUP = """\
!nvidia-smi

import json
import os
import shutil
import subprocess
import sys

from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")  # mai stampato
GH_TOKEN = userdata.get("GH_TOKEN")

SCUDO_REF = "main"
REPO_DIR = "/content/scudo"
RUN_LOG = {}

if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
try:
    # Output catturato: l'URL con il token non deve finire nei log.
    subprocess.run(
        ["git", "clone", "--quiet",
         f"https://x-access-token:{GH_TOKEN}@github.com/naicud/scudo.git", REPO_DIR],
        check=True, capture_output=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
except subprocess.CalledProcessError:
    raise RuntimeError("clone di naicud/scudo fallito: controlla il secret GH_TOKEN") from None
subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin",
                "https://github.com/naicud/scudo.git"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "checkout", "--quiet", SCUDO_REF], check=True)
RUN_LOG["scudo_ref"] = SCUDO_REF
RUN_LOG["scudo_sha"] = subprocess.run(
    ["git", "-C", REPO_DIR, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
).stdout.strip()
print("scudo", RUN_LOG["scudo_sha"])

!pip install -q -e "/content/scudo[train]"
sys.path.insert(0, f"{REPO_DIR}/src")  # l'editable install non e' visibile al kernel gia' avviato"""

CONFIG = """\
from scudo import nemotron
from scudo.protocol import FORMAT_VERSION

RUN = {
    "base_model": "Qwen/Qwen3.5-4B",
    "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "dataset": nemotron.DATASET,
    "dataset_revision": "@REVISION@",  # fissato nel notebook, non letto a runtime
    "format": FORMAT_VERSION,
    "train_per_language": 1000,
    "heldout_per_language": 200,
    "seed": 3407,
    "checkpoint_repo": "Naicud/scudo-checkpoints",
    "save_steps": 25,
    "max_seq_length": 2048,
    "learning_rate": 2e-4,
    "batch_size": 8,
    "grad_accum": 2,
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 16,
}
SEED = RUN["seed"]
CKPT = RUN["checkpoint_repo"]
RUN_LOG["base"] = {"model": RUN["base_model"], "revision": RUN["base_revision"]}
RUN_LOG["dataset"] = {"name": RUN["dataset"], "revision": RUN["dataset_revision"]}
RUN_LOG["format"] = RUN["format"]
print(json.dumps(RUN, indent=2))"""

DATA = """\
from collections import Counter

from datasets import Dataset
from huggingface_hub import hf_hub_download

from scudo.nemotron import (
    DATASET, LANGUAGES, REVISION, heldout_cases, sample_cases, training_rows,
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split(split):
    rows = []
    for lang in LANGUAGES:
        path = hf_hub_download(
            DATASET, f"{lang}/{split}.jsonl", repo_type="dataset", revision=REVISION
        )
        rows.extend(read_jsonl(path))
    return rows


train_rows_raw = load_split("train")
test_rows_raw = load_split("test")

train_cases = sample_cases(
    train_rows_raw, per_language=RUN["train_per_language"], seed=SEED
)
held_cases = heldout_cases(
    test_rows_raw, train_cases, per_language=RUN["heldout_per_language"], seed=SEED
)
train_rows = training_rows(train_cases, seed=SEED)
dataset = Dataset.from_list(
    [{"prompt": row["prompt"], "completion": row["completion"]} for row in train_rows]
)

for name, cases in (("train", train_cases), ("heldout", held_cases)):
    print(name, len(cases))
    print("  per lingua   ", dict(Counter(c.language for c in cases)))
    print("  per superficie", dict(Counter(c.surface for c in cases)))
    print("  per verdetto ", dict(Counter(c.verdict for c in cases)))"""

MODEL = """\
from unsloth import FastLanguageModel

# `text_only` e' nel loader Unsloth ma non e' confermato nella documentazione
# Qwen3.5: se il loader non lo accetta, la cella fallisce qui e va rivista.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=RUN["base_model"],
    revision=RUN["base_revision"],
    max_seq_length=RUN["max_seq_length"],
    load_in_4bit=False,
    load_in_16bit=True,
    text_only=True,
)
tok = getattr(tokenizer, "tokenizer", tokenizer)

# Nessun chat template: il prompt e' grezzo, come nel backend nativo.
LETTER_IDS = {}
for letter in "ABC":
    ids = tok.encode(letter, add_special_tokens=False)
    assert len(ids) == 1, f"la lettera {letter} non e' un singolo token: {ids}"
    LETTER_IDS[letter] = ids[0]
assert len(set(LETTER_IDS.values())) == 3

# La lettera dopo "Decision:\\n" non deve fondersi con il newline precedente.
sample_prompt = train_rows[0]["prompt"]
base_ids = tok.encode(sample_prompt, add_special_tokens=False)
assert tok.encode(sample_prompt + "A", add_special_tokens=False) == base_ids + [LETTER_IDS["A"]]
print("lettere", LETTER_IDS, "- prompt di esempio:", len(base_ids), "token")"""

LORA = """\
# Qwen3.5-4B: 8 layer full_attention (q/k/v/o_proj) e 24 linear_attention
# (in_proj_qkv, in_proj_z, out_proj), letti da config e indice dei pesi alla
# revisione fissata; senza questi ultimi la LoRA toccherebbe solo 8 layer su 32.
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

TRAIN = """\
import hashlib

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

OUTPUT_DIR = "/content/run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Un checkpoint si riprende solo se viene dalla stessa configurazione di run.
RUN_FINGERPRINT = hashlib.sha256(
    json.dumps({**RUN, "format": FORMAT_VERSION}, sort_keys=True).encode()
).hexdigest()
RUN_LOG["run_fingerprint"] = RUN_FINGERPRINT
HfApi().create_repo(CKPT, private=True, exist_ok=True)
remote_files = list_repo_files(CKPT)
if "run_config.json" in remote_files:
    with open(hf_hub_download(CKPT, "run_config.json"), encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored["fingerprint"] != RUN_FINGERPRINT:
        raise RuntimeError(
            "checkpoint di un'altra configurazione: usa un altro checkpoint_repo o svuotalo"
        )
else:
    HfApi().upload_file(
        path_or_fileobj=json.dumps(
            {"fingerprint": RUN_FINGERPRINT, "run": RUN, "format": FORMAT_VERSION}, indent=2
        ).encode(),
        path_in_repo="run_config.json", repo_id=CKPT, commit_message="configurazione del run",
    )

# Ripresa: prima un checkpoint locale, poi `last-checkpoint/` sul checkpoint repo.
resume = get_last_checkpoint(OUTPUT_DIR)
if resume is None:
    if any(name.startswith("last-checkpoint/") for name in remote_files):
        snapshot_download(CKPT, allow_patterns="last-checkpoint/*", local_dir=OUTPUT_DIR)
        resume = f"{OUTPUT_DIR}/last-checkpoint"
print("ripresa da:", resume)

config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=RUN["batch_size"],
    gradient_accumulation_steps=RUN["grad_accum"],
    num_train_epochs=RUN["epochs"],
    learning_rate=RUN["learning_rate"],
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    save_strategy="steps",
    save_steps=RUN["save_steps"],
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=CKPT,
    hub_strategy="checkpoint",
    hub_private_repo=True,
    completion_only_loss=True,
    max_length=RUN["max_seq_length"],
    bf16=True,
    logging_steps=5,
    report_to="none",
    seed=SEED,
)
trainer = SFTTrainer(
    model=model, processing_class=tok, train_dataset=dataset, args=config
)
trainer.train(resume_from_checkpoint=resume)
RUN_LOG["resumed_from"] = resume
RUN_LOG["global_step"] = trainer.state.global_step
print("global_step finale:", trainer.state.global_step)"""

EVAL = """\
import torch
from huggingface_hub import HfApi
from pathlib import Path

from scudo.artifacts import ArtifactIdentity, file_sha256
from scudo.eval_report import EvalResult, build_report
from scudo.protocol import read_options

FastLanguageModel.for_inference(model)
api = HfApi()

# Adapter: salvato, hashato e caricato; l'identita' usa lo SHA del commit di upload.
ADAPTER_DIR = Path("/content/adapter")
model.save_pretrained(str(ADAPTER_DIR))
adapter_sha = file_sha256(ADAPTER_DIR / "adapter_model.safetensors")
commit = api.upload_folder(
    folder_path=str(ADAPTER_DIR), repo_id=CKPT, path_in_repo="adapter",
    commit_message="adapter del tracer",
)
adapter_identity = ArtifactIdentity(CKPT, commit.oid, "adapter_model.safetensors", adapter_sha)
RUN_LOG["adapter"] = adapter_identity.manifest_entry()

# Eval locale: logit dell'ultima posizione sugli id di A/B/C, nessuna generazione.
held_rows = training_rows(held_cases, seed=SEED)
letter_ids = [LETTER_IDS[letter] for letter in "ABC"]
results, readouts = [], {}
for row in held_rows:
    case_id = f"{row['source_id']}:{row['surface']}:{row['language']}"
    try:
        ids = tok.encode(row["prompt"], add_special_tokens=False)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1]
        letter_logits = [float(x) for x in logits.float()[letter_ids].cpu()]
        readout = read_options(letter_logits, row["semantic_ids"])
    except Exception as exc:  # un readout fallito resta un errore, mai NO_VIOLATION
        results.append(EvalResult(case_id, row["language"], row["verdict"], None,
                                  error=type(exc).__name__))
        continue
    results.append(EvalResult(case_id, row["language"], row["verdict"], readout))
    readouts[case_id] = {
        "prompt": row["prompt"],
        "semantic_ids": row["semantic_ids"],
        "logits": list(readout.logits),
    }

report = build_report(
    results, proof_type="local", model=adapter_identity,
    split="nemotron-v3-test-heldout", harness="scudo.notebook/fast_tracer",
)
REPORT_DIR = Path("/content/reports")
REPORT_DIR.mkdir(exist_ok=True)
(REPORT_DIR / "eval_local.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8"
)
(REPORT_DIR / "readouts_local.json").write_text(
    json.dumps(readouts, ensure_ascii=False) + "\\n", encoding="utf-8"
)
api.upload_folder(
    folder_path=str(REPORT_DIR), repo_id=CKPT, path_in_repo="reports",
    commit_message="report eval locale",
)
RUN_LOG["accuracy"] = {
    "overall": report["overall"]["accuracy"],
    **{lang: m["accuracy"] for lang, m in report["by_language"].items()},
}
for lang, metrics in report["by_language"].items():
    print(f"{lang}: accuratezza {metrics['accuracy']:.3f} su {metrics['cases']} casi")
print(f"totale: {report['overall']['accuracy']:.3f}, errori {report['overall']['errors']}")"""

GGUF = """\
from scudo.artifacts import load_verified

GGUF_DIR = Path("/content/gguf")
model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# Il converter puo' scrivere anche il proiettore multimodale: va escluso.
candidates = [
    p for root in (GGUF_DIR, Path("/content/gguf_gguf")) if root.is_dir()
    for p in root.rglob("*.gguf") if "mmproj" not in p.name
]
assert len(candidates) == 1, f"attesi un solo GGUF, trovati: {candidates}"
gguf_path = candidates[0]
gguf_sha = file_sha256(gguf_path)

commit = api.upload_file(
    path_or_fileobj=str(gguf_path), path_in_repo=f"gguf/{gguf_path.name}",
    repo_id=CKPT, commit_message="GGUF q8_0 del tracer",
)
gguf_identity = ArtifactIdentity(CKPT, commit.oid, gguf_path.name, gguf_sha)

# Rilettura dal Hub: il digest deve passare prima che qualsiasi loader tocchi il file.
downloaded = Path(hf_hub_download(
    CKPT, f"gguf/{gguf_path.name}", revision=gguf_identity.revision
))
size = load_verified(
    downloaded, ArtifactIdentity(CKPT, commit.oid, downloaded.name, gguf_sha),
    lambda p: p.stat().st_size,
)
print("GGUF verificato:", gguf_path.name, size, "byte", gguf_sha)
RUN_LOG["gguf"] = gguf_identity.manifest_entry()"""

SUMMARY = """\
# Da incollare nell'issue #2: SHA di scudo, base, dataset, ripresa, step,
# identita' di adapter e GGUF, accuratezze.
print(json.dumps(RUN_LOG, indent=2, ensure_ascii=False))"""

OUTRO = """\
## Dopo il notebook

Questa e' la prova locale. Il passo live non si esegue su Colab: carica il GGUF
verificato nel backend nativo di open-guard, rileggi gli stessi casi held-out
con `readouts_local.json` e confronta i readout con
`scudo.eval_report.compare_readouts`. Fino ad allora l'accuratezza riportata
vale solo per il checkpoint locale, con la revisione e il digest dichiarati."""

CELLS = (
    ("markdown", "intro", INTRO),
    ("code", "setup", SETUP),
    ("code", "config", CONFIG.replace("@REVISION@", nemotron.REVISION)),
    ("code", "data", DATA),
    ("code", "model", MODEL),
    ("code", "lora", LORA),
    ("code", "train", TRAIN),
    ("code", "eval-local", EVAL),
    ("code", "gguf", GGUF),
    ("code", "summary", SUMMARY),
    ("markdown", "outro", OUTRO),
)


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _cell(kind: str, name: str, text: str) -> dict:
    cell = {"cell_type": kind, "id": f"scudo-{name}", "metadata": {}, "source": _lines(text)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def build() -> dict:
    return {
        "cells": [_cell(*cell) for cell in CELLS],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    target = root / NOTEBOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
