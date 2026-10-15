"""Celle Colab condivise dai notebook di training (tracer, Fast v0.1, Accurate) ed eval.

Una cella vive qui una volta sola; i generatori la importano e riempiono solo il
segnaposto `@RUN_NAME@` (es. "del tracer") con `named`. Solo standard library.
"""

import json
from pathlib import Path

RUN_NAME = "@RUN_NAME@"

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
sys.path.insert(0, f"{REPO_DIR}/src")  # l'editable install non e' visibile al kernel gia' avviato

# causal-conv1d 1.7.0 su PyPI e' solo sdist e le wheel GitHub si fermano a torch 2.10:
# con torch 2.14 pip compila da sorgente per 9 architetture CUDA (lungo; serve un nvcc
# con la stessa major CUDA di torch). Opt-in: senza, la convoluzione resta su PyTorch.
BUILD_CAUSAL_CONV1D = False
if BUILD_CAUSAL_CONV1D:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja", "packaging"],
                   check=True)
    conv_build = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                 "--no-build-isolation", "causal-conv1d==1.7.0"])
    RUN_LOG["causal_conv1d_build_returncode"] = conv_build.returncode

# Kernel linear-attention di Qwen3.5 (fla dall'extra train, causal-conv1d opt-in): il
# ripiego su PyTorch e' corretto ma lento, quindi si registra senza fermare il run.
# Sottoprocesso: `fla` importa transformers, che qui deve arrivare dopo Unsloth.
kernel_probe = subprocess.run([sys.executable, "-m", "scudo.linear_kernels"],
                              cwd=f"{REPO_DIR}/src", capture_output=True, text=True)
RUN_LOG["linear_attention_kernels"] = (
    json.loads(kernel_probe.stdout) if kernel_probe.returncode == 0
    else {"probe": f"fallita: {kernel_probe.stderr[-500:]}"}
)
for name, state in RUN_LOG["linear_attention_kernels"].items():
    print(f"{name}: {state}")"""

# L'eval e il Fast v0.1 fissano la cache Hub prima di qualsiasi import di huggingface_hub:
# `scudo.ingest.load_rows` rifiuta una cache fuori da CACHE_DIR.
_GH = 'GH_TOKEN = userdata.get("GH_TOKEN")\n'
_HUB_CACHE = (
    'CACHE_DIR = "/content/hf-cache"\n'
    'os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"'
    "  # prima di qualsiasi import di huggingface_hub\n"
)
SETUP_HUB_CACHE = SETUP.replace(_GH, _GH + _HUB_CACHE)

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

ADAPTER = """\
FastLanguageModel.for_inference(model)
api = HfApi()

# Adapter: salvato, hashato e caricato; l'identita' usa lo SHA del commit di upload.
ADAPTER_DIR = Path("/content/adapter")
model.save_pretrained(str(ADAPTER_DIR))
adapter_sha = file_sha256(ADAPTER_DIR / "adapter_model.safetensors")
commit = api.upload_folder(
    folder_path=str(ADAPTER_DIR), repo_id=CKPT, path_in_repo="adapter",
    commit_message="adapter @RUN_NAME@",
)
adapter_identity = ArtifactIdentity(CKPT, commit.oid, "adapter_model.safetensors", adapter_sha)
RUN_LOG["adapter"] = adapter_identity.manifest_entry()"""

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
    repo_id=CKPT, commit_message="GGUF q8_0 @RUN_NAME@",
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


def named(text: str, run_name: str) -> str:
    """Cella con il nome del run al posto di `@RUN_NAME@`, che deve comparire una volta."""
    if text.count(RUN_NAME) != 1:
        raise ValueError(f"{RUN_NAME}: atteso una volta nella cella")
    return text.replace(RUN_NAME, run_name)


def cell(kind: str, name: str, text: str) -> dict:
    out = {"cell_type": kind, "id": f"scudo-{name}", "metadata": {},
           "source": text.splitlines(keepends=True)}
    if kind == "code":
        out.update(outputs=[], execution_count=None)
    return out


def notebook(cells) -> dict:
    """Notebook nbformat 4 per Colab A100 dalle celle (tipo, nome, testo)."""
    return {
        "cells": [cell(*item) for item in cells],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(relative: str, built: dict) -> Path:
    """Scrive il notebook generato nel repository (percorso relativo alla radice)."""
    target = Path(__file__).resolve().parents[2] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(built, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return target
