"""Genera il notebook Colab dell'eval (issue #4): il nostro Fast e le baseline sui benchmark.

Come `scudo.notebook` per il tracer: le celle vivono qui come stringhe e
`python -m scudo.eval_notebook` riscrive `notebooks/eval.ipynb`. Solo standard
library; le dipendenze pesanti si importano dentro le celle, su Colab.
"""

import json
from pathlib import Path

from scudo.notebook import _cell

NOTEBOOK_PATH = "notebooks/eval.ipynb"
FAST_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # Qwen/Qwen3.5-4B, come il tracer

INTRO = """\
# Scudo - eval di Fast e baseline (issue #4)

Un run per modello sui benchmark del registro (`scudo.manifest.benchmarks()`), con
un report per benchmark: accuratezza, F1 per verdetto, calibrazione e over-refusal
per lingua e superficie (`docs/eval.md`). Ogni report cita SHA di scudo, identita'
e digest dei pesi, runtime, tipo di prova, conteggi del benchmark e verifica della
deduplica dell'ingestione.

- Baseline: prova `reference` (pesi pubblicati nel runtime transformers).
- Il nostro Fast: prova `local` (checkpoint in training, loader Unsloth del tracer).
- Rifiuti espliciti, mai silenziosi: WildGuardTest (gated, campi NON VERIFICATI),
  Rizzo Flow (codice remoto incompatibile con transformers 5.17) e APUS-Reflex
  (runtime della card con transformers==5.16.1) finiscono in `skipped_models` o
  `refused_benchmarks` con il motivo.

Prerequisiti: runtime Colab A100; Colab Secrets `HF_TOKEN` (lettura e scrittura su
`Naicud/scudo-checkpoints`) e `GH_TOKEN` (lettura di `naicud/scudo`). Dopo una
disconnessione si rieseguono setup, config, dati e helper, poi le celle dei modelli
mancanti: ogni modello carica i suoi report appena finisce."""

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
CACHE_DIR = "/content/hf-cache"
os.environ["HF_HUB_CACHE"] = f"{CACHE_DIR}/hub"  # prima di qualsiasi import di huggingface_hub

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
from pathlib import Path

EVAL = {
    "checkpoint_repo": "Naicud/scudo-checkpoints",  # privato: report in eval/<scudo_sha>/
    "report_dir": "/content/eval-reports",
    "seed": 3407,  # permutazione delle opzioni del Fast
    # Report della build di ingestione (`python -m scudo.ingest build`), in un repository
    # privato a un commit fissato. Vuoto: verifica della deduplica NOT_RUN nei report.
    "ingest_report": {"repo": "", "repo_type": "dataset", "revision": "", "path": "report.json"},
    # Il nostro Fast: base fissata e commit dell'upload dell'adapter. Vuoto: Fast saltato.
    "fast": {
        "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": "@FAST_BASE_REVISION@",
        "adapter_repo": "Naicud/scudo-checkpoints",
        "adapter_revision": "",
        "adapter_dir": "adapter",
        "max_seq_length": 2048,
    },
}
RUN_LOG["eval"] = EVAL
print(json.dumps(EVAL, indent=2))"""

DEDUP = """\
from huggingface_hub import hf_hub_download

from scudo.ingest import verify_dedup

# I benchmark sono esclusi dal training solo se la deduplica di quella build li ha
# indicizzati tutti: un report che salta un benchmark del registro e' FAIL.
source = EVAL["ingest_report"]
if source["repo"] and source["revision"]:
    path = hf_hub_download(source["repo"], source["path"], repo_type=source["repo_type"],
                           revision=source["revision"])
    DEDUP = verify_dedup(json.loads(Path(path).read_text(encoding="utf-8")))
    DEDUP["source"] = dict(source)
else:
    DEDUP = {"status": "NOT_RUN", "problems": ["report di ingestione non configurato"]}
RUN_LOG["dedup_verification"] = DEDUP
print(json.dumps(DEDUP, indent=2, ensure_ascii=False))"""

DATA = """\
from functools import partial

from scudo.benchmarks import BenchmarkError, load_cases
from scudo.ingest import load_rows
from scudo.manifest import benchmarks

loader = partial(load_rows, cache_dir=Path(CACHE_DIR))
LOADED, REFUSED = {}, {}
for entry in benchmarks():
    try:
        LOADED[entry.name] = load_cases(entry, loader)
    except BenchmarkError as error:  # rifiuto prima di qualsiasi download (es. WildGuardTest)
        REFUSED[entry.name] = str(error)
for name, loaded in LOADED.items():
    print(json.dumps(loaded.summary(), ensure_ascii=False))
for name, reason in REFUSED.items():
    print("RIFIUTATO", name, "-", reason)
RUN_LOG["benchmarks"] = {name: loaded.summary() for name, loaded in LOADED.items()}
RUN_LOG["refused_benchmarks"] = REFUSED"""

HELPERS = """\
import gc
from collections import Counter

import torch
import transformers
from huggingface_hub import HfApi

from scudo.baselines import BASELINES
from scudo.runner import (
    NO_SUBFOLDERS, ROOT_FILES, RunnerError, TransformersBackend, fast_scorer,
    fetch_verified, load_baseline, reports_for, run, scorer_for,
)

REPORT_DIR = Path(EVAL["report_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)
api = HfApi()
api.create_repo(EVAL["checkpoint_repo"], private=True, exist_ok=True)
SUMMARY = {"scudo_sha": RUN_LOG["scudo_sha"], "dedup_verification": DEDUP["status"],
           "refused_benchmarks": REFUSED, "skipped_models": {}, "models": {}}


def headline(report):
    overall = report["overall"]
    calibration = overall["calibration"] or {}
    over_refusal = overall["over_refusal"] or {}
    return {"cases": overall["cases"], "errors": overall["errors"],
            "accuracy": overall["accuracy"], "f1": overall["f1"],
            "ece": calibration.get("ece"), "brier": calibration.get("brier"),
            "over_refusal_rate": over_refusal.get("rate")}


def evaluate(model, score, **header):
    \"\"\"Un run su ogni benchmark caricato; report scritti e caricati sul repository privato.\"\"\"
    out = {}
    for name, loaded in LOADED.items():
        results = run(loaded, score)
        reports = reports_for(loaded, results, dedup=DEDUP, model=model,
                              scudo_sha=RUN_LOG["scudo_sha"], harness="scudo.eval_notebook",
                              **header)
        for split, report in reports.items():
            path = REPORT_DIR / model / (split.replace("/", "__") + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\\n",
                            encoding="utf-8")
            out[split] = headline(report)
        out[name]["error_codes"] = dict(Counter(item.error for item in results if item.error))
        print(model, name, json.dumps(out[name], ensure_ascii=False))
    api.upload_folder(folder_path=str(REPORT_DIR / model), repo_id=EVAL["checkpoint_repo"],
                      path_in_repo=f"eval/{RUN_LOG['scudo_sha']}/{model}",
                      commit_message=f"eval {model}")
    SUMMARY["models"][model] = out


def evaluate_baseline(name):
    \"\"\"Prova reference in transformers; un rifiuto del runner salta il modello con il motivo.\"\"\"
    try:
        model, tokenizer, weights = load_baseline(name)  # digest verificati prima del load
        score = scorer_for(name, tokenizer, TransformersBackend(model))
    except RunnerError as error:
        SUMMARY["skipped_models"][name] = str(error)
        print("SALTATA", name, "-", error)
        return
    evaluate(name, score, proof_type="reference", weights=weights,
             runtime={"name": "transformers", "version": transformers.__version__},
             format=BASELINES[name].prompt_format)
    del model, score
    gc.collect()
    torch.cuda.empty_cache()"""

GRANITE_POLICY = """\
evaluate_baseline("granite-4.1-3b-policy-guardrails")  # base + adapter LoRA, entrambi verificati"""

GUARDIAN = """\
evaluate_baseline("granite-guardian-4.1-8b")"""

QWEN3GUARD = """\
# La superficie `response` e' UNSUPPORTED (docs/eval.md): quei casi sono errori nel report.
evaluate_baseline("qwen3guard-gen-4b")"""

RIZZO = """\
# Rizzo Flow richiede trust_remote_code (modeling_spark.py, verificato con il blob git
# alla revisione fissata), ma configuration_spark.py non si carica con transformers 5.17:
# il runner la rifiuta e il motivo finisce in SUMMARY["skipped_models"].
evaluate_baseline("rizzo-flow-4b")"""

APUS = """\
# APUS-Reflex gira solo nel runtime della card (openjet_runtime, transformers==5.16.1),
# incompatibile con l'extra train: rifiuto esplicito, nessun ambiente misto.
evaluate_baseline("apus-reflex-4b")"""

FAST = """\
# Ultima cella dei modelli: Unsloth modifica transformers all'import e non deve
# toccare le baseline. Stesso loader del tracer (text_only); se il loader o l'adapter
# non combaciano la cella fallisce, senza fallback.
FAST = EVAL["fast"]
if not FAST["adapter_revision"]:
    SUMMARY["skipped_models"]["scudo-fast"] = "adapter non configurato (EVAL['fast'])"
    print("SALTATO scudo-fast: imposta EVAL['fast']['adapter_revision']")
else:
    import unsloth
    from peft import PeftModel
    from unsloth import FastLanguageModel

    base_root, base_weights = fetch_verified(FAST["base_model"], FAST["base_revision"],
                                             ROOT_FILES, NO_SUBFOLDERS)
    adapter_root, adapter_weights = fetch_verified(
        FAST["adapter_repo"], FAST["adapter_revision"], (f"{FAST['adapter_dir']}/*",))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_root), max_seq_length=FAST["max_seq_length"],
        load_in_4bit=False, load_in_16bit=True, text_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_root / FAST["adapter_dir"]))
    assert any("lora_A" in name for name, _ in model.named_modules()), "adapter non applicato"
    FastLanguageModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    evaluate("scudo-fast", fast_scorer(tok, TransformersBackend(model), seed=EVAL["seed"]),
             proof_type="local", weights=[*base_weights, *adapter_weights],
             runtime={"name": "unsloth", "version": unsloth.__version__})"""

SUMMARY_CELL = """\
# Da incollare nell'issue #4: SHA, verifica della deduplica, benchmark rifiutati,
# modelli saltati con il motivo, metriche principali per modello e benchmark.
RUN_LOG["summary"] = SUMMARY
(REPORT_DIR / "summary.json").write_text(
    json.dumps(RUN_LOG, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
api.upload_file(path_or_fileobj=str(REPORT_DIR / "summary.json"),
                path_in_repo=f"eval/{RUN_LOG['scudo_sha']}/summary.json",
                repo_id=EVAL["checkpoint_repo"], commit_message="riepilogo eval")
print(json.dumps(SUMMARY, indent=2, ensure_ascii=False))"""

OUTRO = """\
## Dopo il notebook

I numeri valgono per i pesi, le revisioni e l'harness dichiarati nei report. Una
verifica della deduplica `FAIL` o `NOT_RUN` va riportata accanto alle metriche:
finche' WildGuardTest non ha campi verificati il cancello blocca la build e nessun
report puo' passarla. Rizzo Flow e APUS-Reflex restano da misurare in un ambiente
separato con il loro runtime."""

CELLS = (
    ("markdown", "intro", INTRO),
    ("code", "setup", SETUP),
    ("code", "config", CONFIG.replace("@FAST_BASE_REVISION@", FAST_BASE_REVISION)),
    ("code", "dedup", DEDUP),
    ("code", "data", DATA),
    ("code", "helpers", HELPERS),
    ("code", "granite-policy", GRANITE_POLICY),
    ("code", "guardian", GUARDIAN),
    ("code", "qwen3guard", QWEN3GUARD),
    ("code", "rizzo", RIZZO),
    ("code", "apus", APUS),
    ("code", "fast", FAST),
    ("code", "summary", SUMMARY_CELL),
    ("markdown", "outro", OUTRO),
)


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
    target = Path(__file__).resolve().parents[2] / NOTEBOOK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
