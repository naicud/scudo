"""Genera il notebook Colab del Fast v0.1 (issue #8): mix completo con il criterio di difficoltà.

`python -m scudo.fast_notebook` riscrive `notebooks/fast_v01.ipynb`. Setup, LoRA,
addestramento, adapter ed export GGUF sono le celle condivise di
`scudo.notebook_cells`; l'eval è l'harness di #4 (`scudo.benchmarks`,
`scudo.runner`, `scudo.eval_report`) più il readout del criterio di difficoltà
(`scudo.difficulty`). Solo standard library; le dipendenze pesanti si importano
dentro le celle, su Colab.
"""

from scudo.eval_notebook import FAST_BASE_REVISION
from scudo.notebook_cells import (
    ADAPTER,
    GGUF,
    LORA,
    SETUP_HUB_CACHE,
    TRAIN,
    named,
    notebook,
    write,
)

NOTEBOOK_PATH = "notebooks/fast_v01.ipynb"
RUN_NAME = "del Fast v0.1"

INTRO = """\
# Scudo - Fast v0.1 (issue #8)

LoRA di `Qwen/Qwen3.5-4B` (revisione del tracer) sul mix del Fast v0.1, costruito
fuori da Colab con `python -m scudo.mix_cli build` e caricato su un repository
dataset privato. Il mix contiene, nello stesso formato decision-token, le righe del
criterio semantico e quelle del criterio di difficoltà `routing.difficulty/1` di
open-guard sulla stessa evidenza: VIOLATION = caso difficile, NO_VIOLATION = facile
(label del primo giro dal disaccordo dei teacher). Le due famiglie si addestrano
insieme.

Prerequisiti:

- runtime Colab con GPU A100 (40 GB);
- Colab Secrets `HF_TOKEN` (lettura del repository del mix e dei benchmark del
  manifest, scrittura su `Naicud/scudo-fast-checkpoints`) e `GH_TOKEN` (lettura di
  `naicud/scudo`);
- in `RUN` la revisione del repository del mix e lo sha256 di `mix-manifest.json`,
  stampati da `python -m scudo.mix_cli upload`: il notebook verifica ogni digest
  prima di leggere un file;
- dopo una disconnessione basta rieseguire tutte le celle: l'addestramento
  riprende da `last-checkpoint` solo con la stessa configurazione (mix compreso).

NON VERIFICATO su Colab: memoria con `max_seq_length` 4096 e batch 4 su 40 GB,
durata del run, loader Unsloth da cartella locale verificata. Questo notebook e'
una prova locale (checkpoint in training), non la prova live nel backend nativo."""

CONFIG = """\
from scudo.difficulty import ROUTING_DIFFICULTY_REF
from scudo.protocol import FORMAT_VERSION

RUN = {
    "base_model": "Qwen/Qwen3.5-4B",
    "base_revision": "@BASE_REVISION@",  # la stessa del tracer
    "format": FORMAT_VERSION,
    "difficulty_criterion": ROUTING_DIFFICULTY_REF,
    # Repository dataset privato del mix: revisione e sha256 di mix-manifest.json li
    # stampa `python -m scudo.mix_cli upload`. Vuoti: il notebook si ferma qui.
    "mix_repo": "Naicud/scudo-fast-mix",
    "mix_revision": "",
    "mix_manifest_sha256": "",
    "eval_per_benchmark": 500,  # campione deterministico per benchmark (due readout a caso)
    "seed": 3407,
    "checkpoint_repo": "Naicud/scudo-fast-checkpoints",
    "save_steps": 500,
    "max_seq_length": 4096,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "grad_accum": 4,
    "epochs": 1,
    "lora_r": 16,
    "lora_alpha": 16,
}
if not (RUN["mix_revision"] and RUN["mix_manifest_sha256"]):
    raise RuntimeError("imposta RUN['mix_revision'] e RUN['mix_manifest_sha256'] dall'upload")
SEED = RUN["seed"]
CKPT = RUN["checkpoint_repo"]
RUN_LOG["base"] = {"model": RUN["base_model"], "revision": RUN["base_revision"]}
RUN_LOG["format"] = RUN["format"]
RUN_LOG["difficulty_criterion"] = RUN["difficulty_criterion"]
print(json.dumps(RUN, indent=2))"""

DATA = """\
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from scudo.artifacts import ArtifactIdentity, load_verified
from scudo.ingest import verify_dedup


def verified_text(filename, sha256):
    \"\"\"File del mix alla revisione fissata; il digest passa prima di qualsiasi lettura.\"\"\"
    identity = ArtifactIdentity(RUN["mix_repo"], RUN["mix_revision"], filename, sha256)
    path = Path(hf_hub_download(RUN["mix_repo"], filename, repo_type="dataset",
                                revision=RUN["mix_revision"]))
    return load_verified(path, identity, lambda p: p.read_text(encoding="utf-8"))


MIX_MANIFEST = json.loads(verified_text("mix-manifest.json", RUN["mix_manifest_sha256"]))
assert MIX_MANIFEST["format"] == FORMAT_VERSION, MIX_MANIFEST["format"]
assert MIX_MANIFEST["difficulty_criterion"] == ROUTING_DIFFICULTY_REF
output, ingest = MIX_MANIFEST["output"], MIX_MANIFEST["ingest_report"]
train_rows = [json.loads(line) for line in
              verified_text(output["filename"], output["sha256"]).splitlines() if line.strip()]
assert len(train_rows) == output["rows"], "righe del mix diverse dal manifest"
# I benchmark dell'eval sono fuori dal training solo se la build ha indicizzato tutto il registro.
DEDUP = verify_dedup(json.loads(verified_text(ingest["filename"], ingest["sha256"])))
RUN_LOG["mix"] = {
    "repo": RUN["mix_repo"], "revision": RUN["mix_revision"],
    "manifest_sha256": RUN["mix_manifest_sha256"], "rows_sha256": output["sha256"],
    "built_at_scudo_sha": MIX_MANIFEST["repo_sha"], "overrides": MIX_MANIFEST["overrides"],
    "override_reason": MIX_MANIFEST["override_reason"],
}
RUN_LOG["dedup_verification"] = DEDUP
print("righe del mix", len(train_rows), dict(Counter(r["task"] for r in train_rows)))
print("composizione", json.dumps(MIX_MANIFEST["composition"], indent=1, ensure_ascii=False))
print("override", MIX_MANIFEST["overrides"], MIX_MANIFEST["override_reason"])
print("deduplica", DEDUP["status"], DEDUP["problems"])"""

MODEL = """\
from datasets import Dataset
from unsloth import FastLanguageModel

from scudo.runner import NO_SUBFOLDERS, ROOT_FILES, fetch_verified, letter_ids

# Ogni file della base verificato alla revisione fissata prima del loader, che legge
# la cartella locale (niente rimappatura su un mirror). NON VERIFICATO su Colab.
base_root, BASE_WEIGHTS = fetch_verified(RUN["base_model"], RUN["base_revision"],
                                         ROOT_FILES, NO_SUBFOLDERS)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(base_root),
    max_seq_length=RUN["max_seq_length"],
    load_in_4bit=False,
    load_in_16bit=True,
    text_only=True,
)
tok = getattr(tokenizer, "tokenizer", tokenizer)
LETTER_IDS = letter_ids(tok)  # A, B, C: un token ciascuna, distinti
RUN_LOG["base_weights"] = [w.manifest_entry() for w in BASE_WEIGHTS]

# Una riga oltre il contesto esce contata: troncarla toglierebbe `Decision:` e la lettera.
kept, dropped = [], Counter()
for start in range(0, len(train_rows), 1024):
    batch = train_rows[start:start + 1024]
    encoded = tok([row["prompt"] for row in batch], add_special_tokens=False)["input_ids"]
    for row, ids in zip(batch, encoded):
        if len(ids) + 2 > RUN["max_seq_length"]:  # lettera ed EOS aggiunto da TRL
            dropped[f"{row['task']}/{row['category']}"] += 1
        else:
            kept.append({"prompt": row["prompt"], "completion": row["completion"]})
assert kept, "nessuna riga di training"

# La lettera dopo "Decision:\\n" non deve fondersi con il newline precedente.
sample_prompt = kept[0]["prompt"]
base_ids = tok.encode(sample_prompt, add_special_tokens=False)
assert tok.encode(sample_prompt + "A", add_special_tokens=False) == base_ids + [LETTER_IDS["A"]]
RUN_LOG["train_rows"] = {"kept": len(kept), "dropped_over_max_seq_length": dict(dropped)}
print("righe tenute", len(kept), "oltre il contesto", dict(dropped))
dataset = Dataset.from_list(kept)"""

EVAL = """\
from dataclasses import replace
from functools import partial

import unsloth
from huggingface_hub import HfApi

from scudo.artifacts import ArtifactIdentity, file_sha256
from scudo.benchmarks import BenchmarkError, load_cases, sample
from scudo.difficulty import as_difficulty, routing_metrics
from scudo.ingest import load_rows
from scudo.manifest import benchmarks
from scudo.runner import TransformersBackend, fast_scorer, reports_for, run

""" + named(ADAPTER, RUN_NAME) + """

# Harness dell'eval (#4) sul campione di ogni benchmark: criterio semantico e, sullo
# stesso caso, criterio di difficoltà; INSUFFICIENT o readout fallito valgono difficile
# come in open-guard (cascade.escalation_reason).
backend = TransformersBackend(model)
score = fast_scorer(tok, backend, seed=SEED)
score_difficulty = fast_scorer(tok, backend, seed=SEED + 1)
loader = partial(load_rows, cache_dir=Path(CACHE_DIR))
header = {"proof_type": "local", "model": f"{RUN['base_model']}+{CKPT}/adapter",
          "weights": [*BASE_WEIGHTS, adapter_identity], "scudo_sha": RUN_LOG["scudo_sha"],
          "runtime": {"name": "unsloth", "version": unsloth.__version__},
          "harness": "scudo.fast_notebook"}
REPORT_DIR = Path("/content/reports")
REPORT_DIR.mkdir(exist_ok=True)
EVAL_SUMMARY, REFUSED, readouts = {}, {}, {}
ALL_RESULTS, ALL_DIFFICULTY = [], {}
for entry in benchmarks():
    try:
        full = load_cases(entry, loader)
    except BenchmarkError as error:  # rifiuto prima di qualsiasi download
        REFUSED[entry.name] = str(error)
        continue
    loaded = sample(full, RUN["eval_per_benchmark"], seed=SEED)
    results = run(loaded, score)
    hard = run(replace(loaded, cases=tuple(as_difficulty(c) for c in loaded.cases)),
               score_difficulty)
    difficulty = {item.case_id: item.readout.verdict if item.readout else None for item in hard}
    ALL_RESULTS += results
    ALL_DIFFICULTY.update(difficulty)
    for split, report in reports_for(loaded, results, dedup=DEDUP, **header).items():
        subset = [r for r in results
                  if split == loaded.name or r.case_id not in loaded.machine_labelled]
        report["benchmark"]["sample"] = {"cases": len(loaded.cases), "of": len(full.cases),
                                         "seed": SEED}
        report["difficulty_routing"] = routing_metrics(subset, difficulty)
        (REPORT_DIR / (split.replace("/", "__") + ".json")).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        routing = report["difficulty_routing"]["overall"]
        EVAL_SUMMARY[split] = {"cases": report["overall"]["cases"],
                               "errors": report["overall"]["errors"],
                               "accuracy": report["overall"].get("accuracy"),
                               "hard_recall_on_wrong": routing["hard_recall_on_wrong"],
                               "hard_rate_on_correct": routing["hard_rate_on_correct"]}
    readouts.update({r.case_id: list(r.readout.logits) for r in results if r.readout})
    print(entry.name, json.dumps(EVAL_SUMMARY[entry.name], ensure_ascii=False))
ROUTING = routing_metrics(ALL_RESULTS, ALL_DIFFICULTY)
(REPORT_DIR / "difficulty_routing.json").write_text(
    json.dumps(ROUTING, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
(REPORT_DIR / "readouts_local.json").write_text(
    json.dumps(readouts, ensure_ascii=False) + "\\n", encoding="utf-8")
api.upload_folder(folder_path=str(REPORT_DIR), repo_id=CKPT, path_in_repo="reports",
                  commit_message="report eval locale del Fast v0.1")
RUN_LOG["eval"] = EVAL_SUMMARY
RUN_LOG["refused_benchmarks"] = REFUSED
RUN_LOG["difficulty_routing"] = ROUTING["overall"]
print("instradamento", json.dumps(ROUTING["overall"], ensure_ascii=False))"""

SUMMARY = """\
# Da incollare nell'issue #8: SHA di scudo, mix (revisione, digest, override), base e
# pesi, ripresa, step, adapter e GGUF, metriche per benchmark e instradamento.
(REPORT_DIR / "run_log.json").write_text(
    json.dumps(RUN_LOG, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
api.upload_file(path_or_fileobj=str(REPORT_DIR / "run_log.json"), path_in_repo="run_log.json",
                repo_id=CKPT, commit_message="run log del Fast v0.1")
print(json.dumps(RUN_LOG, indent=2, ensure_ascii=False))"""

OUTRO = """\
## Dopo il notebook

Questa e' la prova locale. Il passo live non si esegue su Colab: carica il GGUF
verificato nel backend nativo di open-guard e rileggi gli stessi casi
(`readouts_local.json`; i prompt si rigenerano con `scudo.runner.fast_scorer`, stesso
seme e stesso ordine dei casi), poi `scudo.eval_report.compare_readouts`. Il criterio
di difficolta' ha gli stessi byte di `routing.difficulty/1`; i criteri semantici del
mix non sono ancora nell'elenco server-owned di open-guard (`docs/protocol.md`).
`hard_recall_on_wrong` e' la quota degli errori del Fast che il criterio manda
all'Accurate: e' la misura che il secondo giro (#12) deve migliorare."""

CELLS = (
    ("markdown", "intro", INTRO),
    ("code", "setup", SETUP_HUB_CACHE),
    ("code", "config", CONFIG.replace("@BASE_REVISION@", FAST_BASE_REVISION)),
    ("code", "data", DATA),
    ("code", "model", MODEL),
    ("code", "lora", LORA),
    ("code", "train", TRAIN),
    ("code", "eval-local", EVAL),
    ("code", "gguf", named(GGUF, RUN_NAME)),
    ("code", "summary", SUMMARY),
    ("markdown", "outro", OUTRO),
)


def build() -> dict:
    return notebook(CELLS)


def main() -> None:
    print(write(NOTEBOOK_PATH, build()))


if __name__ == "__main__":
    main()
