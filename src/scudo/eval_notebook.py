"""Genera il notebook Colab dell'eval (issue #4): il nostro Fast e le baseline sui benchmark.

Come `scudo.notebook` per il tracer: le celle vivono qui come stringhe (il setup
in `scudo.notebook_cells`) e `python -m scudo.eval_notebook` riscrive `notebooks/eval.ipynb`. Solo standard
library; le dipendenze pesanti si importano dentro le celle, su Colab.
"""

import json

from scudo.base_models import CANDIDATES
from scudo.notebook_cells import SETUP_HUB_CACHE, notebook, write

NOTEBOOK_PATH = "notebooks/eval.ipynb"
FAST_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # Qwen/Qwen3.5-4B, come il tracer
TRACER_ADAPTER_REVISION = "e7124a687ce3721a29bfb6b68a461357c0400659"  # upload dell'adapter, #2
# Upload degli adapter dei run di bake-off su Colab A100 (RUN_LOG["adapter"] di ogni notebook).
BAKEOFF_ADAPTER_REVISIONS = {
    "qwen35-4b": "f4f5a84c06bcb7d121dff9e0823d0478fb6219da",
    "gemma4-e4b": "fa039d8cedcb3e0784adf64fa5a422a975b96428",
    "ministral3-3b": "d2c8d3bbadd87f50cfe9bd9b469481baf23a861e",
}


def fast_entries() -> list[dict]:
    """Voci di EVAL["fast"]: il tracer, poi un candidato del bake-off per base."""
    tracer = {"name": "scudo-fast", "base_model": "Qwen/Qwen3.5-4B",
              "base_revision": FAST_BASE_REVISION, "adapter_repo": "Naicud/scudo-checkpoints",
              "adapter_revision": TRACER_ADAPTER_REVISION, "loader": "FastLanguageModel",
              "text_only": True, "bos_token": None, "bos_token_id": None, "ignore_files": []}
    bakeoff = [{"name": f"bakeoff-{item.slug}", "base_model": item.repo,
                "base_revision": item.revision, "adapter_repo": item.checkpoint_repo,
                "adapter_revision": BAKEOFF_ADAPTER_REVISIONS.get(item.slug, ""),
                "loader": item.loader, "text_only": item.text_only,
                "bos_token": item.bos_token, "bos_token_id": item.bos_token_id,
                "ignore_files": list(item.ignore_files)} for item in CANDIDATES.values()]
    return [{**entry, "adapter_dir": "adapter", "max_seq_length": 2048}
            for entry in (tracer, *bakeoff)]


def _render(entries: list[dict]) -> str:
    def literal(value):
        return repr(value) if value is None or isinstance(value, bool) else json.dumps(
            value, ensure_ascii=False)

    return "\n".join("        {\n" + "".join(f'            "{key}": {literal(value)},\n'
                                             for key, value in entry.items()) + "        },"
                     for entry in entries)

INTRO = """\
# Scudo - eval di Fast e baseline (issue #4)

Un run per modello sui benchmark del registro (`scudo.manifest.benchmarks()`), con
un report per benchmark: accuratezza, F1 per verdetto, calibrazione e over-refusal
per lingua e superficie (`docs/eval.md`). Ogni report cita SHA di scudo, identita'
e digest dei pesi, runtime, tipo di prova, conteggi del benchmark e verifica della
deduplica dell'ingestione.

- Baseline: prova `reference` (pesi pubblicati nel runtime transformers).
- I nostri Fast (il tracer e i candidati del bake-off in `EVAL["fast"]`): prova `local`
  (checkpoint in training, loader Unsloth), stessi casi e stesso seme per tutti.
- Rifiuti espliciti, mai silenziosi: WildGuardTest (gated, campi NON VERIFICATI),
  Rizzo Flow (codice remoto incompatibile con transformers 5.17) e APUS-Reflex
  (runtime della card con transformers==5.16.1) finiscono in `skipped_models` o
  `refused_benchmarks` con il motivo.

Prerequisiti: runtime Colab A100; Colab Secrets `HF_TOKEN` (lettura e scrittura su
`Naicud/scudo-checkpoints`) e `GH_TOKEN` (lettura di `naicud/scudo`). Dopo una
disconnessione si rieseguono setup, config, dati e helper, poi le celle dei modelli
mancanti: ogni modello carica i suoi report appena finisce."""

CONFIG = """\
from pathlib import Path

EVAL = {
    "checkpoint_repo": "Naicud/scudo-checkpoints",  # privato: report in eval/<scudo_sha>/
    "report_dir": "/content/eval-reports",
    "seed": 3407,  # permutazione delle opzioni del Fast e campione dei benchmark
    # Casi per benchmark, campione deterministico con `seed` (benchmarks.sample): stessi
    # casi per ogni modello. Le baseline generative costano una generazione per caso.
    "per_benchmark": 500,
    # Report della build di ingestione (`python -m scudo.ingest build`), in un repository
    # privato a un commit fissato. Vuoto: verifica della deduplica NOT_RUN nei report.
    "ingest_report": {"repo": "", "repo_type": "dataset", "revision": "", "path": "report.json"},
    # I nostri Fast, sugli stessi casi e con lo stesso seme: il tracer (#2) e i candidati
    # del bake-off (scudo.base_models). adapter_revision e' il commit dell'upload
    # dell'adapter (RUN_LOG["adapter"] del notebook di training); vuoto: Fast saltato.
    "fast": [
@FAST@
    ],
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

from scudo.benchmarks import BenchmarkError, load_cases, sample
from scudo.ingest import load_rows
from scudo.manifest import benchmarks

loader = partial(load_rows, cache_dir=Path(CACHE_DIR))
LOADED, REFUSED = {}, {}
for entry in benchmarks():
    try:
        full = load_cases(entry, loader)
        LOADED[entry.name] = sample(full, EVAL["per_benchmark"], seed=EVAL["seed"])
    except BenchmarkError as error:  # rifiuto prima di qualsiasi download
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

from scudo.bakeoff import result_rows
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
    # Tri-state e vista binaria (VIOLATION contro il resto) affiancati: docs/eval.md.
    return {"cases": overall["cases"], "errors": overall["errors"],
            "unsupported": overall["unsupported"],
            "accuracy": overall["accuracy"], "f1": overall["f1"],
            "ece": calibration.get("ece"), "brier": calibration.get("brier"),
            "over_refusal_rate": over_refusal.get("rate"), "binary": overall["binary"]}


def evaluate(model, score, **header):
    \"\"\"Un run su ogni benchmark caricato; report e risultati per caso sul repository privato.

    `results.json` (benchmark -> righe per caso) e' l'ingresso di `python -m scudo.bakeoff`.
    \"\"\"
    out, rows = {}, {}
    for name, loaded in LOADED.items():
        results = run(loaded, score)
        rows[name] = result_rows(results)
        reports = reports_for(loaded, results, dedup=DEDUP, model=model,
                              scudo_sha=RUN_LOG["scudo_sha"], harness="scudo.eval_notebook",
                              **header)
        if not reports:  # nessun caso del benchmark e' coperto dal modello
            out[name] = {"cases": 0, "unsupported": len(results)}
            print(model, name, "NON SUPPORTATO:", len(results), "casi")
            continue
        for split, report in reports.items():
            path = REPORT_DIR / model / (split.replace("/", "__") + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\\n",
                            encoding="utf-8")
            out[split] = headline(report)
        out[name]["error_codes"] = dict(Counter(item.error for item in results if item.error))
        print(model, name, json.dumps(out[name], ensure_ascii=False))
    (REPORT_DIR / model / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False) + "\\n", encoding="utf-8")
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
# toccare le baseline. Un run per voce di EVAL["fast"], tutte sullo stesso percorso del
# runner; loader, text_only e BOS vengono dalla voce, come nel notebook di training.
# Se loader, adapter o BOS non combaciano la cella fallisce, senza fallback.
for FAST in EVAL["fast"]:
    if not FAST["adapter_revision"]:
        SUMMARY["skipped_models"][FAST["name"]] = "adapter non configurato (EVAL['fast'])"
        print("SALTATO", FAST["name"], "- imposta adapter_revision")
        continue
    import unsloth
    from peft import PeftModel

    Loader = getattr(unsloth, FAST["loader"])
    base_root, base_weights = fetch_verified(FAST["base_model"], FAST["base_revision"],
                                             ROOT_FILES, (*NO_SUBFOLDERS, *FAST["ignore_files"]))
    adapter_root, adapter_weights = fetch_verified(
        FAST["adapter_repo"], FAST["adapter_revision"], (f"{FAST['adapter_dir']}/*",))
    model, tokenizer = Loader.from_pretrained(
        model_name=str(base_root), max_seq_length=FAST["max_seq_length"],
        load_in_4bit=False, load_in_16bit=True, text_only=FAST["text_only"],
    )
    model = PeftModel.from_pretrained(model, str(adapter_root / FAST["adapter_dir"]))
    assert any("lora_A" in name for name, _ in model.named_modules()), "adapter non applicato"
    Loader.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if FAST["bos_token_id"] is not None and (tok.bos_token, tok.bos_token_id) != (
            FAST["bos_token"], FAST["bos_token_id"]):
        raise RuntimeError(f"{FAST['name']}: BOS del tokenizer diverso da quello dichiarato")
    score = fast_scorer(tok, TransformersBackend(model), seed=EVAL["seed"],
                        bos_id=FAST["bos_token_id"])
    evaluate(FAST["name"], score, proof_type="local", weights=[*base_weights, *adapter_weights],
             runtime={"name": "unsloth", "version": unsloth.__version__})
    del model, score
    gc.collect()
    torch.cuda.empty_cache()"""

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
    ("code", "setup", SETUP_HUB_CACHE),
    ("code", "config", CONFIG.replace("@FAST@", _render(fast_entries()))),
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
    return notebook(CELLS)


def main() -> None:
    print(write(NOTEBOOK_PATH, build()))


if __name__ == "__main__":
    main()
