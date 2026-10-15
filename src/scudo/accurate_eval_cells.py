"""Celle di eval e riepilogo del notebook dell'Accurate v0.1 (issue #9).

Seguono le celle di `scudo.accurate_cells` e ne usano le variabili (RUN, tok,
BASE_WEIGHTS, DEDUP, ...). L'eval riusa benchmark, runner e report dell'harness
delle baseline; il solo passo nuovo è `scudo.reasoned_runner`.
"""

EVAL = """\
from functools import partial

import unsloth
from huggingface_hub import HfApi

from scudo.artifacts import ArtifactIdentity
from scudo.benchmarks import BenchmarkError, load_cases, sample
from scudo.ingest import load_rows
from scudo.manifest import benchmarks
from scudo.reasoned_runner import accurate_reports, accurate_scorer
from scudo.runner import TransformersBackend, run

FastLanguageModel.for_inference(model)
api = HfApi()

# Adapter: salvato, hashato e caricato; l'identita' usa lo SHA del commit di upload.
ADAPTER_DIR = Path("/content/adapter")
model.save_pretrained(str(ADAPTER_DIR))
adapter_sha = file_sha256(ADAPTER_DIR / "adapter_model.safetensors")
commit = api.upload_folder(
    folder_path=str(ADAPTER_DIR), repo_id=CKPT, path_in_repo="adapter",
    commit_message="adapter dell'Accurate",
)
adapter_identity = ArtifactIdentity(CKPT, commit.oid, "adapter_model.safetensors", adapter_sha)
RUN_LOG["adapter"] = adapter_identity.manifest_entry()

# Sequenza del frame nativo: prompt + Reasoning:, generazione greedy entro il budget,
# token fino allo stop, Answer: e logit di A/B/C. Stop mancante = errore del caso.
# NON VERIFICATO: TransformersBackend.generate (output_logits) sul modello patchato
# da Unsloth; se non regge, la cella fallisce qui, senza fallback.
score, usage = accurate_scorer(tok, TransformersBackend(model), seed=SEED,
                               budget=RUN["max_new_tokens"])
loader = partial(load_rows, cache_dir=Path(CACHE_DIR))
REPORT_DIR = Path("/content/reports")
REPORT_DIR.mkdir(exist_ok=True)
EVAL_SUMMARY, REFUSED, readouts = {}, {}, {}
for entry in benchmarks():
    try:
        full = load_cases(entry, loader)
    except BenchmarkError as error:  # rifiuto prima di qualsiasi download
        REFUSED[entry.name] = str(error)
        continue
    loaded = sample(full, RUN["eval_per_benchmark"], seed=SEED)
    results = run(loaded, score)
    reports = accurate_reports(
        loaded, results, usage, budget=RUN["max_new_tokens"], dedup=DEDUP,
        proof_type="local", model=f"{RUN['base_model']}+{CKPT}/adapter",
        weights=[*BASE_WEIGHTS, adapter_identity], scudo_sha=RUN_LOG["scudo_sha"],
        runtime={"name": "unsloth", "version": unsloth.__version__},
        harness="scudo.accurate_notebook", format=REASONED_FORMAT_VERSION,
    )
    for split, report in reports.items():
        report["benchmark"]["sample"] = {"cases": len(loaded.cases), "of": len(full.cases),
                                         "seed": SEED}
        (REPORT_DIR / (split.replace("/", "__") + ".json")).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
        overall = report["overall"]
        EVAL_SUMMARY[split] = {"cases": overall["cases"], "errors": overall["errors"],
                               "accuracy": overall.get("accuracy"),
                               "token_cost": report.get("token_cost")}
    readouts.update({case_id: usage[case] for case_id, case in zip(loaded.case_ids, loaded.cases)
                     if "logits" in usage.get(case, {})})
    print(entry.name, json.dumps(EVAL_SUMMARY[entry.name], ensure_ascii=False))
(REPORT_DIR / "readouts_local.json").write_text(
    json.dumps(readouts, ensure_ascii=False) + "\\n", encoding="utf-8")
api.upload_folder(folder_path=str(REPORT_DIR), repo_id=CKPT, path_in_repo="reports",
                  commit_message="report eval locale dell'Accurate")
RUN_LOG["eval"] = EVAL_SUMMARY
RUN_LOG["refused_benchmarks"] = REFUSED"""

GGUF_NOTE = """\
## Export GGUF

Stessa cella del tracer. NON VERIFICATO con `load_in_4bit=True`: quali pesi della
base entrano nel merge (shard 16-bit verificati o pesi 4-bit dequantizzati) non e'
verificato; il confronto con il backend nativo (`compare_readouts`) lo misura."""

SUMMARY = """\
# Da incollare nell'issue #9: SHA di scudo, base e licenza, dati e deduplica,
# quantizzazione, ripresa, step, adapter e GGUF, accuratezze e costo in token.
print(json.dumps(RUN_LOG, indent=2, ensure_ascii=False))"""

OUTRO = """\
## Dopo il notebook

Questa e' la prova locale. Il passo live non si esegue su Colab: carica il GGUF
verificato nel backend nativo di open-guard con il frame `reasoning`
`{"max_new_tokens": 512, "reasoning_prefix": "Reasoning:\\n",
"stop_sequence": "\\n</reasoning>\\n", "answer_prefix": "Answer:\\n"}`, rileggi
gli stessi prompt di `readouts_local.json` e confronta con
`scudo.eval_report.compare_readouts`. Cosa deve ancora cambiare in open-guard
(stop mancante come errore, criterio libero nel renderer) e' in
`docs/protocol.md`. Fino ad allora l'accuratezza riportata vale solo per il
checkpoint locale, con revisione e digest dichiarati."""
