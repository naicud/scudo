"""Genera il notebook Colab del tracer veloce (issue #2).

Il notebook è un artefatto generato: le celle proprie vivono qui come stringhe
Python, setup, LoRA, addestramento, adapter e GGUF in `scudo.notebook_cells`, e
`python -m scudo.notebook` riscrive `notebooks/fast_tracer.ipynb`; con
`--base <slug>` scrive il notebook di bake-off del candidato
(`scudo.bakeoff_cells`). Solo standard library; le dipendenze pesanti vengono
importate dentro le celle, su Colab.
"""

from scudo import nemotron
from scudo.notebook_cells import ADAPTER, GGUF, LORA, SETUP, TRAIN, named, notebook, write

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

from scudo.cases import training_rows
from scudo.nemotron import DATASET, LANGUAGES, REVISION, heldout_cases, sample_cases


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

EVAL = """\
import torch
import unsloth
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from pathlib import Path

from scudo.artifacts import ArtifactIdentity, file_sha256
from scudo.eval_report import EvalResult, build_report
from scudo.protocol import read_options

""" + named(ADAPTER, "del tracer") + """

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
        results.append(EvalResult(case_id, row["language"], row["surface"], row["verdict"],
                                  None, error=type(exc).__name__))
        continue
    results.append(EvalResult(case_id, row["language"], row["surface"], row["verdict"], readout))
    readouts[case_id] = {
        "prompt": row["prompt"],
        "semantic_ids": row["semantic_ids"],
        "logits": list(readout.logits),
    }

# Pesi che determinano i readout: shard della base alla revisione fissata e adapter.
base_weights = [
    ArtifactIdentity(RUN["base_model"], RUN["base_revision"], name, file_sha256(Path(
        hf_hub_download(RUN["base_model"], name, revision=RUN["base_revision"])
    )))
    for name in sorted(list_repo_files(RUN["base_model"], revision=RUN["base_revision"]))
    if name.endswith(".safetensors") and "/" not in name
]
report = build_report(
    results, proof_type="local", model=f"{RUN['base_model']}+{CKPT}/adapter",
    weights=[*base_weights, adapter_identity], scudo_sha=RUN_LOG["scudo_sha"],
    runtime={"name": "unsloth", "version": unsloth.__version__},
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
    ("code", "gguf", named(GGUF, "del tracer")),
    ("code", "summary", SUMMARY),
    ("markdown", "outro", OUTRO),
)


def bakeoff_path(slug: str) -> str:
    return f"notebooks/bakeoff_{slug}.ipynb"


def build(base: str | None = None) -> dict:
    """Il tracer; con `base` (slug di `scudo.base_models`) il notebook di bake-off."""
    if base is None:
        return notebook(CELLS)
    # Import locale: le celle del bake-off derivano da quelle di questo modulo.
    from scudo.bakeoff_cells import cells
    from scudo.base_models import candidate

    return notebook(cells(candidate(base)))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m scudo.notebook")
    parser.add_argument("--base", help="slug del candidato: scrive notebooks/bakeoff_<slug>.ipynb")
    base = parser.parse_args(argv).base
    print(write(NOTEBOOK_PATH if base is None else bakeoff_path(base), build(base)))


if __name__ == "__main__":
    main()
