"""Celle del notebook di bake-off: le celle del tracer con la base di un candidato.

Ogni modifica sostituisce un frammento che deve comparire una volta sola nella
cella del tracer (`swap`): se il tracer cambia, il generatore si ferma invece di
produrre in silenzio un notebook diverso. Dati, seme, step, LoRA r/α e lr restano
quelli del tracer; cambiano solo i campi di `scudo.base_models`, che entrano in
`RUN` e quindi nell'impronta del run. Solo standard library.
"""

import json

from scudo import nemotron
from scudo import notebook as tracer
from scudo.base_models import CANDIDATES, CONTROL, VERIFIED_ON, Candidate
from scudo.notebook_cells import GGUF, SETUP, TRAIN, named


def swap(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"frammento atteso una volta nella cella: {old[:60]!r}")
    return text.replace(old, new)


def _literal(value) -> str:
    return repr(value) if value is None or isinstance(value, bool) else json.dumps(value)


def config(item: Candidate) -> str:
    fields = {
        "lora_target_modules": list(item.lora_target_modules),
        "lora_modules": item.lora_modules,
        "loader": item.loader,
        "text_only": item.text_only,
        "bos_token": item.bos_token,
        "bos_token_id": item.bos_token_id,
        "ignore_files": list(item.ignore_files),
    }
    extra = "".join(f'    "{key}": {_literal(value)},\n' for key, value in fields.items())
    control = CANDIDATES[CONTROL]  # la base scritta nel tracer
    text = tracer.CONFIG.replace("@REVISION@", nemotron.REVISION)
    text = swap(text, f'    "base_model": "{control.repo}",\n'
                      f'    "base_revision": "{control.revision}",\n',
                f'    "base_model": "{item.repo}",\n    "base_revision": "{item.revision}",\n')
    text = swap(text, '    "checkpoint_repo": "Naicud/scudo-checkpoints",\n',
                f'    "checkpoint_repo": "{item.checkpoint_repo}",\n')
    text = swap(text, '    "lora_alpha": 16,\n}',
                f'    "lora_alpha": 16,\n    # Base del bake-off (scudo.base_models, '
                f'riletta il {VERIFIED_ON}).\n{extra}}}')
    return swap(text, 'RUN_LOG["base"] = {"model": RUN["base_model"], "revision": RUN["base_revision"]}',
                'RUN_LOG["base"] = {"model": RUN["base_model"], "revision": RUN["base_revision"],\n'
                '                   "loader": RUN["loader"], "text_only": RUN["text_only"],\n'
                '                   "bos_token_id": RUN["bos_token_id"]}')


def data() -> str:
    text = swap(tracer.DATA, "train_rows = training_rows(train_cases, seed=SEED)\n",
                "train_rows = training_rows(train_cases, seed=SEED)\n"
                "# BOS del candidato come testo in testa al prompt (protocollo invariato): gli id\n"
                "# che il trainer ne ricava si verificano nella cella train.\n"
                'BOS_TEXT = RUN["bos_token"] or ""\n')
    return swap(text, '[{"prompt": row["prompt"], "completion": row["completion"]}',
                '[{"prompt": BOS_TEXT + row["prompt"], "completion": row["completion"]}')


MODEL_GUARDS = """\
tok = getattr(tokenizer, "tokenizer", tokenizer)
RUN_LOG["unsloth"] = unsloth.__version__
RUN_LOG["model_class"] = type(model).__name__

# text_only: nessuna torre visiva o audio deve restare caricata, o il loader l'ha ignorato.
TOWERS = ("visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio",
          "multi_modal_projector")
towers = sorted({part for name, _ in model.named_modules() for part in TOWERS if part in name})
RUN_LOG["towers_loaded"] = towers
if RUN["text_only"] and towers:
    raise RuntimeError(f"text_only ignorato dal loader: torri caricate {towers}")

# BOS dichiarato per candidato: lo stesso id del tokenizer in training e serving.
if RUN["bos_token_id"] is not None and (tok.bos_token, tok.bos_token_id) != (
        RUN["bos_token"], RUN["bos_token_id"]):
    raise RuntimeError(f"BOS del tokenizer {tok.bos_token!r}/{tok.bos_token_id}, "
                       f"atteso {RUN['bos_token']!r}/{RUN['bos_token_id']}")
BOS_IDS = [] if RUN["bos_token_id"] is None else [RUN["bos_token_id"]]
"""


def model() -> str:
    text = swap(tracer.MODEL,
                "from unsloth import FastLanguageModel\n\n"
                "# `text_only` e' nel loader Unsloth ma non e' confermato nella documentazione\n"
                "# Qwen3.5: se il loader non lo accetta, la cella fallisce qui e va rivista.\n"
                "model, tokenizer = FastLanguageModel.from_pretrained(",
                "import unsloth\n\n"
                "# Loader e text_only del candidato: un argomento rifiutato ferma la cella.\n"
                'Loader = getattr(unsloth, RUN["loader"])\n'
                "model, tokenizer = Loader.from_pretrained(")
    text = swap(text, "    text_only=True,\n", '    text_only=RUN["text_only"],\n')
    text = swap(text, 'tok = getattr(tokenizer, "tokenizer", tokenizer)\n', MODEL_GUARDS)
    return swap(text,
                "base_ids = tok.encode(sample_prompt, add_special_tokens=False)\n"
                "assert tok.encode(",
                "base_ids = BOS_IDS + tok.encode(sample_prompt, add_special_tokens=False)\n"
                "assert BOS_IDS + tok.encode(")


LORA = """\
# Moduli LoRA del candidato, letti dai pesi alla revisione fissata (scudo.base_models).
# Con FastModel la LoRA resta sui layer di linguaggio anche se una torre e' caricata.
scope = {
    "finetune_vision_layers": False, "finetune_language_layers": True,
    "finetune_attention_modules": True, "finetune_mlp_modules": True,
} if RUN["loader"] == "FastModel" else {}
model = Loader.get_peft_model(
    model,
    r=RUN["lora_r"],
    target_modules=RUN["lora_target_modules"],
    lora_alpha=RUN["lora_alpha"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    **scope,
)
model.print_trainable_parameters()

# Moduli avvolti davvero: tutti nel linguaggio e quanti attesi, altrimenti stop.
lora_names = [name.removesuffix(".lora_A") for name, _ in model.named_modules()
              if name.endswith(".lora_A")]
outside = [name for name in lora_names if any(part in name for part in TOWERS)]
RUN_LOG["lora_modules"] = len(lora_names)
if outside or len(lora_names) != RUN["lora_modules"]:
    raise RuntimeError(f"LoRA su {len(lora_names)} moduli, attesi {RUN['lora_modules']}; "
                       f"fuori dal linguaggio: {outside[:5]}")"""

TRAIN_GUARD = """\
trainer = SFTTrainer(
    model=model, processing_class=tok, train_dataset=dataset, args=config
)
# Il trainer deve vedere gli id del serving: BOS dichiarato, prompt senza altri token
# speciali, poi la lettera. Unsloth sceglie da se' i token speciali: qui si verifica.
first_ids = list(trainer.train_dataset[0]["input_ids"])
expected = BOS_IDS + tok.encode(train_rows[0]["prompt"], add_special_tokens=False)
if first_ids[:len(expected) + 1] != expected + [LETTER_IDS[train_rows[0]["completion"]]]:
    raise RuntimeError("id di training diversi da quelli del serving: controlla BOS e tokenizer")
"""


def train() -> str:
    return swap(TRAIN, "trainer = SFTTrainer(\n    model=model, processing_class=tok, "
                       "train_dataset=dataset, args=config\n)\n", TRAIN_GUARD)


def evaluate(item: Candidate) -> str:
    text = swap(tracer.EVAL, 'commit_message="adapter del tracer"',
                f'commit_message="adapter bake-off {item.slug}"')
    text = swap(text, "FastLanguageModel.for_inference(model)", "Loader.for_inference(model)")
    text = swap(text, 'if name.endswith(".safetensors") and "/" not in name',
                'if name.endswith(".safetensors") and "/" not in name\n'
                '    and name not in RUN["ignore_files"]')
    text = swap(text, 'ids = tok.encode(row["prompt"], add_special_tokens=False)',
                'ids = BOS_IDS + tok.encode(row["prompt"], add_special_tokens=False)')
    return swap(text, 'harness="scudo.notebook/fast_tracer"',
                f'harness="scudo.notebook/bakeoff_{item.slug}"')


def intro(item: Candidate) -> str:
    notes = "".join(f"\n- {note}" for note in item.notes) or "\n- nessuna"
    bos = (f"`{item.bos_token}` (id {item.bos_token_id}) prima degli id del prompt, in "
           "training e in eval" if item.bos_token else "nessuno, come il tracer")
    return f"""\
# Scudo - bake-off del Fast: `{item.repo}`

Stessa ricetta del tracer (#2: 2.000 casi Nemotron v3 IT/EN, seme 3407, LoRA r16/α16,
125 step) sulla base `{item.repo}` alla revisione `{item.revision}` (licenza
`{item.licence}`, riletta il {VERIFIED_ON}). Protocollo `decision-token` invariato.

- loader `{item.loader}`, `text_only={item.text_only}`, {item.lora_modules} moduli LoRA attesi;
- BOS: {bos};
- checkpoint privato: `{item.checkpoint_repo}`.

Note e punti NON VERIFICATI (la cella relativa fallisce in modo esplicito):{notes}

Prerequisiti come il tracer: Colab A100, Secrets `HF_TOKEN` (scrittura su
`{item.checkpoint_repo}`) e `GH_TOKEN`. Dopo una disconnessione si rieseguono tutte
le celle: il training riprende da `last-checkpoint` se l'impronta del run coincide.
Prova locale, non live."""


SUMMARY = """\
# Da riportare nel bake-off: SHA di scudo, base, loader, BOS, moduli LoRA, ripresa,
# step, adapter (commit dell'upload: va in EVAL["fast"] dell'eval), GGUF, accuratezze.
print(json.dumps(RUN_LOG, indent=2, ensure_ascii=False))"""

OUTRO = """\
## Dopo il notebook

Copia il commit dell'adapter (`RUN_LOG["adapter"]`) nella voce del candidato di
`EVAL["fast"]` in `notebooks/eval.ipynb`, poi confronta i candidati con
`python -m scudo.bakeoff` (`docs/eval.md`, *Bake-off*). Il vincitore passa dalla
prova live nel backend nativo prima di sostituire il controllo."""


def cells(item: Candidate) -> tuple:
    run_name = f"bake-off {item.slug}"
    return (
        ("markdown", "intro", intro(item)),
        ("code", "setup", SETUP),
        ("code", "config", config(item)),
        ("code", "data", data()),
        ("code", "model", model()),
        ("code", "lora", LORA),
        ("code", "train", train()),
        ("code", "eval-local", evaluate(item)),
        ("code", "gguf", named(GGUF, run_name)),
        ("code", "summary", SUMMARY),
        ("markdown", "outro", OUTRO),
    )
