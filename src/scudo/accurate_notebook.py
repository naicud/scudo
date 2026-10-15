"""Genera il notebook Colab dell'Accurate v0.1 (issue #9).

`python -m scudo.accurate_notebook` riscrive `notebooks/accurate_reasoned.ipynb`
dalle celle di `scudo.accurate_cells` e `scudo.accurate_eval_cells`, più setup,
addestramento ed export GGUF del tracer (`scudo.notebook`), riusati senza copia.
Solo standard library; le dipendenze pesanti si importano dentro le celle.
"""

import json
from pathlib import Path

from scudo import accurate_cells as cells
from scudo import accurate_eval_cells as eval_cells
from scudo import notebook as tracer
from scudo import reasoning_data

NOTEBOOK_PATH = "notebooks/accurate_reasoned.ipynb"
# Letti il 2026-10-15 da /api/models/Qwen/Qwen3.5-9B/revision/main: cardData.license
# apache-2.0, file LICENSE Apache 2.0, non gated; config e indice dei pesi a questa revisione.
BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BASE_LICENSE = "apache-2.0"


def _filled(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        if text.count(key) != 1:
            raise ValueError(f"{key}: atteso una volta nella cella")
        text = text.replace(key, value)
    return text


CELLS = (
    ("markdown", "intro", cells.INTRO),
    ("code", "setup", tracer.SETUP),
    ("code", "config", _filled(cells.CONFIG, {
        "@BASE_MODEL@": BASE_MODEL, "@BASE_REVISION@": BASE_REVISION,
        "@BASE_LICENSE@": BASE_LICENSE, "@NCSR_REVISION@": reasoning_data.REVISION})),
    ("code", "data", cells.DATA),
    ("code", "model", cells.MODEL),
    ("code", "lora", cells.LORA),
    ("code", "train", tracer.TRAIN),
    ("code", "eval-local", eval_cells.EVAL),
    ("markdown", "gguf-note", eval_cells.GGUF_NOTE),
    ("code", "gguf", _filled(tracer.GGUF, {"del tracer": "dell'Accurate"})),
    ("code", "summary", eval_cells.SUMMARY),
    ("markdown", "outro", eval_cells.OUTRO),
)


def build() -> dict:
    return {
        "cells": [tracer._cell(*cell) for cell in CELLS],
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
