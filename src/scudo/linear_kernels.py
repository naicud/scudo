"""Quali kernel veloci usa transformers per i layer linear-attention di Qwen3.5.

transformers 5.17 (`models/qwen3_5/modeling_qwen3_5.py`) importa ogni funzione da
un pacchetto opzionale e, se l'import fallisce, ripiega sull'implementazione
PyTorch di riferimento: corretta ma molto piu' lenta. Questo modulo ripete gli
stessi import e dice quale percorso e' attivo, senza mai far fallire il run.

Gira in un sottoprocesso (`python -m scudo.linear_kernels`): importare `fla` carica
transformers, che nel kernel del notebook deve arrivare dopo Unsloth.
"""

import importlib
import json

# (distribuzione pip, modulo, funzione), come in transformers 5.17.
KERNELS = (
    ("causal-conv1d", "causal_conv1d", "causal_conv1d_fn"),
    ("causal-conv1d", "causal_conv1d", "causal_conv1d_update"),
    ("flash-linear-attention", "fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    ("flash-linear-attention", "fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
)


def probe(kernels=KERNELS) -> dict[str, str]:
    """Per ogni funzione: "kernel <distribuzione>" o il motivo del ripiego su PyTorch."""
    status = {}
    for distribution, module, function in kernels:
        try:
            found = callable(getattr(importlib.import_module(module), function, None))
            status[function] = (f"kernel {distribution}" if found
                                else f"fallback PyTorch: {module}.{function} assente")
        except Exception as error:  # qualsiasi errore d'import vale come kernel assente
            status[function] = f"fallback PyTorch: {type(error).__name__}: {error}"
    return status


if __name__ == "__main__":
    print(json.dumps(probe(), ensure_ascii=False))
