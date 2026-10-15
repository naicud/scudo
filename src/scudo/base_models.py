"""Basi candidate del Fast per il bake-off (`docs/research/base-models-2026-10-15.md`).

Un candidato fissa tutto ciò che cambia fra un run e l'altro: repository e
revisione della base, licenza, loader Unsloth, `text_only`, moduli LoRA e loro
numero atteso, BOS e checkpoint repo. Il resto della ricetta resta quello del
tracer (`scudo.notebook`). Solo standard library.

Fatti riletti il 15 ottobre 2026 (`VERIFIED_ON`) alla revisione fissata:
`https://huggingface.co/api/models/<repo>/revision/main` per `sha` e
`cardData.license`; indice o header safetensors per i nomi dei moduli;
`tokenizer.json` (`added_tokens`, `post_processor`) per il BOS. Il numero di
moduli LoRA attesi segue i moduli che transformers 5.17 istanzia, non solo le
chiavi del checkpoint (vedi Gemma 4).

BOS: il protocollo non cambia (`scudo.decision-token/1`, stessi byte del
prompt). Un candidato con `bos_token_id` riceve l'id del BOS del suo tokenizer
prima degli id del prompt, nel training (testo del BOS in testa al prompt del
dataset, verificato sugli id del trainer) e nel serving (`runner.fast_scorer`
con `bos_id`). Il backend nativo oggi non mette BOS: per il passo live di un
candidato con BOS serve la stessa regola nel bridge (cancello del bake-off).
"""

from dataclasses import dataclass

VERIFIED_ON = "2026-10-15"
CONTROL = "qwen35-4b"

_ATTENTION_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Candidate:
    slug: str
    repo: str
    revision: str  # commit completo, riletto il VERIFIED_ON
    licence: str  # `cardData.license` alla revisione
    loader: str  # classe Unsloth: FastLanguageModel o FastModel
    text_only: bool  # passato al loader; il notebook fallisce se restano torri caricate
    lora_target_modules: tuple[str, ...]
    lora_modules: int  # moduli con LoRA attesi nel modello caricato
    bos_token: str | None  # testo del BOS del tokenizer, None: nessun BOS
    bos_token_id: int | None
    # File alla radice che il loader non legge: fuori dal download verificato e dai digest.
    ignore_files: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def checkpoint_repo(self) -> str:
        return f"Naicud/scudo-bakeoff-{self.slug}"


CANDIDATES: dict[str, Candidate] = {
    item.slug: item for item in (
        # Controllo: stessa base, moduli e loader del tracer (#2), rieseguito con il suo repo.
        Candidate(
            slug="qwen35-4b",
            repo="Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            licence="apache-2.0",
            loader="FastLanguageModel",
            text_only=True,
            lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                                 "in_proj_qkv", "in_proj_z", "out_proj",
                                 "gate_proj", "up_proj", "down_proj"),
            # 24 linear_attention x 3 + 8 full_attention x 4 + 32 MLP x 3; il layer `mtp`
            # del checkpoint non si carica (`_keys_to_ignore_on_load_unexpected`).
            lora_modules=200,
            bos_token=None,  # tokenizer senza BOS (`bos_token` null, post_processor ByteLevel)
            bos_token_id=None,
        ),
        Candidate(
            slug="gemma4-e4b",
            repo="google/gemma-4-E4B-it",
            revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
            licence="apache-2.0",
            # Doc Unsloth (Gemma 4, train): FastModel con finetune_vision_layers=False.
            loader="FastModel",
            # Unsloth 2026.10.5 (loader.py) salta torri visiva e audio quando il config di
            # testo e' della famiglia (gemma4 -> gemma4_text); la doc non lo dice.
            text_only=True,
            lora_target_modules=_ATTENTION_MLP,
            # 42 layer: q/o_proj e MLP ovunque; k/v_proj solo nei 24 layer senza KV
            # condivisa (num_kv_shared_layers=18: transformers 5.17 non li istanzia,
            # anche se il checkpoint ha le chiavi). per_layer_* esclusi.
            lora_modules=42 * 2 + 24 * 2 + 42 * 3,
            # Il post_processor di tokenizer.json non aggiunge BOS, il chat template si'
            # (`{{ bos_token }}`): i Gemma sono addestrati con BOS in testa.
            bos_token="<bos>",
            bos_token_id=2,
            notes=(
                "text_only su Gemma 4: NON VERIFICATO nella doc Unsloth (solo sorgente "
                "2026.10.5); il notebook fallisce se restano torri caricate",
                "effetto del BOS sulla qualita': NON VERIFICATO, si misura nel bake-off",
                "cancello: conferma del proprietario sull'ADR 0001 (Gemma) prima dell'uso",
                "cancello live: il bridge nativo deve mettere lo stesso BOS",
            ),
        ),
        Candidate(
            slug="ministral3-3b",
            repo="mistralai/Ministral-3-3B-Base-2512",
            revision="6f9c4b12a95b139af68670a6713616b757923735",
            licence="apache-2.0",
            # Doc Unsloth (Ministral 3): notebook con il loader vision (FastModel).
            loader="FastModel",
            # Il config di testo e' `ministral3`, non della famiglia `mistral3`: Unsloth
            # 2026.10.5 terrebbe comunque il modello intero. Lo si carica intero in modo
            # dichiarato e la LoRA resta sui layer di linguaggio (finetune_vision_layers=False).
            text_only=False,
            lora_target_modules=_ATTENTION_MLP,
            lora_modules=26 * 7,  # 26 layer language_model; la torre Pixtral (24 layer) esclusa
            # post_processor di tokenizer.json: `<s>` (id 1) in testa a ogni sequenza.
            bos_token="<s>",
            bos_token_id=1,
            # Pesi in formato Mistral, duplicato degli shard `model-0000N`: transformers
            # legge l'indice `model.safetensors.index.json`.
            ignore_files=("consolidated.safetensors",),
            notes=(
                "caricamento del modello intero (torre Pixtral inclusa) con FastModel e LoRA "
                "solo sul linguaggio: NON VERIFICATO su A100; il notebook fallisce se i "
                "moduli LoRA non sono quelli attesi",
                "export GGUF di Mistral3ForConditionalGeneration da Unsloth: NON VERIFICATO",
                "cancello live: il bridge nativo deve mettere lo stesso BOS",
            ),
        ),
    )
}


def candidate(slug: str) -> Candidate:
    if slug not in CANDIDATES:
        raise KeyError(f"base {slug!r} sconosciuta; candidati: {', '.join(CANDIDATES)}")
    return CANDIDATES[slug]
