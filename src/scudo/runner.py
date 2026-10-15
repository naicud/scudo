"""Runner dell'eval: casi -> readout -> `EvalResult`, per il nostro Fast e per le baseline.

L'orchestrazione è Python puro e si testa con modello e tokenizer finti. torch,
transformers, peft e huggingface_hub si importano solo dentro le funzioni che
caricano i pesi, su Colab. Ogni file del modello si verifica alla revisione
fissata prima di passare la cartella a transformers: i file LFS con lo SHA-256
del puntatore (`scudo.artifacts.load_verified`), gli altri con l'id del blob git.
Un'anomalia per caso diventa un errore del risultato, mai `NO_VIOLATION`; un
rifiuto del modello (token delle etichette non distinti, runtime non supportato)
ferma quel modello senza fallback. Scelte e limiti sono in `docs/eval.md`.
"""

import fnmatch
import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from scudo.artifacts import ArtifactError, ArtifactIdentity, load_verified
from scudo.baselines import BASELINES, BaselineError, LabelReadout, build_input, parse
from scudo.benchmarks import Loaded
from scudo.cases import Case
from scudo.eval_report import EvalResult, build_report
from scudo.protocol import OPTION_LETTERS, Readout, read_options, training_row

# File alla radice di un repository di pesi: config, tokenizer, template, pesi, codice.
ROOT_FILES = ("*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py")
NO_SUBFOLDERS = ("*/*",)


class RunnerError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...
    def apply_chat_template(self, messages: list[dict], **kwargs) -> str: ...


class Backend(Protocol):
    """Logit del modello: `next_logits` all'ultima posizione, `generate` greedy per passo."""

    def next_logits(self, ids: list[int]) -> Sequence[float]: ...
    def generate(self, ids: list[int], max_new_tokens: int
                 ) -> tuple[list[int], list[Sequence[float]]]: ...


# --- Verifica dei file -------------------------------------------------------


@dataclass(frozen=True)
class RemoteFile:
    path: str  # percorso nel repository
    blob_id: str  # id del blob git; per un file LFS è quello del puntatore
    sha256: str | None  # SHA-256 del contenuto, solo per i file LFS


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def selected(files: Sequence[RemoteFile], allow: Sequence[str],
             ignore: Sequence[str] = ()) -> list[RemoteFile]:
    """File scelti dai pattern (fnmatch, `*` attraversa anche `/`)."""
    return [f for f in files
            if any(fnmatch.fnmatch(f.path, p) for p in allow)
            and not any(fnmatch.fnmatch(f.path, p) for p in ignore)]


def verify_snapshot(root: Path, files: Sequence[RemoteFile], *, repo: str,
                    revision: str) -> list[ArtifactIdentity]:
    """Verifica ogni file scelto prima di qualsiasi loader; restituisce le identità dei pesi.

    Un file mancante o con digest diverso blocca (`ArtifactError`), senza fallback.
    """
    root = Path(root)
    weights = []
    for remote in files:
        path = root / remote.path
        if remote.sha256 is not None:
            identity = ArtifactIdentity(repo, revision, Path(remote.path).name, remote.sha256)
            load_verified(path, identity, lambda p: p)
            if remote.path.endswith(".safetensors"):
                weights.append(identity)
        elif not path.is_file():
            raise ArtifactError("ARTIFACT_MISSING", str(path))
        elif git_blob_sha1(path) != remote.blob_id:
            raise ArtifactError("ARTIFACT_DIGEST_MISMATCH", f"{remote.path}: blob git diverso")
    if not weights:
        raise RunnerError("NO_WEIGHTS", f"{repo}@{revision}: nessun file .safetensors scelto")
    return weights


def fetch_verified(repo: str, revision: str, allow: Sequence[str],
                   ignore: Sequence[str] = ()) -> tuple[Path, list[ArtifactIdentity]]:
    """Scarica i file scelti alla revisione fissata e li verifica contro i metadati del Hub."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RunnerError("REVISION_MISMATCH", f"{repo}: {info.sha} != {revision}")
    remote = [RemoteFile(s.rfilename, s.blob_id, s.lfs.sha256 if s.lfs else None)
              for s in info.siblings]
    chosen = selected(remote, allow, ignore)
    root = Path(snapshot_download(repo, revision=revision, allow_patterns=list(allow),
                                  ignore_patterns=list(ignore) or None))
    return root, verify_snapshot(root, chosen, repo=repo, revision=revision)


# --- Token delle etichette --------------------------------------------------


def letter_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Id delle lettere A, B, C: un token ciascuna e distinti, altrimenti rifiuto."""
    ids = {}
    for letter in OPTION_LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RunnerError("LETTER_NOT_SINGLE_TOKEN", f"{letter}: {encoded}")
        ids[letter] = encoded[0]
    if len(set(ids.values())) != len(ids):
        raise RunnerError("LABEL_TOKENS_NOT_DISTINCT", str(ids))
    return ids


# Forme in cui un'etichetta può cominciare nel testo generato: attaccata o dopo uno spazio.
LABEL_FORMS = ("", " ")


def label_first_ids(tokenizer: Tokenizer, labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Primo token di ogni etichetta per forma; rifiuto se due etichette lo condividono."""
    forms = {}
    for form in LABEL_FORMS:
        ids = {label: tokenizer.encode(form + label, add_special_tokens=False)[0]
               for label in labels}
        if len(set(ids.values())) != len(ids):
            raise RunnerError("LABEL_TOKENS_NOT_DISTINCT", f"forma {form!r}: {ids}")
        forms[form] = ids
    return forms


def _label_step(tokenizer: Tokenizer, generated: Sequence[int],
                forms: dict[str, dict[str, int]], label: str) -> tuple[int, str] | None:
    """Primo passo il cui token apre l'etichetta come parola intera, con la sua forma.

    Il confine di parola esclude un token condiviso con un'altra parola (es. `Safety`
    per l'etichetta `Safe`).
    """
    for step, token in enumerate(generated):
        for form, ids in forms.items():
            if token != ids[label]:
                continue
            rest = tokenizer.decode(generated[step:], skip_special_tokens=True).lstrip()
            if rest.startswith(label) and not rest[len(label):len(label) + 1].isalnum():
                return step, form
    return None


# --- Spec delle baseline ------------------------------------------------------


@dataclass(frozen=True)
class BaselineSpec:
    # "options": logit delle lettere; "generate": testo greedy e logit; "openjet": runtime
    # della card, non implementato qui.
    mode: str
    template_kwargs: dict
    max_new_tokens: int | None = None
    trust_remote_code: bool = False
    unsupported: str | None = None  # motivo per cui l'ambiente dell'extra train non la carica


SPECS: dict[str, BaselineSpec] = {
    # Card dell'adapter: chat template con add_generation_prompt, greedy, 20 token.
    "granite-4.1-3b-policy-guardrails": BaselineSpec(
        "generate", {"add_generation_prompt": True}, max_new_tokens=20),
    # Card: vLLM a temperatura 0 con max_tokens=2048; no-think emette solo `<score>`:
    # 64 token sono una scelta nostra, un'uscita troncata resta OUTPUT_UNPARSEABLE.
    "granite-guardian-4.1-8b": BaselineSpec(
        "generate", {"add_generation_prompt": True}, max_new_tokens=64),
    # Card: apply_chat_template(messages, tokenize=False), max_new_tokens=128.
    "qwen3guard-gen-4b": BaselineSpec("generate", {}, max_new_tokens=128),
    # Card: trust_remote_code (modeling_spark.py), logit delle lettere all'ultima posizione.
    # Verificato il 2026-10-15: con transformers 5.17 `configuration_spark.py` fallisce in
    # `validate_rope` (AttributeError su `rope_parameters` per tipo di attenzione); il
    # config.json dichiara transformers 4.57.1 e la card non fissa una versione.
    "rizzo-flow-4b": BaselineSpec(
        "options", {"add_generation_prompt": True}, trust_remote_code=True, unsupported=(
            "configuration_spark.py non si carica con transformers 5.17 (validate_rope); "
            "config scritta con transformers 4.57.1: serve un ambiente separato")),
    "apus-reflex-4b": BaselineSpec(
        "openjet", {}, unsupported=(
            "runtime della card (openjet_runtime, OpenJet.decide) con transformers==5.16.1 e "
            "torch 2.8.0, incompatibile con l'extra train: serve un ambiente separato")),
}


def _chat_ids(tokenizer: Tokenizer, messages: list[dict], kwargs: dict) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def scorer_for(name: str, tokenizer: Tokenizer,
               backend: Backend) -> Callable[[Case], LabelReadout]:
    """Funzione caso -> LabelReadout della baseline; verifica prima i token delle etichette."""
    spec = SPECS[name]
    if spec.mode not in ("options", "generate"):
        raise RunnerError("UNSUPPORTED", f"{name}: {spec.unsupported}")
    baseline = BASELINES[name]

    if spec.mode == "options":
        letters = letter_ids(tokenizer)

        def score_options(case: Case) -> LabelReadout:
            request = build_input(name, case.criterion, case.evidence, case.surface)
            kwargs = {**spec.template_kwargs, **request.get("chat_template_kwargs", {})}
            logits = backend.next_logits(_chat_ids(tokenizer, request["messages"], kwargs))
            label_logits = {label: float(logits[letters[letter]])
                            for letter, label in zip(OPTION_LETTERS, request["options"])}
            return parse(name, label_logits=label_logits)

        return score_options

    forms = label_first_ids(tokenizer, list(baseline.labels))

    def score_generated(case: Case) -> LabelReadout:
        request = build_input(name, case.criterion, case.evidence, case.surface)
        ids = _chat_ids(tokenizer, request["messages"], spec.template_kwargs)
        generated, steps = backend.generate(ids, spec.max_new_tokens)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        label = parse(name, text=text).label  # errori del parser della card prima dei logit
        found = _label_step(tokenizer, generated, forms, label)
        if found is None:
            raise RunnerError("LABEL_TOKEN_NOT_FOUND", f"{label!r} non è un token generato")
        step, form = found
        label_logits = {key: float(steps[step][token]) for key, token in forms[form].items()}
        return parse(name, text=text, label_logits=label_logits)

    return score_generated


def fast_scorer(tokenizer: Tokenizer, backend: Backend, *, seed: int,
                bos_id: int | None = None) -> Callable[[Case], Readout]:
    """Il nostro Fast: prompt decision-token senza chat template, lettere all'ultima posizione.

    La permutazione delle opzioni è quella di `training_row` con un `Random(seed)`
    sulla sequenza dei casi; il gold non entra nel prompt. `bos_id`: id del BOS che
    la base dichiara (`scudo.base_models`), messo prima degli id del prompt invariato.
    """
    letters = letter_ids(tokenizer)
    rng = random.Random(seed)
    prefix = [] if bos_id is None else [bos_id]

    def score(case: Case) -> Readout:
        row = training_row(case.criterion, case.evidence, case.verdict, rng)
        logits = backend.next_logits(
            prefix + tokenizer.encode(row["prompt"], add_special_tokens=False))
        return read_options([float(logits[letters[x]]) for x in OPTION_LETTERS],
                            row["semantic_ids"])

    return score


# --- Esecuzione e report -------------------------------------------------------


def run(loaded: Loaded, score: Callable[[Case], Readout | LabelReadout]) -> list[EvalResult]:
    """Un risultato per caso; gli errori di dominio diventano il `code` del risultato.

    Altre eccezioni (bug, memoria) fermano il run: non si contano come risposte.
    """
    results = []
    for case_id, case in zip(loaded.case_ids, loaded.cases, strict=True):
        readout, error = None, None
        try:
            readout = score(case)
        except (BaselineError, RunnerError) as exc:
            error = exc.code
        except ValueError:  # readout malformato di read_options
            error = "READOUT_MALFORMED"
        results.append(EvalResult(case_id, case.language, case.surface, case.verdict,
                                  readout, error))
    return results


def reports_for(loaded: Loaded, results: Sequence[EvalResult], *, dedup: dict,
                **header) -> dict[str, dict]:
    """Report del benchmark e, se ci sono label non umane, quello sulle sole label umane.

    Ogni report cita conteggi del benchmark e verifica della deduplica.
    """
    out = {loaded.name: build_report(results, split=loaded.name, **header)}
    human = [item for item in results if item.case_id not in loaded.machine_labelled]
    if loaded.machine_labelled and human:
        name = f"{loaded.name}/human-labels"
        out[name] = build_report(human, split=name, **header)
    for report in out.values():
        report["benchmark"] = loaded.summary()
        report["dedup_verification"] = dedup
    return out


# --- Caricamento su Colab (dipendenze pesanti) -----------------------------------


class TransformersBackend:
    """Backend su un modello transformers già caricato; logit grezzi, generazione greedy."""

    def __init__(self, model):
        self.model = model

    def next_logits(self, ids):
        import torch

        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.model.device))
        return out.logits[0, -1].float().cpu()

    def generate(self, ids, max_new_tokens):
        import torch

        input_ids = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                do_sample=False, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_logits=True,
            )
        generated = out.sequences[0, len(ids):].tolist()
        return generated, [step[0].float().cpu() for step in out.logits]


def load_baseline(name: str):
    """(modello, tokenizer, identità dei pesi) di una baseline, solo dopo la verifica dei file."""
    spec = SPECS[name]
    if spec.unsupported:
        raise RunnerError("UNSUPPORTED", f"{name}: {spec.unsupported}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    baseline = BASELINES[name]
    root, weights = fetch_verified(baseline.repo, baseline.revision, ROOT_FILES, NO_SUBFOLDERS)
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=spec.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        root, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=spec.trust_remote_code)
    if baseline.adapter:
        from peft import PeftModel

        repo, revision, subfolder = baseline.adapter
        adapter_root, adapter_weights = fetch_verified(repo, revision, (f"{subfolder}/*",))
        model = PeftModel.from_pretrained(model, adapter_root / subfolder)
        weights += adapter_weights
    model.eval()
    return model, tokenizer, weights
