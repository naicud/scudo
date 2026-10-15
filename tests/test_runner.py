import hashlib
from collections import defaultdict

import pytest

from scudo.aegis import HARMFUL_REQUEST
from scudo.artifacts import ArtifactError
from scudo.benchmarks import Loaded
from scudo.cases import Case
from scudo.protocol import SEMANTIC_IDS
from scudo.runner import (
    NO_SUBFOLDERS,
    ROOT_FILES,
    SPECS,
    RemoteFile,
    RunnerError,
    fast_scorer,
    git_blob_sha1,
    label_first_ids,
    letter_ids,
    reports_for,
    run,
    scorer_for,
    selected,
    verify_snapshot,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeTokenizer:
    """Pezzi multi-carattere preferiti (match più lungo), ogni altro carattere è un token."""

    def __init__(self, pieces=()):
        self.pieces = sorted(pieces, key=len, reverse=True)
        self.ids: dict[str, int] = {}
        self.templates = []

    def _id(self, piece):
        return self.ids.setdefault(piece, len(self.ids))

    def encode(self, text, add_special_tokens=True):
        out, i = [], 0
        while i < len(text):
            piece = next((p for p in self.pieces if text.startswith(p, i)), text[i])
            out.append(self._id(piece))
            i += len(piece)
        return out

    def decode(self, ids, skip_special_tokens=False):
        by_id = {v: k for k, v in self.ids.items()}
        return "".join(by_id[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        assert tokenize is False
        self.templates.append(kwargs)
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<gen>" if kwargs.get("add_generation_prompt") else "")


class FakeBackend:
    def __init__(self, next_logits=None, generated_text=None, tokenizer=None, step_logits=None):
        self._next = next_logits or {}
        self._text, self._tok, self._steps = generated_text, tokenizer, step_logits or {}
        self.prompts = []

    def next_logits(self, ids):
        self.prompts.append(ids)
        return defaultdict(float, self._next)

    def generate(self, ids, max_new_tokens):
        self.prompts.append(ids)
        generated = self._tok.encode(self._text)
        steps = [defaultdict(float, self._steps.get(t, {})) for t in range(len(generated))]
        return generated, steps


def case(surface="prompt", verdict="VIOLATION", source_id="r1", criterion="Is it harmful?"):
    return Case("org/bench", source_id, "en", surface, criterion, "evidence", verdict)


def loaded_of(*cases, machine=()):
    ids = tuple(f"bench:{c.source_id}:{c.surface}:{c.language}" for c in cases)
    return Loaded("bench", "org/bench", "default", SHA, ("test",), len(cases), cases, ids,
                  frozenset(ids[i] for i in machine), {"prompt REDACTED": 2})


# --- verifica dei file


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"hello\n")
    assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def snapshot(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return [
        RemoteFile("config.json", git_blob_sha1(tmp_path / "config.json"), None),
        RemoteFile("model.safetensors", "pointer", hashlib.sha256(b"weights").hexdigest()),
    ]


def test_verified_snapshot_returns_only_weight_identities(tmp_path):
    [identity] = verify_snapshot(tmp_path, snapshot(tmp_path), repo="org/model", revision=SHA)
    assert identity.filename == "model.safetensors"
    assert identity.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert identity.revision == SHA


def test_weight_with_other_digest_blocks(tmp_path):
    files = snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ArtifactError) as error:
        verify_snapshot(tmp_path, files, repo="org/model", revision=SHA)
    assert error.value.code == "ARTIFACT_DIGEST_MISMATCH"


def test_non_lfs_file_with_other_blob_or_missing_blocks(tmp_path):
    files = snapshot(tmp_path)
    (tmp_path / "config.json").write_text('{"trust": "me"}')
    with pytest.raises(ArtifactError) as error:
        verify_snapshot(tmp_path, files, repo="org/model", revision=SHA)
    assert error.value.code == "ARTIFACT_DIGEST_MISMATCH"
    (tmp_path / "config.json").unlink()
    with pytest.raises(ArtifactError) as error:
        verify_snapshot(tmp_path, files, repo="org/model", revision=SHA)
    assert error.value.code == "ARTIFACT_MISSING"


def test_snapshot_without_weights_is_refused(tmp_path):
    with pytest.raises(RunnerError):
        verify_snapshot(tmp_path, snapshot(tmp_path)[:1], repo="org/model", revision=SHA)


def test_root_patterns_skip_subfolders_and_gguf():
    files = [RemoteFile(p, "b", None) for p in (
        "config.json", "model-00001-of-00002.safetensors", "modeling_spark.py",
        "tokenizer.json", "merges.txt", "rizzo-flow-4b-bf16.gguf", "README.md",
        "lora-1.7b/adapter_model.safetensors", "lora-1.7b/adapter_config.json")]
    chosen = [f.path for f in selected(files, ROOT_FILES, NO_SUBFOLDERS)]
    assert chosen == ["config.json", "model-00001-of-00002.safetensors", "modeling_spark.py",
                      "tokenizer.json", "merges.txt"]


# --- token delle etichette


class Encoded(FakeTokenizer):
    def __init__(self, fixed):
        super().__init__()
        self.fixed = fixed

    def encode(self, text, add_special_tokens=True):
        return self.fixed.get(text) or super().encode(text)


def test_letters_must_be_single_distinct_tokens():
    assert set(letter_ids(FakeTokenizer())) == {"A", "B", "C"}
    with pytest.raises(RunnerError) as error:
        letter_ids(Encoded({"A": [7], "B": [7], "C": [8]}))
    assert error.value.code == "LABEL_TOKENS_NOT_DISTINCT"
    with pytest.raises(RunnerError) as error:
        letter_ids(Encoded({"B": [1, 2]}))
    assert error.value.code == "LETTER_NOT_SINGLE_TOKEN"


def test_labels_sharing_the_first_token_refuse_the_baseline():
    # "Yes" e "Yellow" cominciano entrambi con il pezzo "Ye".
    with pytest.raises(RunnerError) as error:
        label_first_ids(FakeTokenizer(pieces=("Ye", " Ye")), ["Yes", "Yellow"])
    assert error.value.code == "LABEL_TOKENS_NOT_DISTINCT"


# --- baseline


QWEN_PIECES = ("Saf", " Saf", "Un", " Un", "Contro", " Contro")


def qwen_backend(text, step_logits):
    tokenizer = FakeTokenizer(QWEN_PIECES)
    return tokenizer, FakeBackend(generated_text=text, tokenizer=tokenizer,
                                  step_logits=step_logits)


def test_generated_label_reads_first_token_logits_at_the_step_that_emits_it():
    tokenizer, _ = qwen_backend("", {})
    forms = label_first_ids(tokenizer, ["Unsafe", "Safe", "Controversial"])
    text = "Safety: Safe\nCategories: None"
    # Passo 0 è "Saf" di "Safety": non è l'etichetta. Il passo giusto è " Saf" (5).
    step = len(tokenizer.encode("Safety:"))
    logits = {step: {forms[" "]["Safe"]: 3.0, forms[" "]["Unsafe"]: 1.0,
                     forms[" "]["Controversial"]: 0.0},
              0: {forms[""]["Unsafe"]: 9.0}}
    tokenizer, backend = qwen_backend(text, logits)
    readout = scorer_for("qwen3guard-gen-4b", tokenizer, backend)(case())
    assert readout.label == "Safe" and readout.verdict == "NO_VIOLATION"
    assert readout.margin == pytest.approx(2.0)
    assert readout.probabilities[SEMANTIC_IDS.index("NO_VIOLATION")] > 0.8
    assert tokenizer.templates == [{}]  # template della card, senza add_generation_prompt


def test_text_logit_mismatch_and_unparseable_output_are_errors_in_the_results():
    tokenizer, _ = qwen_backend("", {})
    forms = label_first_ids(tokenizer, ["Unsafe", "Safe", "Controversial"])
    step = len(tokenizer.encode("Safety:"))
    tokenizer, backend = qwen_backend("Safety: Safe", {step: {forms[" "]["Unsafe"]: 5.0}})
    results = run(loaded_of(case()), scorer_for("qwen3guard-gen-4b", tokenizer, backend))
    assert results[0].error == "LOGIT_TEXT_MISMATCH" and results[0].predicted == "ERROR"
    tokenizer, backend = qwen_backend("I cannot tell.", {})
    results = run(loaded_of(case()), scorer_for("qwen3guard-gen-4b", tokenizer, backend))
    assert results[0].error == "OUTPUT_UNPARSEABLE"


def test_label_glued_to_another_token_is_an_error_not_a_fallback():
    # Il tokenizer fonde `"No` in un solo pezzo: il primo token dell'etichetta non compare.
    tokenizer = FakeTokenizer(('"No', "Amb", " No", " Yes", " Amb"))
    backend = FakeBackend(generated_text='{"score": "No"}', tokenizer=tokenizer)
    scorer = scorer_for("granite-4.1-3b-policy-guardrails", tokenizer, backend)
    [result] = run(loaded_of(case(criterion=HARMFUL_REQUEST)), scorer)
    assert result.error == "LABEL_TOKEN_NOT_FOUND"


def test_granite_policy_uses_the_card_generation_settings():
    tokenizer = FakeTokenizer(("Amb", " No", " Yes", " Amb"))
    forms = label_first_ids(tokenizer, ["No", "Yes", "Ambiguous"])
    step = len(tokenizer.encode('{"score": "'))
    backend = FakeBackend(generated_text='{"score": "No"}', tokenizer=tokenizer,
                          step_logits={step: {forms[""]["No"]: 4.0, forms[""]["Yes"]: 1.0}})
    readout = scorer_for("granite-4.1-3b-policy-guardrails", tokenizer, backend)(
        case(criterion=HARMFUL_REQUEST))
    assert readout.verdict == "VIOLATION"
    assert tokenizer.templates == [{"add_generation_prompt": True}]
    assert SPECS["granite-4.1-3b-policy-guardrails"].max_new_tokens == 20


def test_granite_policy_criterion_without_rule_is_unsupported_and_never_generated():
    tokenizer = FakeTokenizer(("Amb", " No", " Yes", " Amb"))
    backend = FakeBackend(generated_text='{"score": "Yes"}', tokenizer=tokenizer)
    [result] = run(loaded_of(case(verdict="NO_VIOLATION")),
                   scorer_for("granite-4.1-3b-policy-guardrails", tokenizer, backend))
    assert result.error == "UNSUPPORTED" and result.readout is None
    assert backend.prompts == []


def test_unsupported_surface_is_an_error_result():
    tokenizer, backend = qwen_backend("Safety: Safe", {})
    [result] = run(loaded_of(case(surface="response")),
                   scorer_for("qwen3guard-gen-4b", tokenizer, backend))
    assert result.error == "UNSUPPORTED"


def test_rizzo_reads_letter_logits_with_thinking_disabled():
    tokenizer = FakeTokenizer()
    ids = letter_ids(tokenizer)
    # Lettere di Rizzo: A false, B true, C astensione.
    backend = FakeBackend(next_logits={ids["A"]: 0.5, ids["B"]: 2.5, ids["C"]: -1.0})
    readout = scorer_for("rizzo-flow-4b", tokenizer, backend)(case())
    assert readout.label == "true" and readout.verdict == "VIOLATION"
    assert readout.margin == pytest.approx(2.0)
    assert tokenizer.templates == [{"add_generation_prompt": True, "enable_thinking": False}]


@pytest.mark.parametrize(("name", "reason"), [
    ("rizzo-flow-4b", "transformers 5.17"),
    ("apus-reflex-4b", "transformers==5.16.1"),
])
def test_baselines_outside_the_train_environment_are_refused_before_loading(name, reason):
    from scudo.runner import load_baseline

    with pytest.raises(RunnerError) as error:
        load_baseline(name)  # nessun import pesante, nessun download
    assert error.value.code == "UNSUPPORTED" and reason in str(error.value)


def test_apus_is_refused_explicitly():
    with pytest.raises(RunnerError) as error:
        scorer_for("apus-reflex-4b", FakeTokenizer(), FakeBackend())
    assert error.value.code == "UNSUPPORTED"
    assert "transformers==5.16.1" in str(error.value)


# --- Fast


def test_fast_reads_the_decision_token_prompt_without_the_gold():
    tokenizer = FakeTokenizer()
    ids = letter_ids(tokenizer)
    backend = FakeBackend(next_logits={ids["A"]: 1.0, ids["B"]: 3.0, ids["C"]: 0.0})
    score = fast_scorer(tokenizer, backend, seed=7)
    readouts = [score(case(verdict=v, source_id=v)) for v in SEMANTIC_IDS]
    prompts = [tokenizer.decode(p) for p in backend.prompts]
    assert all(p.endswith("\nDecision:\n") and "<gen>" not in p for p in prompts)
    # Stesse logit per lettera, permutazioni diverse: il verdetto segue la lettera B.
    again = fast_scorer(FakeTokenizer(), FakeBackend(next_logits=backend._next), seed=7)
    assert [r.verdict for r in readouts] == [
        again(case(verdict=v, source_id=v)).verdict for v in SEMANTIC_IDS]
    assert all(r.margin == pytest.approx(2.0) for r in readouts)


def test_fast_bos_is_a_declared_id_before_the_unchanged_prompt():
    tokenizer = FakeTokenizer()
    plain, with_bos = FakeBackend(), FakeBackend()
    fast_scorer(tokenizer, plain, seed=7)(case())
    fast_scorer(tokenizer, with_bos, seed=7, bos_id=999)(case())
    assert with_bos.prompts[0] == [999, *plain.prompts[0]]  # stessi byte, solo l'id in testa


def test_malformed_readout_is_an_error_and_other_exceptions_stop_the_run():
    def bad(case):
        raise ValueError("readout malformato")

    [result] = run(loaded_of(case()), bad)
    assert result.error == "READOUT_MALFORMED"

    def crash(case):
        raise MemoryError

    with pytest.raises(MemoryError):
        run(loaded_of(case()), crash)


# --- report


def test_reports_cite_benchmark_and_dedup_and_split_human_labels():
    cases = (case(source_id="a"), case(source_id="b", verdict="NO_VIOLATION"))
    loaded = loaded_of(*cases, machine=(1,))
    tokenizer = FakeTokenizer()
    ids = letter_ids(tokenizer)
    backend = FakeBackend(next_logits={ids["A"]: 0.0, ids["B"]: 2.0, ids["C"]: -1.0})
    results = run(loaded, scorer_for("rizzo-flow-4b", tokenizer, backend))
    dedup = {"status": "FAIL", "problems": ["wildguardtest: non indicizzato dalla deduplica"]}
    reports = reports_for(loaded, results, dedup=dedup, proof_type="reference",
                          model="rizzo-flow-4b", weights=verify_weights(), scudo_sha=SHA,
                          runtime={"name": "transformers", "version": "5.17.0"},
                          harness="test", format="spark-decisions-v3")
    assert set(reports) == {"bench", "bench/human-labels"}
    assert reports["bench"]["overall"]["cases"] == 2
    assert reports["bench/human-labels"]["overall"]["cases"] == 1
    assert reports["bench"]["benchmark"]["dropped"] == {"prompt REDACTED": 2}
    assert reports["bench"]["dedup_verification"] == dedup


def verify_weights():
    from scudo.artifacts import ArtifactIdentity

    return [ArtifactIdentity("org/model", SHA, "model.safetensors", "0" * 64)]


def test_a_benchmark_the_model_cannot_score_at_all_gets_no_report_instead_of_stopping_the_run():
    from scudo.eval_report import EvalResult

    cases = (case(source_id="a"), case(source_id="b", verdict="NO_VIOLATION"))
    loaded = loaded_of(*cases, machine=(1,))
    results = [EvalResult(f"bench:{c.source_id}", c.language, c.surface, c.verdict, None,
                          error="UNSUPPORTED") for c in cases]
    reports = reports_for(loaded, results, dedup={"status": "NOT_RUN"}, proof_type="reference",
                          model="granite-4.1-3b-policy-guardrails", weights=verify_weights(),
                          scudo_sha=SHA, runtime={"name": "transformers", "version": "5.17.0"},
                          harness="test", format="policy")
    assert reports == {}
