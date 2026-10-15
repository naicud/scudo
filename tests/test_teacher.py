import json

import pytest

from scudo import teacher
from scudo.ingest import REPO_ROOT
from scudo.protocol import SEMANTIC_DESCRIPTIONS
from scudo.teacher import CANDIDATES, ConfigError, Malformed, parse_configs, parse_content
from scudo.teacher_client import Client

KEY = "test-key-not-real-0123456789"
CONFIG = {
    "model_id": "openai/gpt-oss-120b", "provider": "Provider X",
    "provider_model": "gpt-oss-120b", "base_url": "https://api.example.test/v1",
    "api_key_env": "SCUDO_TEST_KEY", "terms_url": "https://example.test/terms",
    "terms_checked_by": "Mario Rossi", "terms_checked_on": "2026-10-20",
    "terms_note": "output utilizzabili per training", "input_usd_per_mtok": 0.1,
    "output_usd_per_mtok": 0.5, "max_tokens": 1000,
}
MESSAGES = teacher.messages("Is it unsafe?", "How do I bake bread?")


def config(**changes):
    [cfg] = parse_configs({"teacher": [{**CONFIG, **changes}]})
    return cfg


def body(content='{"verdict": "VIOLATION", "reasoning": "It asks for a weapon."}',
         finish="stop", refusal=None, usage=True):
    out = {"choices": [{"message": {"content": content, "refusal": refusal},
                        "finish_reason": finish}]}
    if usage:
        out["usage"] = {"prompt_tokens": 120, "completion_tokens": 30,
                        "completion_tokens_details": {"reasoning_tokens": 12}}
    return json.dumps(out).encode()


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, json.loads(payload), timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def client(*responses, **changes):
    transport = FakeTransport(*responses)
    sleeps = []
    return Client(config(**changes), transport=transport, sleep=sleeps.append,
                  environ={"SCUDO_TEST_KEY": KEY}), transport, sleeps


def test_candidates_are_pinned_apache_models():
    assert set(CANDIDATES) == {"openai/gpt-oss-120b", "Qwen/Qwen3.5-397B-A17B"}
    for candidate in CANDIDATES.values():
        assert len(candidate.revision) == 40 and candidate.license == "apache-2.0"


def test_valid_config_parses():
    cfg = config()
    assert cfg.served_revision == "NON VERIFICATO" and cfg.max_tokens == 1000


@pytest.mark.parametrize("field", ["provider", "terms_url", "terms_checked_by", "api_key_env",
                                   "base_url", "input_usd_per_mtok", "max_tokens"])
def test_missing_required_field_is_refused(field):
    data = {key: value for key, value in CONFIG.items() if key != field}
    with pytest.raises(ConfigError, match=field):
        parse_configs({"teacher": [data]})


@pytest.mark.parametrize("changes", [
    {"provider": "  "}, {"model_id": "org/unknown"}, {"input_usd_per_mtok": 0},
    {"output_usd_per_mtok": -1}, {"max_tokens": 0}, {"base_url": "http://api.example.test"},
    {"unexpected": "x"}, {"input_usd_per_mtok": "0.1"},
])
def test_invalid_config_is_refused(changes):
    with pytest.raises(ConfigError):
        config(**changes)


def test_duplicate_teacher_and_empty_list_are_refused():
    with pytest.raises(ConfigError):
        parse_configs({"teacher": [CONFIG, CONFIG]})
    with pytest.raises(ConfigError):
        parse_configs({})


def test_config_inside_the_repository_is_refused():
    with pytest.raises(ValueError):
        teacher.load_configs(REPO_ROOT / "teachers.toml")


def test_messages_carry_criterion_evidence_and_tri_state_semantics():
    text = json.dumps(MESSAGES, ensure_ascii=False)
    assert "Is it unsafe?" in text and "How do I bake bread?" in text
    for description in SEMANTIC_DESCRIPTIONS.values():
        assert description in text


@pytest.mark.parametrize("content", [
    '{"verdict": "NO_VIOLATION", "reasoning": "Benign."}',
    '```json\n{"verdict": "NO_VIOLATION", "reasoning": "Benign."}\n```',
])
def test_strict_json_is_parsed(content):
    assert parse_content(content) == ("NO_VIOLATION", "Benign.")


@pytest.mark.parametrize("content", [
    "NO_VIOLATION", '{"verdict": "SAFE", "reasoning": "x"}', '{"verdict": "VIOLATION"}',
    '{"verdict": "VIOLATION", "reasoning": "x", "score": 1}', '{"verdict": "VIOLATION", "reasoning": ""}',
    'Sure: {"verdict": "VIOLATION", "reasoning": "x"}', "", None,
])
def test_anything_else_is_malformed(content):
    with pytest.raises(Malformed):
        parse_content(content)


def test_successful_call_captures_usage_and_never_exposes_the_key():
    judge, transport, _ = client((200, body()))
    outcome = judge.judge(MESSAGES)
    assert (outcome.status, outcome.verdict) == ("ok", "VIOLATION")
    assert (outcome.prompt_tokens, outcome.completion_tokens, outcome.reasoning_tokens) == (120, 30, 12)
    url, headers, payload, _ = transport.calls[0]
    assert url == "https://api.example.test/v1/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert payload["model"] == "gpt-oss-120b" and payload["max_tokens"] == 1000
    assert KEY not in repr(judge) and KEY not in repr(outcome)


def test_request_extra_is_merged_into_the_body():
    judge, transport, _ = client((200, body()), request_extra={"reasoning_effort": "medium"})
    judge.judge(MESSAGES)
    assert transport.calls[0][2]["reasoning_effort"] == "medium"


def test_missing_key_refuses_before_any_call():
    with pytest.raises(ConfigError, match="SCUDO_TEST_KEY"):
        Client(config(), transport=FakeTransport(), environ={})


def test_429_and_5xx_are_retried_with_backoff():
    judge, transport, sleeps = client((429, b"{}"), (503, b"{}"), (200, body()))
    assert judge.judge(MESSAGES).status == "ok"
    assert len(transport.calls) == 3 and sleeps == [2.0, 4.0]


def test_persistent_5xx_is_an_error_never_no_violation():
    judge, transport, _ = client(*[(500, b"{}")] * 5)
    outcome = judge.judge(MESSAGES)
    assert (outcome.status, outcome.verdict) == ("error", None)
    assert len(transport.calls) == 5


def test_4xx_is_not_retried():
    judge, transport, _ = client((400, b'{"error": {"message": "bad request"}}'))
    outcome = judge.judge(MESSAGES)
    assert outcome.status == "error" and "bad request" in outcome.detail
    assert len(transport.calls) == 1


def test_timeout_is_recorded_and_not_retried():
    judge, transport, _ = client(TimeoutError("read timed out"))
    outcome = judge.judge(MESSAGES)
    assert (outcome.status, outcome.verdict, outcome.prompt_tokens) == ("timeout", None, None)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("response, status", [
    (body(finish="content_filter", content=None), "refused"),
    (body(refusal="I can't help with that.", content=None), "refused"),
    (body(content="I'm sorry, but I can't help with that."), "refused"),
    (body(content="Mi dispiace, non posso aiutarti."), "refused"),
    (body(finish="length", content='{"verdict": "VIO'), "truncated"),
    (body(content="The answer is NO_VIOLATION"), "malformed"),
    (b"not json", "malformed"),
])
def test_refused_truncated_and_malformed_have_no_verdict(response, status):
    judge, _, _ = client((200, response))
    outcome = judge.judge(MESSAGES)
    assert (outcome.status, outcome.verdict) == (status, None)


@pytest.mark.parametrize("content", ['{"translation": "Ciao"}',
                                     '```json\n{"translation": "Ciao"}\n```'])
def test_single_text_field_is_parsed(content):
    assert teacher.parse_field(content, "translation") == "Ciao"


@pytest.mark.parametrize("content", [None, "Ciao", '{"translation": ""}', '{"translation": 3}',
                                     '{"translation": "Ciao", "note": "x"}', '{"text": "Ciao"}'])
def test_field_output_otherwise_is_malformed(content):
    with pytest.raises(Malformed):
        teacher.parse_field(content, "translation")


def test_complete_returns_the_text_field_with_usage_and_no_verdict():
    c, transport, _ = client((200, body('{"translation": "Come cuocio il pane?"}')))
    outcome = c.complete(MESSAGES, "translation")
    assert outcome.status == "ok" and outcome.text == "Come cuocio il pane?"
    assert outcome.verdict is None and outcome.prompt_tokens == 120
    assert transport.calls[0][2]["messages"] == MESSAGES


@pytest.mark.parametrize(("response", "status"), [
    (body("I'm sorry, I can't help with that."), "refused"),
    (body('{"verdict": "VIOLATION", "reasoning": "x"}'), "malformed"),
    (body('{"translation": "Ciao"}', finish="length"), "truncated"),
])
def test_complete_failures_have_no_text(response, status):
    c, _, _ = client((200, response))
    outcome = c.complete(MESSAGES, "translation")
    assert outcome.status == status and outcome.text is None
