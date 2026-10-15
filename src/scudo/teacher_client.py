"""Client chat completions compatibile OpenAI su urllib, solo standard library.

Ritenta con backoff esponenziale solo 429 e 5xx; registra l'uso di token della
risposta. La chiave si legge dalla variabile d'ambiente della configurazione, va
solo nell'header e non compare mai in record, repr o errori. Rifiuto,
troncamento, output malformato, timeout ed errore restano tali, senza verdetto:
non diventano mai `NO_VIOLATION`.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from scudo.protocol import Verdict
from scudo.teacher import ConfigError, Malformed, TeacherConfig, parse_content

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


# Euristica dichiarata: un testo non JSON che apre con una formula di rifiuto.
_REFUSAL = re.compile(r"^\W*(i'?m sorry|i am sorry|sorry|i can'?t|i cannot|i won'?t|i'?m unable|"
                      r"i am unable|mi dispiace|non posso)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Outcome:
    status: str  # ok, refused, truncated, malformed, timeout, error
    verdict: Verdict | None = None
    reasoning: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    content: str | None = None  # testo restituito, per l'audit
    detail: str = ""
    http_status: int | None = None  # solo per gli errori HTTP


Transport = Callable[[str, dict, bytes, float], tuple[int, bytes]]


def urllib_transport(url: str, headers: dict, payload: bytes, timeout: float) -> tuple[int, bytes]:
    """POST con urllib; un timeout diventa TimeoutError, un errore di rete OSError."""
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError(str(error.reason)) from None
        raise OSError(f"errore di rete: {error.reason}") from None


def _int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _error_detail(status: int, raw: bytes) -> str:
    try:
        message = json.loads(raw)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = ""
    return f"HTTP {status}" + (f": {str(message)[:200]}" if message else "")


class Client:
    """Un teacher configurato; ritenta solo 429 e 5xx, con backoff esponenziale."""

    def __init__(self, config: TeacherConfig, *, transport: Transport = urllib_transport,
                 sleep: Callable[[float], None] = time.sleep, environ: Mapping = os.environ,
                 retries: int = 4, backoff_s: float = 2.0):
        key = environ.get(config.api_key_env, "")
        if not key.strip():
            raise ConfigError([f"{config.model_id}: variabile d'ambiente "
                               f"{config.api_key_env} vuota o assente"])
        self.config = config
        self._key = key
        self._transport, self._sleep = transport, sleep
        self._retries, self._backoff = retries, backoff_s

    def __repr__(self) -> str:
        return f"Client({self.config.model_id} via {self.config.provider})"

    def judge(self, chat: list[dict]) -> Outcome:
        cfg = self.config
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        body = {**cfg.request_extra, "model": cfg.provider_model, "messages": chat,
                "max_tokens": cfg.max_tokens}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        payload = json.dumps(body, ensure_ascii=False).encode()
        for attempt in range(self._retries + 1):
            try:
                status, raw = self._transport(url, headers, payload, cfg.timeout_s)
            except TimeoutError:
                return Outcome("timeout", detail=f"nessuna risposta entro {cfg.timeout_s} s")
            except OSError as error:
                return Outcome("error", detail=str(error)[:200])
            if status == 200:
                return _outcome(raw)
            if status not in RETRY_STATUSES:
                return Outcome("error", detail=_error_detail(status, raw), http_status=status)
            if attempt < self._retries:
                self._sleep(self._backoff * 2 ** attempt)
        return Outcome("error", detail=f"{_error_detail(status, raw)} dopo {self._retries + 1} "
                       "tentativi", http_status=status)


def _outcome(raw: bytes) -> Outcome:
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        message = choice.get("message") or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return Outcome("malformed", detail="risposta API non conforme a chat completions")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    content = message.get("content") if isinstance(message.get("content"), str) else None
    base = {"prompt_tokens": _int(usage.get("prompt_tokens")),
            "completion_tokens": _int(usage.get("completion_tokens")),
            "reasoning_tokens": _int(details.get("reasoning_tokens")),
            "finish_reason": choice.get("finish_reason"), "content": content}
    refusal = message.get("refusal")
    if base["finish_reason"] == "content_filter" or (isinstance(refusal, str) and refusal.strip()):
        return Outcome("refused", detail=str(refusal or "content_filter")[:200], **base)
    if base["finish_reason"] == "length":
        return Outcome("truncated", detail="max_tokens raggiunto", **base)
    try:
        verdict, reasoning = parse_content(content)
    except Malformed as error:
        if content and _REFUSAL.match(content):
            return Outcome("refused", detail="formula di rifiuto (euristica)", **base)
        return Outcome("malformed", detail=str(error), **base)
    return Outcome("ok", verdict=verdict, reasoning=reasoning, **base)
