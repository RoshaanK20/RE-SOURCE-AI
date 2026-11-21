"""Local and OpenAI-compatible LLM adapters with actionable Ollama diagnostics.

The Ollama API is accessed through stdlib urllib. No credentials are required for
local Ollama; no network calls are made unless a user requests a model operation.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLMError(RuntimeError):
    """Readable failure in a model call."""


class LLMConnectionError(LLMError):
    pass


class LLMModelError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


def _strip_thinking(text: str) -> str:
    # Some coder models include thinking text inline. It must never become
    # part of the SEARCH/REPLACE source patch.
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _normal_host(host: str | None) -> str:
    value = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    from urllib.parse import urlsplit
    parsed = urlsplit(value)
    if not parsed.hostname or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query:
        raise ValueError("OLLAMA_HOST must be a host and port, e.g. http://127.0.0.1:11434")
    return value


def _json_get(url: str, timeout: float = 3) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post(url: str, payload: dict, headers: dict | None = None, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_message(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read(2048).decode("utf-8", errors="replace")
        data = json.loads(raw)
        return str(data.get("error") or data.get("message") or raw).strip()[:300]
    except (ValueError, OSError, UnicodeError):
        return f"HTTP {error.code}"


def probe_ollama(model: str = "qwen2.5-coder:7b", host: str | None = None,
                 *, smoke: bool = False, timeout: int = 30) -> dict:
    """Probe server, model availability and optionally a REAL completion.

    A successful /api/tags call does NOT establish that inference works. The
    smoke test uses /api/chat and checks for a non-empty completion.
    Returns a JSON-serializable observation with no guessed success.
    """
    url = _normal_host(host)
    result = {"host": url, "requested_model": model, "server_reachable": False,
              "model_installed": False, "completion_tested": False,
              "completion_ok": False, "installed_models": [], "stage": "connection", "error": ""}
    try:
        data = _json_get(f"{url}/api/tags", timeout=min(timeout, 5))
        result["server_reachable"] = True
        result["installed_models"] = sorted({str(m.get("name", "")) for m in data.get("models", []) if m.get("name")})
        # An explicit tag is required. Qwen2.5-Coder 7B and 3B are different
        # experiments, so a broad fuzzy model-name match would be misleading.
        result["model_installed"] = model in result["installed_models"]
        if not result["model_installed"]:
            result["stage"] = "model_missing"
            result["error"] = f"Model '{model}' is not installed. Run: ollama pull {model}"
            return result
        result["stage"] = "ready_for_completion"
        if smoke:
            result["completion_tested"] = True
            answer = OllamaLLM(model=model, host=url, timeout=timeout).chat([
                {"role": "user", "content": "Reply with only the word READY."}
            ])
            result["completion_ok"] = bool(answer.text)
            result["completion_excerpt"] = answer.text[:120]
            result["seconds"] = round(answer.seconds, 3)
            result["stage"] = "completion_ok" if answer.text else "empty_completion"
        return result
    except urllib.error.HTTPError as exc:
        result["stage"] = "http_error"
        result["error"] = f"Ollama HTTP {exc.code}: {_http_message(exc)}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["stage"] = "connection"
        result["error"] = f"Cannot reach Ollama at {url}: {exc}. Start Ollama and check OLLAMA_HOST."
    except (ValueError, KeyError, TypeError, LLMError) as exc:
        result["stage"] = "response_error"
        result["error"] = f"Ollama response problem: {type(exc).__name__}: {exc}"
    return result


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5-coder:7b", host: str | None = None,
                 temperature: float = 0.0, num_ctx: int = 8192, seed: int = 0, timeout: int = 600):
        self.model, self.temperature, self.num_ctx, self.seed, self.timeout = (
            model, temperature, num_ctx, seed, timeout)
        self.host = _normal_host(host)
        self.name = f"ollama:{model}"

    def chat(self, messages: list[dict]) -> LLMReply:
        start = time.monotonic()
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": self.temperature,
                               "num_ctx": self.num_ctx, "seed": self.seed}}
        data = None
        for attempt in range(2):
            try:
                data = _post(f"{self.host}/api/chat", payload, timeout=self.timeout)
                break
            except urllib.error.HTTPError as exc:
                detail = _http_message(exc)
                if exc.code in (400, 404) or "model" in detail.lower() and "not found" in detail.lower():
                    raise LLMModelError(
                        f"Ollama rejected model '{self.model}' (HTTP {exc.code}: {detail}). "
                        f"Check 'ollama list' and run 'ollama pull {self.model}'.") from exc
                if attempt == 1 or exc.code not in (429, 500, 502, 503, 504):
                    raise LLMResponseError(f"Ollama HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if attempt == 1:
                    raise LLMConnectionError(
                        f"Ollama connection failed at {self.host}: {exc}. "
                        "Start Ollama and run the LLM diagnostic check.") from exc
            if attempt == 0:
                time.sleep(0.5)
        if not isinstance(data, dict):
            raise LLMResponseError("Ollama did not return a JSON object")
        content = data.get("message") or {}
        if not isinstance(content, dict):
            raise LLMResponseError("Ollama response is missing a valid message")
        answer = _strip_thinking(content.get("content") or "")
        if not answer:
            problem = data.get("error") or "Model returned an empty completion"
            raise LLMResponseError(
                f"{problem}. Try a smaller coder model, increase the context limit, "
                "or inspect Ollama logs for resource errors.")
        return LLMReply(answer, int(data.get("prompt_eval_count") or 0),
                        int(data.get("eval_count") or 0), time.monotonic() - start)


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 temperature: float = 0.0, seed: int = 0):
        self.model, self.base_url, self.temperature, self.seed = model, base_url.rstrip('/'), temperature, seed
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        self.name = f"openai-compat:{model}"

    def chat(self, messages: list[dict]) -> LLMReply:
        start = time.monotonic()
        data = _post(f"{self.base_url}/chat/completions",
                     {"model": self.model, "messages": messages,
                      "temperature": self.temperature, "seed": self.seed},
                     headers={"Authorization": f"Bearer {self.api_key}"})
        usage = data.get("usage") or {}
        return LLMReply(_strip_thinking(data["choices"][0]["message"]["content"]),
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                        time.monotonic() - start)


class ScriptedLLM:
    """Replays predefined answers; testing only, NEVER a real Qwen result."""

    def __init__(self, answers: list[str], name: str = "scripted"):
        self.answers, self.calls, self.name = list(answers), [], name

    def chat(self, messages: list[dict]) -> LLMReply:
        self.calls.append(messages)
        text = self.answers.pop(0) if self.answers else ""
        return LLMReply(text)


def make_llm(spec: str, seed: int = 0, temperature: float = 0.0):
    if spec.startswith("openai:"):
        model, _, base = spec[len("openai:"):].partition("@")
        return OpenAICompatLLM(model, base or "http://localhost:8000/v1",
                               temperature=temperature, seed=seed)
    if spec.startswith("ollama:"):
        spec = spec[len("ollama:"):]
    return OllamaLLM(spec, temperature=temperature, seed=seed)
