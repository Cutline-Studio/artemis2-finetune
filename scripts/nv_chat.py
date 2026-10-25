"""Thin client for integrate.api.nvidia.com. Key stays in the env; never printed."""
from __future__ import annotations
import json, os, time, urllib.error, urllib.request
from typing import Any

DEFAULT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class RateLimited(Exception):
    pass


def chat(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
    enable_thinking: bool = False,
    timeout: int = 120,
    retries: int = 4,
) -> dict[str, Any]:
    key = os.environ.get("NVIDIA_BUILD_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_BUILD_API_KEY is not set")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if seed is not None:
        body["seed"] = seed
    if model.startswith("nvidia/nemotron-3"):
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    data = json.dumps(body).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            DEFAULT_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode(errors="replace")[:400]
            if e.code in (429, 503, 502):
                wait = 2 ** attempt + 1
                time.sleep(wait)
                last_err = RateLimited(f"{e.code}: {body_txt}")
                continue
            raise RuntimeError(f"NVIDIA API {e.code}: {body_txt}") from None
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"NVIDIA API failed after retries: {last_err}")


def reply_text(resp: dict) -> str:
    msg = resp["choices"][0]["message"]
    return (msg.get("content") or "").strip()
