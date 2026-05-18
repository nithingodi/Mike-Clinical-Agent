from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json
import time
import urllib.request

from src.chronosieve.core.runtime_context import RuntimeContextPackage
from src.chronosieve.backends.llamacpp_prompt_cache_backend import (
    LlamaCppPromptCacheBackend,
)


@dataclass
class LlamaCppPromptCacheCallResult:
    session_id: str
    base_url: str
    model: str
    stable_prefix_sha256: str
    stable_prefix_tokens: int
    user_prompt: str
    latency_seconds: float

    content: str
    reasoning_content_preview: str
    finish_reason: str | None

    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    cache_n: int | None
    prompt_n: int | None
    predicted_n: int | None

    cache_ratio: float | None
    raw_response_preview: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LlamaCppPromptCacheLiveBackend:
    """
    Live llama.cpp prompt-cache backend.

    This backend still does not mutate KV directly.
    It:
    - builds stable ChronoSieve prefix artifacts
    - sends requests to llama.cpp OpenAI-compatible endpoint
    - records cache metrics from llama.cpp response

    This is the measurable bridge before true KV-direct.
    """

    backend_name = "llamacpp_prompt_cache_live_backend"

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "gemma-4-31B-it-Q4_K_M.gguf",
        api_key: str = "local-dummy-key",
        timeout_seconds: int = 240,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.artifact_backend = LlamaCppPromptCacheBackend()

    def prepare_artifacts(
        self,
        *,
        package: RuntimeContextPackage,
        output_dir: str | Path,
    ):
        return self.artifact_backend.build(
            package=package,
            output_dir=output_dir,
        )

    def call_with_package(
        self,
        *,
        package: RuntimeContextPackage,
        output_dir: str | Path,
        user_prompt: str,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> LlamaCppPromptCacheCallResult:
        artifact_result = self.prepare_artifacts(
            package=package,
            output_dir=output_dir,
        )

        stable_prefix = Path(artifact_result.stable_prefix_path).read_text(
            encoding="utf-8"
        )

        return self.call_with_prefix(
            session_id=package.session_id,
            stable_prefix=stable_prefix,
            stable_prefix_sha256=artifact_result.stable_prefix_sha256,
            stable_prefix_tokens=artifact_result.stable_prefix_tokens,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def call_with_prefix(
        self,
        *,
        session_id: str,
        stable_prefix: str,
        stable_prefix_sha256: str,
        stable_prefix_tokens: int,
        user_prompt: str,
        max_tokens: int = 700,
        temperature: float = 0.0,
    ) -> LlamaCppPromptCacheCallResult:
        url = self.base_url + "/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": stable_prefix,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        started = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            response = json.loads(resp.read().decode("utf-8"))
        latency = round(time.time() - started, 3)

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}

        usage = response.get("usage", {}) or {}
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        timings = response.get("timings", {}) or {}

        prompt_tokens = usage.get("prompt_tokens")
        cached_tokens = prompt_details.get("cached_tokens")

        cache_ratio = None
        if prompt_tokens:
            cache_ratio = round((cached_tokens or 0) / prompt_tokens, 4)

        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""

        return LlamaCppPromptCacheCallResult(
            session_id=session_id,
            base_url=self.base_url,
            model=self.model,
            stable_prefix_sha256=stable_prefix_sha256,
            stable_prefix_tokens=stable_prefix_tokens,
            user_prompt=user_prompt,
            latency_seconds=latency,
            content=content,
            reasoning_content_preview=reasoning[:1200],
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_n=timings.get("cache_n"),
            prompt_n=timings.get("prompt_n"),
            predicted_n=timings.get("predicted_n"),
            cache_ratio=cache_ratio,
            raw_response_preview=json.dumps(response, ensure_ascii=False)[:4000],
            metadata={
                "backend": self.backend_name,
                "kv_direct": False,
                "prompt_cache_expected": True,
            },
        )