from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import hashlib
import json

from src.chronosieve.core.runtime_context import RuntimeContextPackage, RuntimeContextSegment
from src.chronosieve.core.utils import estimate_tokens


@dataclass
class PromptCacheArtifact:
    name: str
    path: str
    segment_type: str
    sha256: str
    char_count: int
    token_estimate: int
    priority: int
    archive_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LlamaCppPromptCacheResult:
    session_id: str
    output_dir: str
    stable_prefix_path: str
    runtime_prompt_path: str
    manifest_path: str
    stable_prefix_sha256: str
    runtime_prompt_sha256: str
    stable_prefix_tokens: int
    runtime_prompt_tokens: int
    artifacts: list[PromptCacheArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "output_dir": self.output_dir,
            "stable_prefix_path": self.stable_prefix_path,
            "runtime_prompt_path": self.runtime_prompt_path,
            "manifest_path": self.manifest_path,
            "stable_prefix_sha256": self.stable_prefix_sha256,
            "runtime_prompt_sha256": self.runtime_prompt_sha256,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "runtime_prompt_tokens": self.runtime_prompt_tokens,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
        }

    def as_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Llama.cpp Prompt Cache Backend Artifacts")
        lines.append("")
        lines.append(f"Session: `{self.session_id}`")
        lines.append(f"Output dir: `{self.output_dir}`")
        lines.append(f"Stable prefix tokens: `{self.stable_prefix_tokens}`")
        lines.append(f"Runtime prompt tokens: `{self.runtime_prompt_tokens}`")
        lines.append(f"Stable prefix hash: `{self.stable_prefix_sha256[:16]}`")
        lines.append(f"Runtime prompt hash: `{self.runtime_prompt_sha256[:16]}`")
        lines.append("")
        lines.append("## Files")
        lines.append(f"- stable prefix: `{self.stable_prefix_path}`")
        lines.append(f"- runtime prompt: `{self.runtime_prompt_path}`")
        lines.append(f"- manifest: `{self.manifest_path}`")
        lines.append("")
        lines.append("## Segment artifacts")
        for artifact in self.artifacts:
            lines.append(
                f"- `{artifact.name}` · {artifact.segment_type} · "
                f"priority={artifact.priority} · tokens={artifact.token_estimate} · "
                f"hash={artifact.sha256[:12]}"
            )
        lines.append("")
        lines.append("## Cache intent")
        lines.append("- Put `stable_prefix.md` at the beginning of repeated llama.cpp prompts.")
        lines.append("- Keep section ordering stable to maximize LCP/prompt-cache reuse.")
        lines.append("- Keep external references out of active context unless rehydration selects them.")
        return "\n".join(lines).strip() + "\n"


class LlamaCppPromptCacheBackend:
    """
    Live-adjacent llama.cpp backend.

    It does not mutate KV directly.
    It writes stable, ordered prompt-cache artifacts that llama.cpp can reuse via
    its prompt-cache / slot / checkpoint machinery.

    Later:
    - stable_prefix.md can become a pinned KV prefix
    - warning_context can become tagged non-authoritative KV
    - blocked_truth_registry can become a negative truth constraint segment
    """

    STABLE_PREFIX_ORDER = [
        "blocked_truth_registry",
        "protected_prefix",
        "warning_context",
        "working_context",
    ]

    RUNTIME_ONLY_ORDER = [
        "temporary_evidence",
        "external_reference_index",
    ]

    def __init__(self, *, max_external_chars: int = 2500):
        self.max_external_chars = max_external_chars

    def build(
        self,
        *,
        package: RuntimeContextPackage,
        output_dir: str | Path,
    ) -> LlamaCppPromptCacheResult:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        segment_by_type = {s.segment_type: s for s in package.segments}
        artifacts: list[PromptCacheArtifact] = []

        # Write individual segment files.
        for segment in package.segments:
            filename = f"{segment.segment_type}.md"
            path = out / filename
            content = self._render_segment_file(segment)
            self._write_text(path, content)
            artifacts.append(self._artifact_from_file(filename, path, segment))

        stable_prefix = self._build_stable_prefix(segment_by_type)
        runtime_prompt = self._build_runtime_prompt(segment_by_type, stable_prefix)

        stable_prefix_path = out / "stable_prefix.md"
        runtime_prompt_path = out / "runtime_prompt.md"
        manifest_path = out / "backend_manifest.json"

        self._write_text(stable_prefix_path, stable_prefix)
        self._write_text(runtime_prompt_path, runtime_prompt)

        stable_hash = self._sha256(stable_prefix)
        runtime_hash = self._sha256(runtime_prompt)

        result = LlamaCppPromptCacheResult(
            session_id=package.session_id,
            output_dir=str(out),
            stable_prefix_path=str(stable_prefix_path),
            runtime_prompt_path=str(runtime_prompt_path),
            manifest_path=str(manifest_path),
            stable_prefix_sha256=stable_hash,
            runtime_prompt_sha256=runtime_hash,
            stable_prefix_tokens=estimate_tokens(stable_prefix),
            runtime_prompt_tokens=estimate_tokens(runtime_prompt),
            artifacts=artifacts,
            metadata={
                "backend": "llamacpp_prompt_cache_backend",
                "kv_direct": False,
                "prompt_cache_expected": True,
                "stable_prefix_order": self.STABLE_PREFIX_ORDER,
                "runtime_only_order": self.RUNTIME_ONLY_ORDER,
                "llama_cpp_relevant_flags": [
                    "--cache-prompt",
                    "--cache-ram",
                    "--slots",
                    "--ctx-checkpoints",
                    "--slot-prompt-similarity",
                ],
                "note": (
                    "Artifacts are designed to make ChronoSieve runtime context "
                    "stable and cacheable under llama.cpp prompt-cache behavior. "
                    "No direct KV mutation is performed."
                ),
            },
        )

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        return result

    def _build_stable_prefix(
        self,
        segment_by_type: dict[str, RuntimeContextSegment],
    ) -> str:
        lines: list[str] = []
        lines.append("# ChronoSieve Stable Runtime Prefix")
        lines.append("")
        lines.append(
            "This prefix is ordered for llama.cpp prompt-cache reuse. "
            "Blocked truth has highest priority, followed by protected memory, "
            "warnings, and deterministic working context."
        )
        lines.append("")

        for segment_type in self.STABLE_PREFIX_ORDER:
            segment = segment_by_type.get(segment_type)
            if not segment:
                continue
            lines.append(self._render_segment_for_prefix(segment))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _build_runtime_prompt(
        self,
        segment_by_type: dict[str, RuntimeContextSegment],
        stable_prefix: str,
    ) -> str:
        lines: list[str] = []
        lines.append(stable_prefix.strip())
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("# ChronoSieve Runtime-Only Context")
        lines.append("")
        lines.append(
            "The following sections are not intended to become durable KV memory "
            "unless a future ChronoSieve decision promotes them."
        )
        lines.append("")

        for segment_type in self.RUNTIME_ONLY_ORDER:
            segment = segment_by_type.get(segment_type)
            if not segment:
                continue

            content = segment.content.strip()
            if segment_type == "external_reference_index" and len(content) > self.max_external_chars:
                content = content[: self.max_external_chars] + "\n...[external references truncated]"

            lines.append(self._render_segment_for_prefix(segment, override_content=content))
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _render_segment_file(segment: RuntimeContextSegment) -> str:
        refs = ", ".join(segment.archive_refs) if segment.archive_refs else "none"
        return (
            f"# ChronoSieve Segment: {segment.segment_type}\n\n"
            f"priority: {segment.priority}\n"
            f"token_estimate: {segment.token_estimate}\n"
            f"archive_refs: {refs}\n\n"
            f"{segment.content.strip()}\n"
        )

    @staticmethod
    def _render_segment_for_prefix(
        segment: RuntimeContextSegment,
        override_content: str | None = None,
    ) -> str:
        content = override_content if override_content is not None else segment.content.strip()
        return (
            f"## {segment.segment_type}\n\n"
            f"{content.strip()}"
        )

    def _artifact_from_file(
        self,
        name: str,
        path: Path,
        segment: RuntimeContextSegment,
    ) -> PromptCacheArtifact:
        text = path.read_text(encoding="utf-8")
        return PromptCacheArtifact(
            name=name,
            path=str(path),
            segment_type=segment.segment_type,
            sha256=self._sha256(text),
            char_count=len(text),
            token_estimate=estimate_tokens(text),
            priority=segment.priority,
            archive_refs=segment.archive_refs,
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
