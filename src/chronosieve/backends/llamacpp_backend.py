from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from src.chronosieve.core.runtime_backend import RuntimeBackendPlan
from src.chronosieve.core.runtime_context import RuntimeContextPackage


LlamaCppOperationType = Literal[
    "would_pin_prefix",
    "would_prefill_working_context",
    "would_load_warning_context",
    "would_register_blocked_truth",
    "would_register_external_archive",
    "would_skip_loading",
]


@dataclass
class LlamaCppPlannedOperation:
    operation_type: LlamaCppOperationType
    segment_type: str
    content_preview: str
    priority: int
    token_estimate: int
    archive_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LlamaCppBackendPlan:
    session_id: str
    mode: str
    kv_direct: bool
    operations: list[LlamaCppPlannedOperation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "kv_direct": self.kv_direct,
            "operations": [op.to_dict() for op in self.operations],
            "metadata": self.metadata,
        }

    def as_markdown(self) -> str:
        lines = []
        lines.append("# Llama.cpp ChronoSieve Backend Plan")
        lines.append("")
        lines.append(f"Session: `{self.session_id}`")
        lines.append(f"Mode: `{self.mode}`")
        lines.append(f"KV direct: `{self.kv_direct}`")
        lines.append("")

        for op in self.operations:
            lines.append(f"## {op.operation_type}")
            lines.append(f"- segment_type: `{op.segment_type}`")
            lines.append(f"- priority: `{op.priority}`")
            lines.append(f"- token_estimate: `{op.token_estimate}`")
            if op.archive_refs:
                lines.append(f"- archive_refs: {', '.join(f'`{r}`' for r in op.archive_refs)}")
            lines.append("")
            lines.append(op.content_preview)
            lines.append("")

        return "\n".join(lines).strip() + "\n"


class LlamaCppPlannedBackend:
    """
    Planned llama.cpp backend.

    This does not touch real KV cache yet.
    It maps ChronoSieve runtime context segments into the operations a future
    llama.cpp KV-direct backend would perform.
    """

    backend_name = "llamacpp_planned_backend"

    def __init__(self, *, max_preview_chars: int = 1200):
        self.max_preview_chars = max_preview_chars

    def plan(self, package: RuntimeContextPackage) -> LlamaCppBackendPlan:
        operations: list[LlamaCppPlannedOperation] = []

        for segment in package.segments:
            operation_type = self._operation_for_segment(segment.segment_type)

            operations.append(LlamaCppPlannedOperation(
                operation_type=operation_type,
                segment_type=segment.segment_type,
                content_preview=self._compact(segment.content),
                priority=segment.priority,
                token_estimate=segment.token_estimate,
                archive_refs=segment.archive_refs,
                metadata={
                    **segment.metadata,
                    "source_action_types": segment.source_action_types,
                },
            ))

        operations.sort(key=lambda op: -op.priority)

        return LlamaCppBackendPlan(
            session_id=package.session_id,
            mode="planned_only",
            kv_direct=False,
            operations=operations,
            metadata={
                "note": (
                    "This is the planned mapping from ChronoSieve runtime context "
                    "to future llama.cpp KV/cache operations. It does not mutate KV yet."
                )
            },
        )

    @staticmethod
    def _operation_for_segment(segment_type: str) -> LlamaCppOperationType:
        if segment_type == "protected_prefix":
            return "would_pin_prefix"
        if segment_type == "working_context":
            return "would_prefill_working_context"
        if segment_type == "warning_context":
            return "would_load_warning_context"
        if segment_type == "blocked_truth_registry":
            return "would_register_blocked_truth"
        if segment_type == "external_reference_index":
            return "would_register_external_archive"
        if segment_type == "temporary_evidence":
            return "would_prefill_working_context"
        return "would_skip_loading"

    def _compact(self, text: str) -> str:
        clean = str(text).strip()
        if len(clean) > self.max_preview_chars:
            return clean[: self.max_preview_chars] + "\n...[truncated]"
        return clean