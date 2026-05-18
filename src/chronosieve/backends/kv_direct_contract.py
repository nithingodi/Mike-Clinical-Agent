from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Protocol
import hashlib

from src.chronosieve.core.runtime_context import RuntimeContextPackage, RuntimeContextSegment


KVDirectOperationType = Literal[
    "select_or_create_slot",
    "prefill_segment",
    "pin_segment",
    "load_with_warning",
    "register_blocked_truth",
    "register_external_reference",
    "evict_segment",
    "clear_temporary_segments",
    "inspect_runtime_state",
]

KVDirectSegmentAuthority = Literal[
    "authoritative",
    "non_authoritative",
    "blocked",
    "external",
    "temporary",
]


@dataclass
class KVDirectOperation:
    operation_type: KVDirectOperationType
    segment_id: str
    segment_type: str
    authority: KVDirectSegmentAuthority
    priority: int
    content: str | None = None
    archive_refs: list[str] = field(default_factory=list)
    token_estimate: int = 0
    pinned: bool = False
    temporary: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KVDirectPlan:
    session_id: str
    backend_name: str
    mode: str
    operations: list[KVDirectOperation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "backend_name": self.backend_name,
            "mode": self.mode,
            "operations": [op.to_dict() for op in self.operations],
            "metadata": self.metadata,
        }

    def as_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# ChronoSieve KV-Direct Contract Plan")
        lines.append("")
        lines.append(f"Session: `{self.session_id}`")
        lines.append(f"Backend: `{self.backend_name}`")
        lines.append(f"Mode: `{self.mode}`")
        lines.append("")

        for op in self.operations:
            lines.append(f"## {op.operation_type}")
            lines.append(f"- segment_id: `{op.segment_id}`")
            lines.append(f"- segment_type: `{op.segment_type}`")
            lines.append(f"- authority: `{op.authority}`")
            lines.append(f"- priority: `{op.priority}`")
            lines.append(f"- pinned: `{op.pinned}`")
            lines.append(f"- temporary: `{op.temporary}`")
            lines.append(f"- token_estimate: `{op.token_estimate}`")
            if op.archive_refs:
                lines.append(f"- archive_refs: {', '.join(f'`{r}`' for r in op.archive_refs)}")
            if op.content:
                lines.append("")
                lines.append(op.content[:1200] + ("...[truncated]" if len(op.content) > 1200 else ""))
            lines.append("")

        return "\n".join(lines).strip() + "\n"


class KVDirectRuntime(Protocol):
    """
    Future live backend interface.

    Implementations may use:
    - patched llama.cpp server endpoints
    - sidecar runtime process
    - local llama.cpp binding
    - another serving runtime with explicit prefix/KV controls
    """

    backend_name: str

    def apply_plan(self, plan: KVDirectPlan) -> dict[str, Any]:
        ...

    def inspect_runtime_state(self, session_id: str) -> dict[str, Any]:
        ...


class KVDirectContractPlanner:
    """
    Converts RuntimeContextPackage into the operation contract a true KV-direct
    backend must eventually implement.

    This is not live KV mutation. It defines the required operations.
    """

    backend_name = "kv_direct_contract_planner"

    def plan(self, package: RuntimeContextPackage) -> KVDirectPlan:
        operations: list[KVDirectOperation] = []

        operations.append(KVDirectOperation(
            operation_type="select_or_create_slot",
            segment_id=self._stable_id(package.session_id, "slot"),
            segment_type="session_slot",
            authority="authoritative",
            priority=100,
            content=None,
            token_estimate=0,
            metadata={
                "session_id": package.session_id,
                "purpose": "Select or create runtime slot for this ChronoSieve session.",
            },
        ))

        for segment in package.segments:
            operations.extend(self._ops_for_segment(package.session_id, segment))

        operations.append(KVDirectOperation(
            operation_type="inspect_runtime_state",
            segment_id=self._stable_id(package.session_id, "inspect"),
            segment_type="runtime_state",
            authority="external",
            priority=1,
            content=None,
            token_estimate=0,
            metadata={
                "purpose": "Verify which segments are pinned, active, warning-only, blocked, or external.",
            },
        ))

        return KVDirectPlan(
            session_id=package.session_id,
            backend_name=self.backend_name,
            mode="contract_only",
            operations=operations,
            metadata={
                "kv_direct_live": False,
                "note": (
                    "This is the operation contract for a future live KV-direct backend. "
                    "No llama.cpp KV cache is mutated by this planner."
                ),
            },
        )

    def _ops_for_segment(
        self,
        session_id: str,
        segment: RuntimeContextSegment,
    ) -> list[KVDirectOperation]:
        segment_id = self._stable_id(session_id, segment.segment_type, segment.content)
        ops: list[KVDirectOperation] = []

        if segment.segment_type == "protected_prefix":
            ops.append(KVDirectOperation(
                operation_type="prefill_segment",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="authoritative",
                priority=segment.priority,
                content=segment.content,
                archive_refs=segment.archive_refs,
                token_estimate=segment.token_estimate,
                pinned=True,
                metadata={"purpose": "Prefill durable protected prefix."},
            ))
            ops.append(KVDirectOperation(
                operation_type="pin_segment",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="authoritative",
                priority=segment.priority,
                archive_refs=segment.archive_refs,
                token_estimate=segment.token_estimate,
                pinned=True,
                metadata={"purpose": "Prevent durable protected prefix from eviction."},
            ))

        elif segment.segment_type == "working_context":
            ops.append(KVDirectOperation(
                operation_type="prefill_segment",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="authoritative",
                priority=segment.priority,
                content=segment.content,
                archive_refs=segment.archive_refs,
                token_estimate=segment.token_estimate,
                metadata={"purpose": "Load active deterministic working context."},
            ))

        elif segment.segment_type == "warning_context":
            ops.append(KVDirectOperation(
                operation_type="load_with_warning",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="non_authoritative",
                priority=segment.priority,
                content=segment.content,
                archive_refs=segment.archive_refs,
                token_estimate=segment.token_estimate,
                metadata={
                    "purpose": "Load useful but non-authoritative context with warning boundary.",
                    "truth_promotion_allowed": False,
                },
            ))

        elif segment.segment_type == "blocked_truth_registry":
            ops.append(KVDirectOperation(
                operation_type="register_blocked_truth",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="blocked",
                priority=segment.priority,
                content=segment.content,
                archive_refs=segment.archive_refs,
                token_estimate=segment.token_estimate,
                pinned=True,
                metadata={
                    "purpose": "Register invalidated/superseded truth that must not re-enter active context.",
                    "truth_promotion_allowed": False,
                },
            ))

        elif segment.segment_type == "temporary_evidence":
            ops.append(KVDirectOperation(
                operation_type="prefill_segment",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="temporary",
                priority=segment.priority,
                content=segment.content,
                archive_refs=segment.archive_refs,
                token_estimate=segment.token_estimate,
                temporary=True,
                metadata={"purpose": "Load temporary rehydrated evidence for current task only."},
            ))
            ops.append(KVDirectOperation(
                operation_type="clear_temporary_segments",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="temporary",
                priority=0,
                archive_refs=segment.archive_refs,
                temporary=True,
                metadata={"purpose": "Clear temporary evidence after current task unless promoted."},
            ))

        elif segment.segment_type == "external_reference_index":
            ops.append(KVDirectOperation(
                operation_type="register_external_reference",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="external",
                priority=segment.priority,
                content=None,
                archive_refs=segment.archive_refs,
                token_estimate=0,
                metadata={
                    "purpose": "Keep archive refs addressable without loading full evidence into KV.",
                    "load_into_kv": False,
                },
            ))

        else:
            ops.append(KVDirectOperation(
                operation_type="evict_segment",
                segment_id=segment_id,
                segment_type=segment.segment_type,
                authority="external",
                priority=0,
                content=None,
                archive_refs=segment.archive_refs,
                token_estimate=0,
                metadata={"purpose": "Unknown segment type should not stay in active KV."},
            ))

        return ops

    @staticmethod
    def _stable_id(*parts: str) -> str:
        raw = "::".join(str(p) for p in parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"seg_{digest}"