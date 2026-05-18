from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .kv_sync import KVSyncPlan, KVAction
from .utils import estimate_tokens


RuntimeSegmentType = Literal[
    "protected_prefix",
    "working_context",
    "warning_context",
    "temporary_evidence",
    "blocked_truth_registry",
    "external_reference_index",
]


@dataclass
class RuntimeContextSegment:
    segment_type: RuntimeSegmentType
    content: str
    source_action_types: list[str] = field(default_factory=list)
    archive_refs: list[str] = field(default_factory=list)
    priority: int = 0
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeContextPackage:
    session_id: str
    segments: list[RuntimeContextSegment]
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "segments": [s.to_dict() for s in self.segments],
            "token_estimate": self.token_estimate,
        }

    def as_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# ChronoSieve Runtime Context Package")
        lines.append("")
        lines.append(f"Session: `{self.session_id}`")
        lines.append(f"Estimated tokens: `{self.token_estimate}`")
        lines.append("")

        for segment in self.segments:
            lines.append(f"## {segment.segment_type}")
            lines.append(f"priority: `{segment.priority}`")
            if segment.archive_refs:
                refs = ", ".join(f"`{r}`" for r in segment.archive_refs)
                lines.append(f"archive_refs: {refs}")
            lines.append("")
            lines.append(segment.content.strip())
            lines.append("")

        return "\n".join(lines).strip() + "\n"


class RuntimeContextBuilder:
    """
    Converts a backend-neutral KVSyncPlan into ordered runtime context segments.

    This is the direct predecessor to KV-direct backend work:
    - protected_prefix can become pinned/preserved KV prefix
    - working_context can become current active context
    - warning_context can become authority guardrails
    - temporary_evidence can be prefetched/rehydrated for one turn
    - blocked_truth_registry can be used to prevent false memory re-entry
    """

    def __init__(
        self,
        *,
        max_segment_chars: int = 3000,
    ):
        self.max_segment_chars = max_segment_chars

    def build(self, plan: KVSyncPlan) -> RuntimeContextPackage:
        segments: list[RuntimeContextSegment] = []

        protected = self._segment_from_actions(
            segment_type="protected_prefix",
            title="Protected / pinned memory",
            actions=[
                a for a in plan.actions
                if a.action_type == "pin_or_prefill"
            ],
            instruction=(
                "These items are allowed to be pinned or prefetched as durable context. "
                "Treat them as stable guardrails or validated continuity anchors."
            ),
        )
        if protected:
            segments.append(protected)

        working = self._segment_from_actions(
            segment_type="working_context",
            title="Active working context",
            actions=[
                a for a in plan.actions
                if a.action_type == "keep_current"
            ],
            instruction=(
                "These items are active deterministic working context for near-future turns."
            ),
        )
        if working:
            segments.append(working)

        warning = self._segment_from_actions(
            segment_type="warning_context",
            title="Non-authoritative memory",
            actions=[
                a for a in plan.actions
                if a.action_type == "load_with_warning"
            ],
            instruction=(
                "These items may be useful but must not be treated as authoritative truth. "
                "Use them only with explicit authority warnings or revalidation."
            ),
        )
        if warning:
            segments.append(warning)

        temporary = self._segment_from_actions(
            segment_type="temporary_evidence",
            title="Temporary rehydrated evidence",
            actions=[
                a for a in plan.actions
                if a.action_type == "temporary_prefill"
            ],
            instruction=(
                "These items are temporary evidence for the current task only. "
                "They should decay after the turn unless promoted by governance."
            ),
        )
        if temporary:
            segments.append(temporary)

        blocked = self._segment_from_actions(
            segment_type="blocked_truth_registry",
            title="Blocked / invalidated truth registry",
            actions=[
                a for a in plan.actions
                if a.action_type == "block"
            ],
            instruction=(
                "These items must not re-enter active context as truth. "
                "They represent invalidated, superseded, or unsafe interpretations."
            ),
        )
        if blocked:
            segments.append(blocked)

        external = self._segment_from_actions(
            segment_type="external_reference_index",
            title="External-only archive references",
            actions=[
                a for a in plan.actions
                if a.action_type in {"external_only", "evict"}
            ],
            instruction=(
                "These references remain externally addressable but should not be loaded "
                "into active context unless a future rehydration policy selects them."
            ),
            include_textless=True,
        )
        if external:
            segments.append(external)

        token_estimate = sum(s.token_estimate for s in segments)

        return RuntimeContextPackage(
            session_id=plan.session_id,
            segments=segments,
            token_estimate=token_estimate,
        )

    def _segment_from_actions(
        self,
        *,
        segment_type: RuntimeSegmentType,
        title: str,
        actions: list[KVAction],
        instruction: str,
        include_textless: bool = False,
    ) -> RuntimeContextSegment | None:
        if not actions:
            return None

        lines: list[str] = []
        archive_refs: list[str] = []
        source_action_types: list[str] = []

        lines.append(f"# {title}")
        lines.append("")
        lines.append(instruction)
        lines.append("")

        for action in sorted(actions, key=lambda a: (-a.priority, a.text or "")):
            source_action_types.append(action.action_type)
            if action.archive_ref:
                archive_refs.append(action.archive_ref)

            if action.text:
                lines.append(f"- {action.text}")
            elif include_textless and action.archive_ref:
                lines.append(f"- External reference: {action.archive_ref}")
            else:
                continue

            if action.authority != "authoritative":
                lines.append(f"  - authority: {action.authority}")
            if action.reason:
                lines.append(f"  - reason: {action.reason}")

        content = "\n".join(lines).strip()
        if len(content) > self.max_segment_chars:
            content = content[: self.max_segment_chars] + "\n...[runtime segment truncated]"

        return RuntimeContextSegment(
            segment_type=segment_type,
            content=content,
            source_action_types=self._dedupe(source_action_types),
            archive_refs=self._dedupe(archive_refs),
            priority=max((a.priority for a in actions), default=0),
            token_estimate=estimate_tokens(content),
            metadata={"action_count": len(actions)},
        )

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            clean = " ".join(str(item).split())
            if clean and clean not in seen:
                seen.add(clean)
                out.append(clean)
        return out