from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, Literal

from .runtime_context import RuntimeContextPackage, RuntimeContextSegment


BackendActionType = Literal[
    "prepare_protected_prefix",
    "prepare_working_context",
    "prepare_warning_context",
    "prepare_temporary_evidence",
    "prepare_block_registry",
    "register_external_refs",
]


@dataclass
class RuntimeBackendAction:
    action_type: BackendActionType
    segment_type: str
    content: str
    priority: int
    token_estimate: int
    archive_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeBackendPlan:
    session_id: str
    backend_name: str
    actions: list[RuntimeBackendAction]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "backend_name": self.backend_name,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata,
        }


class RuntimeBackend(Protocol):
    """
    Backend contract.

    A backend may be:
    - prompt-only OpenAI-compatible API
    - llama.cpp prompt-cache backend
    - future KV-direct backend
    - vLLM-style prefix cache backend
    - custom local runtime
    """

    backend_name: str

    def plan(self, package: RuntimeContextPackage) -> RuntimeBackendPlan:
        ...

    def apply(self, plan: RuntimeBackendPlan) -> dict[str, Any]:
        ...


class PromptOnlyRuntimeBackend:
    """
    Safe baseline backend.

    It does not mutate KV directly.
    It converts runtime context segments into ordered prompt/context sections.

    This is useful today because the current Mike path uses an OpenAI-compatible
    ChatOpenAI interface, not a direct KV handle.
    """

    backend_name = "prompt_only_runtime_backend"

    def plan(self, package: RuntimeContextPackage) -> RuntimeBackendPlan:
        actions: list[RuntimeBackendAction] = []

        for segment in package.segments:
            action_type = self._action_type_for_segment(segment)
            actions.append(RuntimeBackendAction(
                action_type=action_type,
                segment_type=segment.segment_type,
                content=segment.content,
                priority=segment.priority,
                token_estimate=segment.token_estimate,
                archive_refs=segment.archive_refs,
                metadata=segment.metadata,
            ))

        actions.sort(key=lambda a: -a.priority)

        return RuntimeBackendPlan(
            session_id=package.session_id,
            backend_name=self.backend_name,
            actions=actions,
            metadata={
                "mode": "prompt_context_only",
                "kv_direct": False,
                "note": (
                    "This backend prepares ordered runtime context but does not "
                    "mutate KV cache directly."
                ),
            },
        )

    def apply(self, plan: RuntimeBackendPlan) -> dict[str, Any]:
        context_blocks: list[str] = []

        for action in plan.actions:
            context_blocks.append(
                f"## {action.segment_type}\n\n{action.content.strip()}"
            )

        runtime_context = "\n\n---\n\n".join(context_blocks).strip()

        return {
            "backend_name": self.backend_name,
            "session_id": plan.session_id,
            "kv_direct": False,
            "runtime_context": runtime_context,
            "action_count": len(plan.actions),
            "metadata": plan.metadata,
        }

    @staticmethod
    def _action_type_for_segment(segment: RuntimeContextSegment) -> BackendActionType:
        mapping: dict[str, BackendActionType] = {
            "protected_prefix": "prepare_protected_prefix",
            "working_context": "prepare_working_context",
            "warning_context": "prepare_warning_context",
            "temporary_evidence": "prepare_temporary_evidence",
            "blocked_truth_registry": "prepare_block_registry",
            "external_reference_index": "register_external_refs",
        }
        return mapping.get(segment.segment_type, "prepare_working_context")