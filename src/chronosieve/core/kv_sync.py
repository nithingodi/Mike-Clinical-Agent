from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .utils import estimate_tokens


KVActionType = Literal[
    "pin_or_prefill",
    "keep_current",
    "external_only",
    "evict",
    "block",
    "load_with_warning",
    "temporary_prefill",
]

KVAuthority = Literal[
    "authoritative",
    "non_authoritative",
    "blocked",
    "external",
    "temporary",
]


@dataclass
class KVAction:
    """
    A runtime-memory action proposed from ChronoSieve memory governance.

    This is intentionally backend-neutral. Later, llama.cpp / KV-direct code can
    implement these actions physically.
    """

    action_type: KVActionType
    authority: KVAuthority
    text: str | None
    reason: str
    priority: int

    source_event_id: str | None = None
    task_id: str | None = None
    archive_ref: str | None = None
    candidate_type: str | None = None
    memory_status: str | None = None
    kv_future_hint: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KVSyncPlan:
    """
    A backend-neutral runtime memory plan.

    Later phases can translate this into actual KV/cache operations:
    - pin_or_prefill -> protected prefix / cache anchor
    - keep_current -> working context
    - external_only -> archive only
    - evict -> do not load
    - block -> prevent re-entry as truth
    - load_with_warning -> allowed only with authority warning
    - temporary_prefill -> short-lived evidence bundle
    """

    session_id: str
    actions: list[KVAction]
    summary: dict[str, int]
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "actions": [a.to_dict() for a in self.actions],
            "summary": self.summary,
            "token_estimate": self.token_estimate,
        }

    def as_markdown(self) -> str:
        groups: dict[str, list[KVAction]] = {}
        for action in self.actions:
            groups.setdefault(action.action_type, []).append(action)

        preferred_order = [
            "pin_or_prefill",
            "keep_current",
            "temporary_prefill",
            "load_with_warning",
            "block",
            "external_only",
            "evict",
        ]

        lines: list[str] = []
        lines.append("# ChronoSieve KV Sync Plan")
        lines.append("")
        lines.append(f"Session: `{self.session_id}`")
        lines.append(f"Estimated active/prefill tokens: `{self.token_estimate}`")
        lines.append("")

        lines.append("## Summary")
        for key in preferred_order:
            if key in self.summary:
                lines.append(f"- {key}: {self.summary[key]}")
        lines.append("")

        for key in preferred_order:
            items = groups.get(key, [])
            if not items:
                continue

            lines.append(f"## {key}")
            for item in items:
                label = item.text or f"[external evidence: {item.archive_ref}]"
                label = self._compact(label, 500)
                lines.append(f"- **{item.authority}** · priority={item.priority} · {label}")
                if item.reason:
                    lines.append(f"  - reason: {item.reason}")
                if item.archive_ref:
                    lines.append(f"  - archive_ref: `{item.archive_ref}`")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _compact(text: str, max_chars: int) -> str:
        clean = " ".join(str(text).split())
        if len(clean) > max_chars:
            return clean[:max_chars] + "...[truncated]"
        return clean


class KVSyncPlanner:
    """
    Converts committed ChronoSieve memory events into backend-neutral runtime
    memory actions.

    This is the bridge from:
        ChronoSieve governance -> future KV-direct enforcement

    It does not touch llama.cpp or any KV cache directly.
    """

    def __init__(
        self,
        *,
        max_text_chars: int = 700,
        include_external_actions: bool = True,
    ):
        self.max_text_chars = max_text_chars
        self.include_external_actions = include_external_actions

    def plan(
        self,
        *,
        session_id: str,
        events: list[dict[str, Any]],
        carry_packet_text: str | None = None,
        correction_registry: dict[str, Any] | None = None,
        rehydrated_archive_refs: list[str] | None = None,
    ) -> KVSyncPlan:
        actions: list[KVAction] = []

        for event in events:
            action = self._action_from_event(event)
            if action is None:
                continue
            if action.action_type in {"external_only", "evict"} and not self.include_external_actions:
                continue
            actions.append(action)

        for ref in rehydrated_archive_refs or []:
            actions.append(KVAction(
                action_type="temporary_prefill",
                authority="temporary",
                text=f"Temporarily rehydrate archived evidence for current audit/recall task: {ref}",
                reason="Archive evidence was selected for current turn only.",
                priority=75,
                archive_ref=ref,
                tags=["rehydrated_archive", "temporary"],
                token_estimate=estimate_tokens(ref),
            ))

        actions.extend(self._actions_from_corrections(correction_registry or {}))

        actions = self._dedupe_actions(actions)
        actions.sort(key=lambda a: (-a.priority, a.action_type, a.text or ""))

        summary: dict[str, int] = {}
        token_estimate = 0

        for action in actions:
            summary[action.action_type] = summary.get(action.action_type, 0) + 1
            if action.action_type in {
                "pin_or_prefill",
                "keep_current",
                "temporary_prefill",
                "load_with_warning",
            }:
                token_estimate += action.token_estimate

        return KVSyncPlan(
            session_id=session_id,
            actions=actions,
            summary=summary,
            token_estimate=token_estimate,
        )

    def _action_from_event(self, event: dict[str, Any]) -> KVAction | None:
        decision = event.get("decision", {}) or {}
        candidate = event.get("candidate", {}) or {}

        status = decision.get("status")
        ctype = candidate.get("candidate_type")
        source_type = candidate.get("source_type")
        event_id = event.get("event_id")
        task_id = event.get("task_id")
        archive_ref = decision.get("archive_ref") or candidate.get("evidence_ref")
        kv_hint = decision.get("kv_future_hint")

        text = decision.get("carry_text") or candidate.get("content")
        text = self._compact_text(text)

        reason = str(decision.get("reason") or "ChronoSieve memory decision.")

        if status == "fossil":
            action_type: KVActionType = "pin_or_prefill"
            authority: KVAuthority = "authoritative"
            priority = 100

        elif status == "active":
            action_type = "keep_current"
            authority = "authoritative"
            priority = 80

        elif status == "archived":
            action_type = "external_only"
            authority = "external"
            priority = 25
            text = None

        elif status == "decay":
            action_type = "evict"
            authority = "external"
            priority = 10
            text = None

        elif status == "invalidated":
            action_type = "block"
            authority = "blocked"
            priority = 100

        elif status == "non_authoritative":
            action_type = "load_with_warning"
            authority = "non_authoritative"
            priority = 65

        else:
            action_type = "external_only"
            authority = "external"
            priority = 5
            text = None

        # Generic overrides / priority shaping.
        tags: list[str] = []

        if ctype:
            tags.append(str(ctype))
        if source_type:
            tags.append(str(source_type))
        if status:
            tags.append(str(status))

        if ctype == "caveats" and status in {"fossil", "active"}:
            action_type = "pin_or_prefill"
            authority = "authoritative"
            priority = 95
            tags.append("guardrail")

        if ctype in {"claims", "interpretations"} and status in {"active", "fossil"}:
            # Defensive planner guard. Governor should already prevent this.
            action_type = "load_with_warning"
            authority = "non_authoritative"
            priority = 70
            tags.append("planner_downgrade_claim")

        if ctype == "artifacts":
            if status not in {"invalidated", "non_authoritative"}:
                action_type = "external_only"
                authority = "external"
                priority = 30
                text = None
                tags.append("artifact_external_only")

        if kv_hint == "block_truth":
            action_type = "block" if status == "invalidated" else "load_with_warning"
            authority = "blocked" if status == "invalidated" else "non_authoritative"
            priority = max(priority, 90)
            tags.append("block_truth")

        token_estimate = estimate_tokens(text or "")

        return KVAction(
            action_type=action_type,
            authority=authority,
            text=text,
            reason=reason,
            priority=priority,
            source_event_id=event_id,
            task_id=task_id,
            archive_ref=archive_ref,
            candidate_type=ctype,
            memory_status=status,
            kv_future_hint=kv_hint,
            tags=self._dedupe(tags),
            metadata={
                "source_type": source_type,
                "governor_notes": event.get("governor_notes", []),
            },
            token_estimate=token_estimate,
        )

    def _actions_from_corrections(self, correction_registry: dict[str, Any]) -> list[KVAction]:
        actions: list[KVAction] = []

        for corr in correction_registry.get("corrections", []) or []:
            old_state = corr.get("old_state")
            new_state = corr.get("new_state")
            reason = corr.get("reason")

            text = (
                f"Correction: {old_state} -> {new_state}. "
                f"Reason: {reason}"
            )

            actions.append(KVAction(
                action_type="block",
                authority="blocked",
                text=self._compact_text(text),
                reason="Correction registry blocks superseded/unsafe interpretation.",
                priority=100,
                archive_ref=(corr.get("evidence_refs") or [None])[-1],
                tags=["correction", str(corr.get("correction_type", "correction"))],
                metadata={
                    "correction_id": corr.get("correction_id"),
                    "affected_task_ids": corr.get("affected_task_ids", []),
                },
                token_estimate=estimate_tokens(text),
            ))

        return actions

    def _compact_text(self, text: Any) -> str | None:
        if text is None:
            return None

        clean = " ".join(str(text).split())
        if not clean:
            return None

        if len(clean) > self.max_text_chars:
            return clean[: self.max_text_chars] + "...[truncated]"

        return clean

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

    def _dedupe_actions(self, actions: list[KVAction]) -> list[KVAction]:
        seen: set[tuple[str, str, str | None, str | None]] = set()
        out: list[KVAction] = []

        for action in actions:
            key = (
                action.action_type,
                action.authority,
                action.text,
                action.archive_ref,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(action)

        return out