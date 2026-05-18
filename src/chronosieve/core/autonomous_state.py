from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from src.chronosieve.core.schemas import utc_now


StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass
class AutonomousStepRecord:
    """
    One bounded autonomous execution step.

    A broad task may produce hundreds or thousands of these steps. Each step
    should stay small enough for a bounded model/tool call.
    """

    step_id: str
    resource_id: str
    action: str
    status: StepStatus
    reason: str = ""
    error: str | None = None
    memory_event_count: int = 0
    archive_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutonomousTaskState:
    """
    Persistent progress state for one broad long-horizon goal.

    This is intentionally domain-neutral. The resources may be DICOM files,
    code files, legal docs, sports logs, support tickets, or anything else an
    adapter exposes.
    """

    run_id: str
    session_id: str
    goal: str
    resources_total: int
    resources_pending: list[str] = field(default_factory=list)
    resources_completed: list[str] = field(default_factory=list)
    resources_failed: list[str] = field(default_factory=list)
    resources_skipped: list[str] = field(default_factory=list)
    step_records: list[AutonomousStepRecord] = field(default_factory=list)
    aggregate_notes: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compact_progress(self, *, max_recent_steps: int = 12) -> dict[str, Any]:
        """
        Small progress view safe to show the planner.

        The planner does not need the entire history of every prior step. It
        needs counts, recent step summaries, unresolved questions, and a few
        aggregate notes.
        """
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "resources_total": self.resources_total,
            "completed_count": len(self.resources_completed),
            "pending_count": len(self.resources_pending),
            "failed_count": len(self.resources_failed),
            "skipped_count": len(self.resources_skipped),
            "recent_steps": [s.to_dict() for s in self.step_records[-max_recent_steps:]],
            "aggregate_notes": self.aggregate_notes[-20:],
            "unresolved_questions": self.unresolved_questions[-20:],
            "metadata": self.metadata,
        }

    def mark_completed(self, resource_id: str) -> None:
        if resource_id in self.resources_pending:
            self.resources_pending.remove(resource_id)
        if resource_id not in self.resources_completed:
            self.resources_completed.append(resource_id)
        self.updated_at = utc_now()

    def mark_failed(self, resource_id: str) -> None:
        if resource_id in self.resources_pending:
            self.resources_pending.remove(resource_id)
        if resource_id not in self.resources_failed:
            self.resources_failed.append(resource_id)
        self.updated_at = utc_now()

    def mark_skipped(self, resource_id: str) -> None:
        if resource_id in self.resources_pending:
            self.resources_pending.remove(resource_id)
        if resource_id not in self.resources_skipped:
            self.resources_skipped.append(resource_id)
        self.updated_at = utc_now()
