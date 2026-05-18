from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Literal
import uuid


MemoryStatus = Literal[
    "fossil",
    "active",
    "archived",
    "decay",
    "invalidated",
    "non_authoritative",
]

CandidateType = Literal[
    "task_intent",
    "tools_used",
    "tool_outputs",
    "artifacts",
    "observations",
    "interpretations",
    "claims",
    "caveats",
    "corrections",
    "uncertainties",
    "learnings",
]

SourceType = Literal[
    "user_request",
    "task_answer",
    "tool_trace",
    "deterministic_extractor",
    "sieve_brain",
    "python_governor",
    "prior_memory",
]

CorrectionType = Literal[
    "factual_correction",
    "reference_frame_change",
    "task_scope_change",
    "preference_correction",
    "invalidation",
    "supersession",
    "conceptual_correction",
    "architectural_correction",
]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ContextBudget:
    ctx_size: int = 32768
    safe_budget: int = 26000
    carry_soft_limit: int = 4000
    carry_hard_limit: int = 6000
    hard_stop_budget: int = 29000
    emergency_buffer: int = 3000


@dataclass
class MemoryCandidate:
    """
    A generic thing ChronoSieve may remember, archive, decay, invalidate,
    or rehydrate later.

    This object must stay domain-neutral.
    Mike/DICOM details belong in metadata, not top-level schema fields.
    """
    candidate_id: str
    candidate_type: CandidateType
    content: str
    source_type: SourceType
    evidence_ref: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Feature hints. Gemma may fill these; Python may validate/override.
    evidence_strength: int | None = None   # 0-3
    future_utility: int | None = None      # 0-3
    audit_value: int | None = None         # 0-3
    correction_value: int | None = None    # 0-3
    risk_if_wrong: int | None = None       # 0-3
    token_cost: int | None = None          # rough estimate
    redundancy: int | None = None          # 0-3

    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SieveDecision:
    """
    Brain 2 recommends this. Python Governor validates it.
    """
    decision_id: str
    candidate_id: str
    status: MemoryStatus
    reason: str
    carry_text: str | None = None
    archive_ref: str | None = None
    rehydration_triggers: list[str] = field(default_factory=list)
    kv_future_hint: str | None = None
    confidence: float | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEvent:
    """
    Final committed memory event after Gemma recommendation + Python validation.
    """
    event_id: str
    task_id: str
    candidate: MemoryCandidate
    decision: SieveDecision
    governor_notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceArchiveEntry:
    """
    Full raw evidence from a turn/task. This is audit storage, not active context.
    """
    archive_id: str
    session_id: str
    task_id: str
    user_request: str
    final_answer: str
    trace: list[dict[str, Any]]
    image_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskLedgerEntry:
    """
    Compact task-level state. This is not raw trace.
    """
    task_id: str
    session_id: str
    user_request: str
    task_summary: str
    status: str
    evidence_refs: list[str] = field(default_factory=list)
    active_findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    non_authoritative_items: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionRecord:
    correction_id: str
    session_id: str
    correction_type: CorrectionType
    old_state: str | None
    new_state: str
    reason: str
    affected_task_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CarryPacket:
    """
    Clean briefing note for Brain 1.
    This replaces full raw chat history as the main continuity mechanism.
    """
    session_id: str
    content_md: str
    token_estimate: int
    included_event_ids: list[str] = field(default_factory=list)
    included_archive_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
