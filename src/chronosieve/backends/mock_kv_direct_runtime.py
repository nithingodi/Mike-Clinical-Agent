from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from src.chronosieve.backends.kv_direct_contract import (
    KVDirectPlan,
    KVDirectOperation,
)


@dataclass
class MockKVSlot:
    session_id: str
    slot_id: str
    pinned_segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    warning_segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocked_truth: dict[str, dict[str, Any]] = field(default_factory=dict)
    external_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    temporary_segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    evicted_segments: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MockKVDirectRuntime:
    """
    In-memory simulation of a future KV-direct runtime.

    This does not call llama.cpp.
    It validates the lifecycle implied by KVDirectPlan:
    - select/create slot
    - prefill active/protected segments
    - pin protected memory
    - load non-authoritative warning memory
    - register blocked truth
    - keep archives external
    - clear temporary memory
    """

    backend_name = "mock_kv_direct_runtime"

    def __init__(self):
        self.slots: dict[str, MockKVSlot] = {}

    def apply_plan(self, plan: KVDirectPlan) -> dict[str, Any]:
        slot = self._get_or_create_slot(plan.session_id)

        applied: list[dict[str, Any]] = []

        for op in plan.operations:
            result = self._apply_operation(slot, op)
            applied.append(result)

        return {
            "backend_name": self.backend_name,
            "session_id": plan.session_id,
            "mode": "mock_runtime",
            "operation_count": len(plan.operations),
            "applied": applied,
            "runtime_state": slot.to_dict(),
        }

    def inspect_runtime_state(self, session_id: str) -> dict[str, Any]:
        slot = self._get_or_create_slot(session_id)
        return slot.to_dict()

    def _get_or_create_slot(self, session_id: str) -> MockKVSlot:
        if session_id not in self.slots:
            self.slots[session_id] = MockKVSlot(
                session_id=session_id,
                slot_id=f"mock_slot::{session_id}",
            )
        return self.slots[session_id]

    def _apply_operation(
        self,
        slot: MockKVSlot,
        op: KVDirectOperation,
    ) -> dict[str, Any]:
        if op.operation_type == "select_or_create_slot":
            return self._result(op, "slot_selected")

        if op.operation_type == "prefill_segment":
            target = slot.temporary_segments if op.temporary else slot.active_segments
            target[op.segment_id] = self._segment_record(op)

            if op.pinned:
                slot.pinned_segments[op.segment_id] = self._segment_record(op)

            return self._result(op, "segment_prefilled")

        if op.operation_type == "pin_segment":
            slot.pinned_segments[op.segment_id] = self._segment_record(op)
            return self._result(op, "segment_pinned")

        if op.operation_type == "load_with_warning":
            slot.warning_segments[op.segment_id] = self._segment_record(op)
            return self._result(op, "warning_segment_loaded")

        if op.operation_type == "register_blocked_truth":
            slot.blocked_truth[op.segment_id] = self._segment_record(op)
            return self._result(op, "blocked_truth_registered")

        if op.operation_type == "register_external_reference":
            slot.external_refs[op.segment_id] = self._segment_record(op)
            return self._result(op, "external_reference_registered")

        if op.operation_type == "evict_segment":
            removed = (
                slot.active_segments.pop(op.segment_id, None)
                or slot.warning_segments.pop(op.segment_id, None)
                or slot.temporary_segments.pop(op.segment_id, None)
                or slot.pinned_segments.pop(op.segment_id, None)
            )
            slot.evicted_segments[op.segment_id] = removed or self._segment_record(op)
            return self._result(op, "segment_evicted")

        if op.operation_type == "clear_temporary_segments":
            if op.segment_id in slot.temporary_segments:
                removed = slot.temporary_segments.pop(op.segment_id)
                slot.evicted_segments[op.segment_id] = removed
            return self._result(op, "temporary_segments_cleared")

        if op.operation_type == "inspect_runtime_state":
            return self._result(op, "runtime_state_inspected")

        return self._result(op, "unknown_operation_ignored")

    @staticmethod
    def _segment_record(op: KVDirectOperation) -> dict[str, Any]:
        return {
            "segment_id": op.segment_id,
            "segment_type": op.segment_type,
            "authority": op.authority,
            "priority": op.priority,
            "content": op.content,
            "archive_refs": op.archive_refs,
            "token_estimate": op.token_estimate,
            "pinned": op.pinned,
            "temporary": op.temporary,
            "metadata": op.metadata,
        }

    @staticmethod
    def _result(op: KVDirectOperation, status: str) -> dict[str, Any]:
        return {
            "operation_type": op.operation_type,
            "segment_id": op.segment_id,
            "segment_type": op.segment_type,
            "status": status,
        }