from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.chronosieve.core.agent_anatomy import AgentAnatomy
from src.chronosieve.core.autonomous_state import (
    AutonomousTaskState,
    AutonomousStepRecord,
)
from src.chronosieve.core.resource_adapter import (
    ResourceAdapter,
    ResourceRef,
    ObservationPacket,
)
from src.chronosieve.core.schemas import (
    EvidenceArchiveEntry,
    MemoryEvent,
    TaskLedgerEntry,
    new_id,
    utc_now,
)
from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.sieve_worker import GemmaSieveWorker
from src.chronosieve.core.policy_governor import PolicyGovernor
from src.chronosieve.core.carry_packet import CarryPacketBuilder
from src.chronosieve.core.utils import extract_json_object, write_json


@dataclass
class PlannerDecision:
    """
    Model-proposed next bounded action.

    The runtime may override this for safety, budget, or availability.
    """

    action: str
    resource_id: str | None
    reason: str
    instruction: str
    expected_value: str = ""
    should_stop: bool = False
    stop_reason: str | None = None
    context_to_carry_next: list[str] = field(default_factory=list)
    budget_estimate: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutonomousRunResult:
    run_id: str
    session_id: str
    goal: str
    resources_total: int
    steps_attempted: int
    completed_count: int
    failed_count: int
    skipped_count: int
    memory_event_count: int
    carry_packet_tokens: int
    storage_dir: str
    state_path: str
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutonomousChronoSieveExecutor:
    """
    Generic long-horizon autonomous executor.

    This is the missing layer between a broad user goal and ChronoSieve memory
    governance. It is intentionally domain-neutral:

        goal -> resources -> one bounded step -> observation -> candidates
        -> Sieve -> Governor -> archive/carry -> next bounded step

    v0 intentionally uses prompt-level memory governance. It does not require
    KV-direct v1.
    """

    def __init__(
        self,
        *,
        session_id: str,
        llm: Any,
        storage: ChronoSieveStorage,
        adapter: ResourceAdapter,
        anatomy: AgentAnatomy | None = None,
        sieve_worker: GemmaSieveWorker | None = None,
        governor: PolicyGovernor | None = None,
        carry_builder: CarryPacketBuilder | None = None,
        output_root: str | Path = "storage/chronosieve_autonomous_runs",
    ):
        self.session_id = session_id
        self.llm = llm
        self.storage = storage
        self.adapter = adapter
        self.anatomy = anatomy or AgentAnatomy()
        self.sieve_worker = sieve_worker or GemmaSieveWorker(llm=llm)
        self.governor = governor or PolicyGovernor()
        self.carry_builder = carry_builder or CarryPacketBuilder()
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        goal: str,
        max_steps: int = 25,
        planner_mode: str = "model",  # model | sequential
        compact_every: int = 10,
    ) -> AutonomousRunResult:
        run_id = new_id("autorun")
        resources = self.adapter.list_resources(goal=goal)
        resource_by_id = {r.resource_id: r for r in resources}

        state = AutonomousTaskState(
            run_id=run_id,
            session_id=self.session_id,
            goal=goal,
            resources_total=len(resources),
            resources_pending=[r.resource_id for r in resources],
            metadata={
                "adapter_name": getattr(self.adapter, "adapter_name", "unknown_adapter"),
                "planner_mode": planner_mode,
                "compact_every": compact_every,
            },
        )

        stopped_reason = "max_steps_reached"
        total_events = 0

        for step_index in range(1, max_steps + 1):
            if not state.resources_pending:
                stopped_reason = "all_resources_processed"
                break

            carry_packet = self.storage.read_carry_packet_text()

            decision = self._choose_next_action(
                goal=goal,
                state=state,
                resources=resources,
                carry_packet=carry_packet,
                planner_mode=planner_mode,
            )

            if decision.should_stop:
                stopped_reason = decision.stop_reason or "planner_requested_stop"
                break

            resource_id = decision.resource_id
            if (
                not resource_id
                or resource_id not in resource_by_id
                or resource_id not in state.resources_pending
            ):
                resource_id = state.resources_pending[0]
                decision.resource_id = resource_id
                decision.reason += (" Runtime override: planner did not choose a valid pending resource.")

            resource = resource_by_id[resource_id]
            step_id = new_id("autostep")
            archive_id = new_id("archive")

            step_record = AutonomousStepRecord(
                step_id=step_id,
                resource_id=resource.resource_id,
                action=decision.action or "inspect_resource",
                status="running",
                reason=decision.reason,
                archive_id=archive_id,
                metadata={
                    "planner_decision": decision.to_dict(),
                    "step_index": step_index,
                },
            )
            state.step_records.append(step_record)

            try:
                observation = self.adapter.inspect_resource(
                    resource=resource,
                    instruction=decision.instruction or goal,
                )

                self._archive_observation(
                    archive_id=archive_id,
                    task_id=step_id,
                    goal=goal,
                    resource=resource,
                    observation=observation,
                )

                candidates = self.adapter.candidates_from_observation(
                    goal=goal,
                    task_id=step_id,
                    archive_id=archive_id,
                    observation=observation,
                    resource=resource,
                )

                sieve_result = self.sieve_worker.decide(
                    user_request=goal,
                    final_answer=observation.summary,
                    candidates=candidates,
                    prior_carry_packet=carry_packet,
                    recent_ledger=self.storage.read_recent_ledger(limit=12),
                    alias_memory=self.storage.read_alias_memory(),
                    correction_registry=self.storage.read_correction_registry(),
                )

                events = self.governor.validate(
                    task_id=step_id,
                    candidates=candidates,
                    decisions=sieve_result["decisions"],
                )

                for event in events:
                    self.storage.append_memory_event(event)

                self.storage.append_ledger(TaskLedgerEntry(
                    task_id=step_id,
                    session_id=self.session_id,
                    user_request=f"Autonomous step for goal: {goal}",
                    task_summary=str(sieve_result.get("task_summary") or observation.summary),
                    status="completed" if not observation.error else "completed_with_observation_error",
                    evidence_refs=[archive_id],
                    active_findings=self._texts_for_status(events, {"active", "fossil"}),
                    caveats=self._texts_for_type(events, {"caveats"}),
                    uncertainties=self._texts_for_type(events, {"uncertainties"}),
                    non_authoritative_items=self._texts_for_status(events, {"non_authoritative"}),
                    metadata={
                        "run_id": run_id,
                        "resource_id": resource.resource_id,
                        "resource_uri": resource.uri,
                        "planner_decision": decision.to_dict(),
                        "observation_error": observation.error,
                        "sieve_parse_success": sieve_result.get("parse_success"),
                        "sieve_fallback_used": sieve_result.get("fallback_used"),
                    },
                ))

                total_events += len(events)
                step_record.status = "completed"
                step_record.completed_at = utc_now()
                step_record.memory_event_count = len(events)
                state.mark_completed(resource.resource_id)

                if observation.warnings:
                    state.unresolved_questions.extend(observation.warnings[:5])

                if compact_every > 0 and step_index % compact_every == 0:
                    state.aggregate_notes.append(
                        f"Processed {len(state.resources_completed)}/{state.resources_total} resources as of step {step_index}."
                    )

                self._rebuild_carry_packet()
                self._write_state(state)

            except Exception as exc:
                step_record.status = "failed"
                step_record.error = str(exc)
                step_record.completed_at = utc_now()
                state.mark_failed(resource.resource_id)
                self._write_state(state)

        packet = self._rebuild_carry_packet()
        state_path = self._write_state(state)

        return AutonomousRunResult(
            run_id=run_id,
            session_id=self.session_id,
            goal=goal,
            resources_total=state.resources_total,
            steps_attempted=len(state.step_records),
            completed_count=len(state.resources_completed),
            failed_count=len(state.resources_failed),
            skipped_count=len(state.resources_skipped),
            memory_event_count=total_events,
            carry_packet_tokens=packet.token_estimate,
            storage_dir=str(self.storage.session_dir),
            state_path=str(state_path),
            stopped_reason=stopped_reason,
            metadata={
                "adapter_name": getattr(self.adapter, "adapter_name", "unknown_adapter"),
                "planner_mode": planner_mode,
                "kv_direct_required": False,
            },
        )

    def _choose_next_action(
        self,
        *,
        goal: str,
        state: AutonomousTaskState,
        resources: list[ResourceRef],
        carry_packet: str,
        planner_mode: str,
    ) -> PlannerDecision:
        if planner_mode == "sequential":
            rid = state.resources_pending[0] if state.resources_pending else None
            return PlannerDecision(
                action="inspect_resource",
                resource_id=rid,
                reason="Sequential planner selected the next pending resource.",
                instruction=(
                    "Inspect this resource for evidence relevant to the goal. "
                    "Extract generic observations, uncertainty, caveats, and source refs."
                ),
                expected_value="Generic evidence observation for ChronoSieve governance.",
                raw={"planner_mode": "sequential"},
            )

        prompt_payload = {
            "role": "autonomous_task_planner",
            "goal": goal,
            "agent_anatomy": self.anatomy.compact_briefing(),
            "adapter_capabilities": self.adapter.describe_capabilities(),
            "progress_state": state.compact_progress(),
            "governed_carry_packet_preview": carry_packet[:3000],
            "available_pending_resources_preview": [
                r.to_dict() for r in self._pending_resource_preview(state, resources, limit=20)
            ],
            "required_output_shape": {
                "action": "inspect_resource|stop",
                "resource_id": "one pending resource_id or null",
                "reason": "why this bounded next step is best",
                "instruction": "specific instruction for inspecting exactly this resource",
                "expected_value": "what useful evidence may be gained",
                "should_stop": False,
                "stop_reason": None,
                "context_to_carry_next": ["brief notes only"],
                "budget_estimate": {"input_tokens": 0, "output_tokens": 0, "tool_cost": "low|medium|high"},
            },
        }

        system = SystemMessage(content=self._planner_system_prompt())
        human = HumanMessage(content=json.dumps(prompt_payload, indent=2, ensure_ascii=False, default=str))

        try:
            response = self.llm.invoke([system, human])
            raw_text = getattr(response, "content", str(response))
            parsed = extract_json_object(raw_text)
        except Exception as exc:
            parsed = {
                "action": "inspect_resource",
                "resource_id": state.resources_pending[0] if state.resources_pending else None,
                "reason": f"Planner failed; runtime selected next pending resource. Error: {exc}",
                "instruction": "Inspect this resource for generic evidence relevant to the goal.",
                "expected_value": "Fallback evidence observation.",
                "should_stop": False,
                "stop_reason": None,
                "context_to_carry_next": [],
                "budget_estimate": {},
            }

        action = str(parsed.get("action") or "inspect_resource")
        should_stop = bool(parsed.get("should_stop")) or action == "stop"

        return PlannerDecision(
            action=action,
            resource_id=parsed.get("resource_id"),
            reason=str(parsed.get("reason") or "Planner selected next bounded action."),
            instruction=str(parsed.get("instruction") or "Inspect this resource for generic evidence relevant to the goal."),
            expected_value=str(parsed.get("expected_value") or ""),
            should_stop=should_stop,
            stop_reason=parsed.get("stop_reason"),
            context_to_carry_next=list(parsed.get("context_to_carry_next") or []),
            budget_estimate=dict(parsed.get("budget_estimate") or {}),
            raw=parsed,
        )

    @staticmethod
    def _planner_system_prompt() -> str:
        return """
You are the ChronoSieve Autonomous Task Planner.

You do not solve the whole task in one response.
You choose exactly one bounded next action for a long-horizon task.

Core rules:
- Do not ask to load all resources into context.
- Do not append full history indefinitely.
- Choose one pending resource/action at a time.
- Prefer actions that reduce uncertainty or advance the goal.
- Respect the agent anatomy and context budget.
- Raw evidence should be archived, not carried forever.
- Claims, interpretations, and proxy results are not automatically authoritative.
- Stop only when the goal is sufficiently complete or no useful resources remain.

Return JSON only with this shape:
{
  "action": "inspect_resource|stop",
  "resource_id": "pending resource id or null",
  "reason": "why this is the next best bounded step",
  "instruction": "specific instruction for inspecting exactly this resource",
  "expected_value": "what useful evidence may be gained",
  "should_stop": false,
  "stop_reason": null,
  "context_to_carry_next": ["brief notes only"],
  "budget_estimate": {"input_tokens": 0, "output_tokens": 0, "tool_cost": "low|medium|high"}
}
""".strip()

    @staticmethod
    def _pending_resource_preview(
        state: AutonomousTaskState,
        resources: list[ResourceRef],
        *,
        limit: int = 20,
    ) -> list[ResourceRef]:
        pending = set(state.resources_pending[:limit])
        return [r for r in resources if r.resource_id in pending][:limit]

    def _archive_observation(
        self,
        *,
        archive_id: str,
        task_id: str,
        goal: str,
        resource: ResourceRef,
        observation: ObservationPacket,
    ) -> None:
        entry = EvidenceArchiveEntry(
            archive_id=archive_id,
            session_id=self.session_id,
            task_id=task_id,
            user_request=goal,
            final_answer=observation.summary,
            trace=[
                {
                    "tool": "autonomous_resource_adapter.inspect_resource",
                    "tool_input": {
                        "resource": resource.to_dict(),
                        "action": observation.action,
                    },
                    "observation": observation.to_dict(),
                    "observation_token_estimate": observation.token_estimate,
                }
            ],
            image_paths=[],
            metadata={
                "autonomous_executor": True,
                "resource_id": resource.resource_id,
                "resource_uri": resource.uri,
                "resource_type": resource.resource_type,
                "observation_error": observation.error,
            },
        )
        self.storage.append_archive(entry)

    def _rebuild_carry_packet(self):
        packet = self.carry_builder.build(
            session_id=self.session_id,
            recent_events=self.storage.read_recent_events(limit=80),
            new_events=[],
            alias_memory=self.storage.read_alias_memory(),
            correction_registry=self.storage.read_correction_registry(),
        )
        self.storage.write_carry_packet(packet)
        return packet

    def _write_state(self, state: AutonomousTaskState) -> Path:
        run_dir = self.output_root / state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "autonomous_task_state.json"
        write_json(path, state.to_dict())
        return path

    @staticmethod
    def _texts_for_status(events: list[MemoryEvent], statuses: set[str], limit: int = 20) -> list[str]:
        out = []
        for event in events:
            if event.decision.status in statuses:
                out.append((event.decision.carry_text or event.candidate.content)[:700])
        return out[-limit:]

    @staticmethod
    def _texts_for_type(events: list[MemoryEvent], candidate_types: set[str], limit: int = 20) -> list[str]:
        out = []
        for event in events:
            if event.candidate.candidate_type in candidate_types:
                out.append((event.decision.carry_text or event.candidate.content)[:700])
        return out[-limit:]
