from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.chronosieve.core.agent_anatomy import AgentAnatomy
from src.chronosieve.core.kv_sync import KVSyncPlanner, KVAction
from src.chronosieve.core.runtime_context import RuntimeContextBuilder
from src.chronosieve.core.runtime_backend import PromptOnlyRuntimeBackend
from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.utils import (
    estimate_tokens,
    extract_json_object,
    read_jsonl,
)

CHRONOSIEVE_ROLE_BRIEFING = """
# ChronoSieve Role Briefing

ChronoSieve is a generic memory-governance framework for long-horizon agents.
It is not a domain-specific benchmark solver and not a normal RAG pipeline.

The system has multiple actors. Each actor has a narrow job.

## Actor 1: Problem / User Request
The user provides the current problem or question.
The user question is the goal, not evidence by itself.

## Actor 2: ResourceAdapter
The ResourceAdapter is the world interface.
It exposes domain material as generic ResourceRef objects and inspects one bounded resource/action at a time.
It may know the domain, but ChronoSieve core must stay domain-neutral.
The adapter does not decide long-term memory.

Examples of resources:
- chat turn
- DICOM file
- code file
- PDF section
- support ticket
- legal clause
- log line
- financial filing section

## Actor 3: ObservationPacket
An ObservationPacket is what was observed from exactly one bounded inspection.
It is evidence input, not automatically durable memory and not automatically truth.

## Actor 4: MemoryCandidate
A MemoryCandidate is a proposed memory item extracted from an observation, answer, trace, correction, caveat, or artifact.
It is only a candidate. It has not yet earned future attention.

## Actor 5: Brain 1 / Task Agent
Brain 1 performs the task or answers using tools/evidence.
Brain 1 can be useful but can also overgeneralize, hallucinate, substitute proxies, or lose caveats.
Brain 1 claims are not automatically authoritative.

## Actor 6: Brain 2 / Sieve Worker
Brain 2 is the memory-governance judge.
It does not solve the task again.
It decides what future attention should do with each MemoryCandidate:
- fossil
- active
- archived
- decay
- invalidated
- non_authoritative

Brain 2 may recommend carry text, rehydration triggers, and KV future hints.

## Actor 7: PolicyGovernor
PolicyGovernor is the law.
It enforces invariants after Brain 2:
- Brain 1 claims and interpretations do not become authoritative truth by default.
- Proxy/substitute results must not become the original requested truth.
- High-risk caveats must survive.
- Invalidated/superseded items must not re-enter as truth.
- Raw payloads and artifacts stay external unless explicitly promoted safely.

PolicyGovernor can override Brain 2.

## Actor 8: ChronoSieveStorage
Storage is the temporal substrate.
It keeps:
- evidence_archive
- memory_events
- task_ledger
- sieve_worker_logs
- carry_packet
- alias_memory
- correction_registry
- artifacts_index

Storage is not active context. It is durable memory substrate.

## Actor 9: CarryPacket
CarryPacket is a bounded briefing note.
It is not a transcript and not the whole memory system.
It contains selected governed memory for continuity.

## Actor 10: Rehydration / Temporary Evidence
Rehydration temporarily loads archived evidence for the current question/task.
Temporary evidence is allowed when active memory is insufficient.
Temporary evidence is not automatically promoted to durable truth.
It must go through Sieve/Governor later if it should survive.

## Actor 11: KVSyncPlanner
KVSyncPlanner translates memory status into runtime-memory actions:
- fossil -> pin_or_prefill
- active -> keep_current
- archived -> external_only
- decay -> evict
- invalidated -> block
- non_authoritative -> load_with_warning
- rehydrated evidence -> temporary_prefill

It does not answer questions.

## Actor 12: RuntimeContextBuilder
RuntimeContextBuilder converts KV actions into runtime context segments:
- protected_prefix
- working_context
- warning_context
- temporary_evidence
- blocked_truth_registry
- external_reference_index

It packages memory for the backend.

## Actor 13: RuntimeBackend
RuntimeBackend prepares the runtime representation.
Today this may be prompt-only or prompt-cache based.
Future backends may mutate KV directly.
The backend is a staging layer, not a reasoning authority.

## Actor 14: MemoryAccessController
MemoryAccessController is the answer-time librarian/investigator.
It inspects the current runtime memory and memory inventories.
It decides whether active memory is enough or whether archived evidence should be temporarily rehydrated.
It does not answer the user question.
It selects handles only from ChronoSieve memory structures.

## Actor 15: AnswerSynthesizer
AnswerSynthesizer is the final witness.
It answers only from the governed runtime context.
It must respect authority boundaries:
- protected/working memory can support answers
- temporary evidence can directly support current answers
- warning context is non-authoritative unless corroborated
- blocked truth must not be reused
- external refs are handles, not content
If unsupported, it must say UNKNOWN.

## Core Doctrine
Raw context is not memory.
Saved text is not truth.
Retrieved text is not automatically authoritative.
Memory is a governed state machine.
The goal is not to carry everything; the goal is to preserve the right things with the right authority.
"""


@dataclass
class MemoryAccessDecision:
    """
    Generic answer-time memory-access decision.

    This is not benchmark-specific. It chooses from ChronoSieve's own
    memory/archive handles.
    """

    needs_temporary_evidence: bool
    selected_archive_refs: list[str] = field(default_factory=list)
    selected_event_ids: list[str] = field(default_factory=list)
    selected_ledger_task_ids: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    raw_selector_output: str = ""
    selector_used: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryAccessResult:
    question: str
    answer: str
    latency_seconds: float
    prompt_tokens: int
    runtime_context_tokens: int
    memory_access_decision: MemoryAccessDecision
    kv_action_summary: dict[str, int]
    runtime_segments: dict[str, int]
    temporary_evidence_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_access_decision"] = self.memory_access_decision.to_dict()
        return data


class ChronoSieveMemoryAccessController:
    """
    Generic answer-time controller.

    Purpose:
        Let the answering model use ChronoSieve's own design instead of being
        handed only a carry packet.

    It does not know LoCoMo, DICOM, medicine, code, legal, finance, etc.
    It only knows ChronoSieve storage structures:
        - carry packet
        - memory events
        - ledger
        - evidence archive
        - correction registry
        - KV sync plan
        - runtime context package
        - prompt-only runtime backend
    """

    def __init__(
        self,
        *,
        storage: ChronoSieveStorage,
        llm: Any,
        anatomy: AgentAnatomy | None = None,
        max_event_inventory: int = 80,
        max_archive_inventory: int = 80,
        max_ledger_inventory: int = 40,
        max_selected_archives: int = 5,
        max_selected_events: int = 8,
        max_selected_ledger: int = 5,
        max_temporary_evidence_chars: int = 9000,
        runtime_segment_chars: int = 12000,
        max_selector_context_chars: int = 18000,
    ):
        self.storage = storage
        self.llm = llm
        self.anatomy = anatomy or AgentAnatomy()

        self.max_event_inventory = max_event_inventory
        self.max_archive_inventory = max_archive_inventory
        self.max_ledger_inventory = max_ledger_inventory

        self.max_selected_archives = max_selected_archives
        self.max_selected_events = max_selected_events
        self.max_selected_ledger = max_selected_ledger

        self.max_temporary_evidence_chars = max_temporary_evidence_chars
        self.runtime_segment_chars = runtime_segment_chars
        self.max_selector_context_chars = max_selector_context_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> MemoryAccessResult:
        """
        Full answer-time ChronoSieve path:

            inspect memory inventory
            → decide temporary evidence
            → build KVSyncPlan
            → build RuntimeContextPackage
            → apply runtime backend
            → answer from governed context
        """
        started = time.time()

        events = read_jsonl(self.storage.events_path)
        archives = read_jsonl(self.storage.archive_path)
        ledger = read_jsonl(self.storage.ledger_path)
        carry_packet = self.storage.read_carry_packet_text()
        corrections = self.storage.read_correction_registry()

        base_runtime = self._build_runtime_context(
            events=events,
            corrections=corrections,
            carry_packet=carry_packet,
            temporary_evidence_md="",
            temporary_archive_refs=[],
        )

        inventory = self._build_memory_inventory(
            question=question,
            events=events,
            archives=archives,
            ledger=ledger,
            base_runtime_context=base_runtime["runtime_context"],
        )

        decision = self._decide_memory_access(
            question=question,
            inventory=inventory,
            base_runtime_context=base_runtime["runtime_context"],
        )

        temporary_evidence_md = self._build_temporary_evidence(
            decision=decision,
            events=events,
            archives=archives,
            ledger=ledger,
        )

        final_runtime = self._build_runtime_context(
            events=events,
            corrections=corrections,
            carry_packet=carry_packet,
            temporary_evidence_md=temporary_evidence_md,
            temporary_archive_refs=decision.selected_archive_refs,
        )

        answer, prompt_tokens = self._answer_from_runtime(
            question=question,
            runtime_context=final_runtime["runtime_context"],
            access_decision=decision,
        )

        latency = round(time.time() - started, 3)

        return MemoryAccessResult(
            question=question,
            answer=answer,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            runtime_context_tokens=estimate_tokens(final_runtime["runtime_context"]),
            memory_access_decision=decision,
            kv_action_summary=final_runtime["kv_action_summary"],
            runtime_segments=final_runtime["runtime_segments"],
            temporary_evidence_tokens=estimate_tokens(temporary_evidence_md),
            metadata={
                "session_id": self.storage.session_id,
                "events_seen": len(events),
                "archives_seen": len(archives),
                "ledger_rows_seen": len(ledger),
                "carry_packet_tokens": estimate_tokens(carry_packet),
                "base_runtime_tokens": estimate_tokens(base_runtime["runtime_context"]),
            },
        )

    # ------------------------------------------------------------------
    # Memory inventory
    # ------------------------------------------------------------------

    def _build_memory_inventory(
        self,
        *,
        question: str,
        events: list[dict[str, Any]],
        archives: list[dict[str, Any]],
        ledger: list[dict[str, Any]],
        base_runtime_context: str,
    ) -> dict[str, Any]:
        event_hits = []
        for event in events:
            text = self._event_search_text(event)
            score = self._generic_score(question, text)
            event_hits.append({
                "event_id": event.get("event_id"),
                "task_id": event.get("task_id"),
                "archive_ref": self._event_archive_ref(event),
                "score": score,
                "status": (event.get("decision") or {}).get("status"),
                "candidate_type": (event.get("candidate") or {}).get("candidate_type"),
                "authority_hint": self._authority_hint(event),
                "preview": self._compact(
                    (event.get("decision") or {}).get("carry_text")
                    or (event.get("candidate") or {}).get("content")
                    or "",
                    700,
                ),
            })

        archive_hits = []
        for archive in archives:
            text = self._archive_search_text(archive)
            score = self._generic_score(question, text)
            archive_hits.append({
                "archive_id": archive.get("archive_id"),
                "task_id": archive.get("task_id"),
                "score": score,
                "metadata_preview": self._compact_json(archive.get("metadata", {}), 700),
                "user_request_preview": self._compact(archive.get("user_request", ""), 500),
                "final_answer_preview": self._compact(archive.get("final_answer", ""), 900),
                "trace_tool_preview": [
                    str(step.get("tool", "unknown_tool"))
                    for step in (archive.get("trace") or [])[:6]
                ],
            })

        ledger_hits = []
        for row in ledger:
            text = self._ledger_search_text(row)
            score = self._generic_score(question, text)
            ledger_hits.append({
                "task_id": row.get("task_id"),
                "archive_refs": row.get("evidence_refs", []),
                "score": score,
                "status": row.get("status"),
                "task_summary_preview": self._compact(row.get("task_summary", ""), 700),
                "active_findings_preview": [
                    self._compact(x, 400) for x in (row.get("active_findings") or [])[:6]
                ],
                "caveats_preview": [
                    self._compact(x, 400) for x in (row.get("caveats") or [])[:4]
                ],
                "non_authoritative_preview": [
                    self._compact(x, 400) for x in (row.get("non_authoritative_items") or [])[:4]
                ],
            })

        event_hits = self._rank_hits(event_hits, id_key="event_id")[:self.max_event_inventory]
        archive_hits = self._rank_hits(archive_hits, id_key="archive_id")[:self.max_archive_inventory]
        ledger_hits = self._rank_hits(ledger_hits, id_key="task_id")[:self.max_ledger_inventory]

        base_runtime_preview = self._compact(base_runtime_context, 5000)

        inventory = {
            "agent_anatomy": self.anatomy.compact_briefing(),
            "available_memory_tools": [
                {
                    "name": "inspect_runtime_context",
                    "purpose": "Inspect protected, active, warning, blocked, temporary, and external-reference runtime memory.",
                },
                {
                    "name": "search_memory_events",
                    "purpose": "Search committed ChronoSieve MemoryEvents by generic text/status/metadata.",
                },
                {
                    "name": "search_evidence_archive",
                    "purpose": "Search archived raw evidence and task observations by generic text/metadata.",
                },
                {
                    "name": "search_task_ledger",
                    "purpose": "Search compact task ledger summaries and evidence refs.",
                },
                {
                    "name": "rehydrate_selected_handles",
                    "purpose": "Temporarily load selected archive/event/ledger evidence for this question only.",
                },
                {
                    "name": "build_runtime_context",
                    "purpose": "Convert governed memory + temporary evidence into runtime context segments.",
                },
            ],
            "base_runtime_context_preview": base_runtime_preview,
            "memory_event_inventory": event_hits,
            "archive_inventory": archive_hits,
            "ledger_inventory": ledger_hits,
        }

        text = json.dumps(inventory, ensure_ascii=False, default=str)
        if len(text) > self.max_selector_context_chars:
            inventory["archive_inventory"] = archive_hits[: max(10, self.max_archive_inventory // 2)]
            inventory["memory_event_inventory"] = event_hits[: max(20, self.max_event_inventory // 2)]
            inventory["ledger_inventory"] = ledger_hits[: max(10, self.max_ledger_inventory // 2)]

        return inventory

    def _rank_hits(self, rows: list[dict[str, Any]], *, id_key: str) -> list[dict[str, Any]]:
        # Generic ordering: relevance first, then keep nonzero and recent-ish rows.
        # This is not benchmark-specific; it is a simple memory index.
        rows = [r for r in rows if r.get(id_key)]
        rows.sort(key=lambda r: int(r.get("score") or 0), reverse=True)

        positive = [r for r in rows if int(r.get("score") or 0) > 0]
        if positive:
            return positive

        # If no lexical overlap exists, still expose recent handles so the
        # selector can decide whether current memory is insufficient.
        return rows[-20:]

    # ------------------------------------------------------------------
    # Memory access decision
    # ------------------------------------------------------------------

    def _decide_memory_access(
        self,
        *,
        question: str,
        inventory: dict[str, Any],
        base_runtime_context: str,
    ) -> MemoryAccessDecision:
        system = SystemMessage(content=self._memory_selector_system_prompt())

        human = HumanMessage(content=json.dumps({
            "question": question,
            "controller_role": "generic_chronosieve_answer_time_memory_access",
            "important": [
                "Do not answer the question.",
                "Do not use benchmark labels, gold answers, or hidden evidence ids.",
                "Select only from provided ChronoSieve handles.",
                "If base runtime memory is enough, choose no temporary evidence.",
                "If exact evidence is needed, select archive refs and/or memory event ids.",
                "Temporary evidence is allowed because rehydration is part of ChronoSieve.",
            ],
            "inventory": inventory,
            "return_json_shape": {
                "needs_temporary_evidence": True,
                "selected_archive_refs": ["archive_..."],
                "selected_event_ids": ["event_..."],
                "selected_ledger_task_ids": ["task_or_autostep_..."],
                "reason": "why these handles are enough or why no temporary evidence is needed",
                "confidence": 0.0,
            },
        }, indent=2, ensure_ascii=False, default=str))

        valid_archives = {
            row.get("archive_id")
            for row in inventory.get("archive_inventory", [])
            if row.get("archive_id")
        }
        valid_events = {
            row.get("event_id")
            for row in inventory.get("memory_event_inventory", [])
            if row.get("event_id")
        }
        valid_tasks = {
            row.get("task_id")
            for row in inventory.get("ledger_inventory", [])
            if row.get("task_id")
        }

        try:
            response = self.llm.invoke([system, human])
            raw = str(getattr(response, "content", response))
            parsed = extract_json_object(raw)

            selected_archives = [
                str(x) for x in parsed.get("selected_archive_refs", [])
                if x in valid_archives
            ][: self.max_selected_archives]

            selected_events = [
                str(x) for x in parsed.get("selected_event_ids", [])
                if x in valid_events
            ][: self.max_selected_events]

            selected_tasks = [
                str(x) for x in parsed.get("selected_ledger_task_ids", [])
                if x in valid_tasks
            ][: self.max_selected_ledger]

            confidence = self._safe_float(parsed.get("confidence"), default=0.0)

            return MemoryAccessDecision(
                needs_temporary_evidence=bool(parsed.get("needs_temporary_evidence"))
                or bool(selected_archives or selected_events or selected_tasks),
                selected_archive_refs=selected_archives,
                selected_event_ids=selected_events,
                selected_ledger_task_ids=selected_tasks,
                reason=str(parsed.get("reason") or "Memory selector completed."),
                confidence=confidence,
                raw_selector_output=raw[:2500],
                selector_used="llm_selector",
            )

        except Exception as exc:
            return self._fallback_memory_access_decision(
                inventory=inventory,
                reason=f"LLM selector failed; used generic fallback. Error: {exc}",
            )

    def _fallback_memory_access_decision(
        self,
        *,
        inventory: dict[str, Any],
        reason: str,
    ) -> MemoryAccessDecision:
        archives = [
            str(r["archive_id"])
            for r in inventory.get("archive_inventory", [])
            if r.get("archive_id") and int(r.get("score") or 0) > 0
        ][: self.max_selected_archives]

        events = [
            str(r["event_id"])
            for r in inventory.get("memory_event_inventory", [])
            if r.get("event_id") and int(r.get("score") or 0) > 0
        ][: self.max_selected_events]

        tasks = [
            str(r["task_id"])
            for r in inventory.get("ledger_inventory", [])
            if r.get("task_id") and int(r.get("score") or 0) > 0
        ][: self.max_selected_ledger]

        return MemoryAccessDecision(
            needs_temporary_evidence=bool(archives or events or tasks),
            selected_archive_refs=archives,
            selected_event_ids=events,
            selected_ledger_task_ids=tasks,
            reason=reason,
            confidence=0.35,
            raw_selector_output="",
            selector_used="generic_lexical_fallback",
        )

    @staticmethod
    def _memory_selector_system_prompt() -> str:
        return (
        CHRONOSIEVE_ROLE_BRIEFING
        + """

# Your Current Role: MemoryAccessController

You are Actor 14: the answer-time librarian/investigator.

You are not Brain 1.
You are not Brain 2.
You are not the PolicyGovernor.
You are not the AnswerSynthesizer.
You do not answer the user's question.

Your job is to decide whether the AnswerSynthesizer already has enough governed runtime memory,
or whether more evidence must be temporarily rehydrated from ChronoSieve's own storage.

## Inputs You Receive
You receive:
1. the user question
2. a preview of the current runtime context
3. searchable inventories of:
   - MemoryEvents
   - EvidenceArchive entries
   - TaskLedger rows

These inventories are not the full world.
They are handles into ChronoSieve's governed memory substrate.

## Available Conceptual Tools
You may conceptually use:

1. inspect_runtime_context
   Look at protected, active, warning, blocked, temporary, and external-reference runtime memory.

2. search_memory_events
   Search committed MemoryEvents by content, status, authority, metadata, and archive refs.

3. search_evidence_archive
   Search archived evidence when current runtime memory is insufficient.

4. search_task_ledger
   Search compact task-level summaries and evidence refs.

5. rehydrate_selected_handles
   Temporarily load selected archive/event/ledger evidence for this question only.

6. build_runtime_context
   Rebuild runtime context from memory events, corrections, and temporary evidence.

## Authority Rules
- Protected and active memory can be enough if it directly answers the question.
- Warning/non-authoritative memory can guide search, but cannot be treated as final truth.
- Blocked/invalidated memory constrains the answer. It should not support the answer.
- Archive refs are handles. You may select them for temporary evidence.
- Temporary evidence is allowed because rehydration is part of ChronoSieve.
- Select the smallest sufficient set of handles.

## Forbidden Behavior
- Do not answer the user question.
- Do not invent archive ids, event ids, task ids, or facts.
- Do not use benchmark labels, gold answers, evidence IDs, or external knowledge.
- Do not select handles just because they are recent.
- Do not over-select everything. Choose only what is likely needed.

## Decision Standard
Ask yourself:
1. Does protected/working context already answer this?
2. Is the answer exact or evidence-sensitive?
3. Are there warning/blocked constraints?
4. Do I need temporary archived evidence?
5. Which smallest set of handles should be loaded?

Return JSON only:
{
  "needs_temporary_evidence": true,
  "selected_archive_refs": ["archive_id"],
  "selected_event_ids": ["event_id"],
  "selected_ledger_task_ids": ["task_id"],
  "reason": "brief rationale grounded in the provided inventory",
  "confidence": 0.0
}
""".strip()
    )

    # ------------------------------------------------------------------
    # Temporary evidence
    # ------------------------------------------------------------------

    def _build_temporary_evidence(
        self,
        *,
        decision: MemoryAccessDecision,
        events: list[dict[str, Any]],
        archives: list[dict[str, Any]],
        ledger: list[dict[str, Any]],
    ) -> str:
        archive_by_id = {
            str(row.get("archive_id")): row
            for row in archives
            if row.get("archive_id")
        }
        event_by_id = {
            str(row.get("event_id")): row
            for row in events
            if row.get("event_id")
        }
        ledger_by_task = {
            str(row.get("task_id")): row
            for row in ledger
            if row.get("task_id")
        }

        sections: list[str] = []

        sections.append("# ChronoSieve Temporary Evidence")
        sections.append("")
        sections.append(
            "This section was selected by the generic answer-time memory access controller. "
            "It is temporary evidence for the current question only. It must not become durable "
            "truth unless a later Sieve/Governor pass promotes it."
        )
        sections.append("")
        sections.append(f"Selection reason: {decision.reason}")
        sections.append("")

        if decision.selected_archive_refs:
            sections.append("## Rehydrated archive evidence")
            for ref in decision.selected_archive_refs:
                archive = archive_by_id.get(ref)
                if not archive:
                    continue
                sections.append(f"### Archive `{ref}`")
                sections.append(f"- task_id: `{archive.get('task_id')}`")
                if archive.get("user_request"):
                    sections.append("- original_request: " + self._compact(archive.get("user_request"), 900))
                if archive.get("metadata"):
                    sections.append("- metadata: " + self._compact_json(archive.get("metadata"), 1300))
                if archive.get("final_answer"):
                    sections.append("- archived_summary: " + self._compact(archive.get("final_answer"), 1800))

                for idx, step in enumerate(archive.get("trace", []) or [], start=1):
                    tool = step.get("tool", "unknown_tool")
                    observation = step.get("observation", "")
                    if observation:
                        sections.append(f"- trace_step_{idx} `{tool}`: " + self._compact(observation, 1800))
                sections.append("")

        if decision.selected_event_ids:
            sections.append("## Selected memory events")
            for event_id in decision.selected_event_ids:
                event = event_by_id.get(event_id)
                if not event:
                    continue

                candidate = event.get("candidate", {}) or {}
                decision_row = event.get("decision", {}) or {}

                sections.append(f"### Event `{event_id}`")
                sections.append(f"- task_id: `{event.get('task_id')}`")
                sections.append(f"- memory_status: `{decision_row.get('status')}`")
                sections.append(f"- candidate_type: `{candidate.get('candidate_type')}`")
                sections.append(f"- archive_ref: `{decision_row.get('archive_ref') or candidate.get('evidence_ref')}`")
                sections.append("- memory_text: " + self._compact(
                    decision_row.get("carry_text") or candidate.get("content") or "",
                    1600,
                ))
                if candidate.get("metadata"):
                    sections.append("- metadata: " + self._compact_json(candidate.get("metadata"), 1100))
                if decision_row.get("reason"):
                    sections.append("- governance_reason: " + self._compact(decision_row.get("reason"), 800))
                sections.append("")

        if decision.selected_ledger_task_ids:
            sections.append("## Selected task ledger rows")
            for task_id in decision.selected_ledger_task_ids:
                row = ledger_by_task.get(task_id)
                if not row:
                    continue

                sections.append(f"### Task `{task_id}`")
                sections.append("- task_summary: " + self._compact(row.get("task_summary", ""), 1200))
                sections.append("- evidence_refs: " + self._compact_json(row.get("evidence_refs", []), 800))
                if row.get("active_findings"):
                    sections.append("- active_findings: " + self._compact_json(row.get("active_findings"), 1500))
                if row.get("caveats"):
                    sections.append("- caveats: " + self._compact_json(row.get("caveats"), 1200))
                if row.get("non_authoritative_items"):
                    sections.append("- non_authoritative_items: " + self._compact_json(row.get("non_authoritative_items"), 1200))
                sections.append("")

        content = "\n".join(sections).strip() + "\n"

        if len(content) > self.max_temporary_evidence_chars:
            content = content[: self.max_temporary_evidence_chars]
            content += "\n...[ChronoSieve temporary evidence truncated]\n"

        return content

    # ------------------------------------------------------------------
    # Runtime context
    # ------------------------------------------------------------------

    def _build_runtime_context(
        self,
        *,
        events: list[dict[str, Any]],
        corrections: dict[str, Any],
        carry_packet: str,
        temporary_evidence_md: str,
        temporary_archive_refs: list[str],
    ) -> dict[str, Any]:
        kv_plan = KVSyncPlanner(
            include_external_actions=True,
            max_text_chars=900,
        ).plan(
            session_id=self.storage.session_id,
            events=events,
            carry_packet_text=carry_packet,
            correction_registry=corrections,
            rehydrated_archive_refs=temporary_archive_refs,
        )

        if temporary_evidence_md.strip():
            kv_plan.actions.append(KVAction(
                action_type="temporary_prefill",
                authority="temporary",
                text=temporary_evidence_md,
                reason="Answer-time memory access selected this temporary evidence for the current question.",
                priority=92,
                archive_ref=",".join(temporary_archive_refs) or None,
                tags=["answer_time_memory_access", "temporary_evidence"],
                token_estimate=estimate_tokens(temporary_evidence_md),
                metadata={
                    "source": "ChronoSieveMemoryAccessController",
                    "temporary": True,
                },
            ))
            kv_plan.summary["temporary_prefill"] = kv_plan.summary.get("temporary_prefill", 0) + 1
            kv_plan.token_estimate += estimate_tokens(temporary_evidence_md)

        runtime_package = RuntimeContextBuilder(
            max_segment_chars=self.runtime_segment_chars,
        ).build(kv_plan)

        backend = PromptOnlyRuntimeBackend()
        backend_plan = backend.plan(runtime_package)
        applied = backend.apply(backend_plan)

        runtime_segments = {
            segment.segment_type: segment.token_estimate
            for segment in runtime_package.segments
        }

        return {
            "runtime_context": applied["runtime_context"],
            "runtime_package": runtime_package.to_dict(),
            "backend_plan": backend_plan.to_dict(),
            "kv_action_summary": kv_plan.summary,
            "runtime_segments": runtime_segments,
        }

    # ------------------------------------------------------------------
    # Answer synthesis
    # ------------------------------------------------------------------

    def _answer_from_runtime(
        self,
        *,
        question: str,
        runtime_context: str,
        access_decision: MemoryAccessDecision,
    ) -> tuple[str, int]:
        system = SystemMessage(content=self._answer_system_prompt())

        human = HumanMessage(content=(
            "# Agent Anatomy\n\n"
            + json.dumps(self.anatomy.compact_briefing(), indent=2, ensure_ascii=False)
            + "\n\n---\n\n"
            + "# Memory Access Decision\n\n"
            + json.dumps(access_decision.to_dict(), indent=2, ensure_ascii=False)
            + "\n\n---\n\n"
            + "# ChronoSieve Governed Runtime Context\n\n"
            + runtime_context
            + "\n\n---\n\n"
            + "# User Question\n\n"
            + question
            + "\n\n# Answer\n"
        ))

        response = self.llm.invoke([system, human])
        answer = str(getattr(response, "content", response)).strip()
        prompt_tokens = estimate_tokens(system.content + "\n" + human.content)

        return answer, prompt_tokens

    @staticmethod
    def _answer_system_prompt() -> str:
        return (
        CHRONOSIEVE_ROLE_BRIEFING
        + """

# Your Current Role: AnswerSynthesizer

You are Actor 15: the final witness.

You are not the ResourceAdapter.
You are not Brain 2.
You are not the MemoryAccessController.
You are not the PolicyGovernor.
You cannot fetch new memory yourself.
You answer only from the runtime context that ChronoSieve prepared for you.

## Inputs You Receive
You receive:
1. Agent Anatomy
2. Memory Access Decision
3. ChronoSieve Governed Runtime Context
4. User Question

The runtime context may include these sections:

## protected_prefix
Stable/pinned memory.
Use as durable memory unless directly contradicted by temporary evidence or blocked truth.

## working_context
Active deterministic working memory.
Use for normal continuity.

## temporary_evidence
Evidence loaded for this current question only.
If it directly answers the question, prefer it over vague summaries.
Do not treat it as durable beyond this answer.

## warning_context
Useful but non-authoritative memory.
Do not present it as settled truth unless direct evidence also supports it.
If used, qualify it.

## blocked_truth_registry
Claims/facts that must not re-enter as truth.
If the user asks about these, explain that they are blocked/invalidated/non-authoritative.

## external_reference_index
Archive handles only.
Do not infer content from a handle alone.

## Answer Rules
- Use only the provided ChronoSieve runtime context.
- Do not invent missing facts.
- Do not use outside knowledge.
- Do not use benchmark gold answers or hidden labels.
- Prefer direct temporary evidence for exact-detail questions.
- Resolve relative time only when explicit timestamps are present in the runtime context.
- Preserve authority boundaries.
- If the answer is unsupported, say UNKNOWN.
- For short-answer QA, return only the shortest supported answer.
- Do not explain the ChronoSieve machinery unless the user asks.

## Internal Checklist Before Answering
1. Is there direct temporary evidence?
2. Is there protected or working memory that supports the answer?
3. Is the only support warning/non-authoritative?
4. Is there blocked truth that prevents using a claim?
5. Is the answer exact enough?
6. If not supported, answer UNKNOWN.

Output only the answer unless the user asks for reasoning.
""".strip()
    )

    # ------------------------------------------------------------------
    # Generic search/index helpers
    # ------------------------------------------------------------------

    def _generic_score(self, query: str, text: str) -> int:
        q = self._tokens(query)
        t = self._tokens(text)
        if not q or not t:
            return 0
        return len(q & t)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "what",
            "which", "when", "where", "were", "was", "are", "you", "did",
            "have", "their", "about", "into", "between", "both", "would",
            "could", "should", "does", "done", "there", "here", "your",
            "they", "them", "then", "than", "into", "onto", "also",
        }
        raw = re.findall(r"[a-zA-Z0-9_\-]+", str(text).lower())
        return {x for x in raw if len(x) > 2 and x not in stop}

    def _event_search_text(self, event: dict[str, Any]) -> str:
        candidate = event.get("candidate", {}) or {}
        decision = event.get("decision", {}) or {}
        return "\n".join([
            str(event.get("event_id", "")),
            str(event.get("task_id", "")),
            str(candidate.get("candidate_type", "")),
            str(candidate.get("source_type", "")),
            str(candidate.get("content", "")),
            json.dumps(candidate.get("metadata", {}), ensure_ascii=False, default=str),
            str(decision.get("status", "")),
            str(decision.get("carry_text", "")),
            str(decision.get("reason", "")),
            str(decision.get("archive_ref", "")),
        ])

    def _archive_search_text(self, archive: dict[str, Any]) -> str:
        parts = [
            str(archive.get("archive_id", "")),
            str(archive.get("task_id", "")),
            str(archive.get("user_request", "")),
            str(archive.get("final_answer", "")),
            json.dumps(archive.get("metadata", {}), ensure_ascii=False, default=str),
        ]

        for step in archive.get("trace", []) or []:
            parts.append(str(step.get("tool", "")))
            parts.append(self._compact(step.get("observation", ""), 1500))

        return "\n".join(parts)

    def _ledger_search_text(self, row: dict[str, Any]) -> str:
        return "\n".join([
            str(row.get("task_id", "")),
            str(row.get("user_request", "")),
            str(row.get("task_summary", "")),
            json.dumps(row.get("evidence_refs", []), ensure_ascii=False, default=str),
            json.dumps(row.get("active_findings", []), ensure_ascii=False, default=str),
            json.dumps(row.get("caveats", []), ensure_ascii=False, default=str),
            json.dumps(row.get("uncertainties", []), ensure_ascii=False, default=str),
            json.dumps(row.get("non_authoritative_items", []), ensure_ascii=False, default=str),
            json.dumps(row.get("metadata", {}), ensure_ascii=False, default=str),
        ])

    @staticmethod
    def _event_archive_ref(event: dict[str, Any]) -> str | None:
        candidate = event.get("candidate", {}) or {}
        decision = event.get("decision", {}) or {}
        return decision.get("archive_ref") or candidate.get("evidence_ref")

    @staticmethod
    def _authority_hint(event: dict[str, Any]) -> str:
        decision = event.get("decision", {}) or {}
        status = decision.get("status")

        if status in {"fossil", "active"}:
            return "authoritative_runtime_memory"
        if status == "non_authoritative":
            return "warning_only_non_authoritative"
        if status == "invalidated":
            return "blocked_truth"
        if status in {"archived", "decay"}:
            return "external_or_evicted"
        return "unknown"

    @staticmethod
    def _compact(text: Any, max_chars: int = 900) -> str:
        clean = " ".join(str(text).split())
        if len(clean) > max_chars:
            return clean[:max_chars] + "...[truncated]"
        return clean

    @staticmethod
    def _compact_json(obj: Any, max_chars: int = 900) -> str:
        try:
            text = json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            text = str(obj)
        clean = " ".join(text.split())
        if len(clean) > max_chars:
            return clean[:max_chars] + "...[truncated]"
        return clean

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            out = float(value)
            return max(0.0, min(1.0, out))
        except Exception:
            return default