from __future__ import annotations

from typing import Any, Callable

from .schemas import (
    EvidenceArchiveEntry,
    TaskLedgerEntry,
    CorrectionRecord,
    ContextBudget,
    new_id,
)
from .storage import ChronoSieveStorage
from .sieve_worker import GemmaSieveWorker
from .policy_governor import PolicyGovernor
from .carry_packet import CarryPacketBuilder
from src.chronosieve.adapters.mike.trace_parser import MikeTraceParser
from .rehydration import RehydrationPolicy

class ChronoSieveSession:
    """
    Generic ChronoSieve session controller, currently wired with a Mike adapter.

    Brain 1: existing task agent callable, e.g. invoke_mike_with_trace()
    Brain 2: GemmaSieveWorker, a restricted memory-governance call
    Governor: Python hard-invariant enforcement
    Storage: file-backed temporal memory

    v1 design goal:
    Every turn is mostly fresh, but not memoryless.
    Brain 1 receives current request + carry packet, not full raw chat history forever.
    """

    def __init__(
        self,
        *,
        session_id: str,
        task_agent_callable: Callable[[str, list[dict[str, str]]], dict[str, Any]],
        sieve_worker: GemmaSieveWorker,
        storage: ChronoSieveStorage,
        trace_parser: MikeTraceParser | None = None,
        governor: PolicyGovernor | None = None,
        carry_builder: CarryPacketBuilder | None = None,
        context_budget: ContextBudget | None = None,
        rehydration_policy: RehydrationPolicy | None = None,
    ):
        self.session_id = session_id
        self.task_agent_callable = task_agent_callable
        self.sieve_worker = sieve_worker
        self.storage = storage
        self.trace_parser = trace_parser or MikeTraceParser()
        self.governor = governor or PolicyGovernor()
        self.carry_builder = carry_builder or CarryPacketBuilder()
        self.context_budget = context_budget or ContextBudget()
        self.rehydration_policy = rehydration_policy or RehydrationPolicy()

    def handle_turn(
        self,
        user_request: str,
        recent_display_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Main controller entrypoint.

        recent_display_history is optional and should be only the most recent 1-2 turns.
        The authoritative memory is the ChronoSieve carry packet and storage, not Streamlit history.
        """
        task_id = new_id("task")
        archive_id = new_id("archive")

        prior_carry_packet = self.storage.read_carry_packet_text()

        rehydration_bundle = self.rehydration_policy.build_bundle(
            user_request=user_request,
            recent_archives=self.storage.read_recent_archives(limit=8),
            recent_ledger=self.storage.read_recent_ledger(limit=12),
            artifacts_index=self.storage.read_artifacts_index(),
        )

        history_for_brain1 = self._build_brain1_history(
            carry_packet=prior_carry_packet,
            rehydrated_evidence=(
                rehydration_bundle.content_md if rehydration_bundle else None
            ),
            recent_display_history=recent_display_history or [],
        )

        # Brain 1 performs the task.
        task_result = self.task_agent_callable(user_request, history_for_brain1)
        final_answer = str(task_result.get("answer", ""))
        trace = list(task_result.get("trace", []) or [])
        image_paths = list(task_result.get("image_paths", []) or [])

        # Archive full raw evidence first. Archive everything, carry selectively.
        archive_entry = EvidenceArchiveEntry(
            archive_id=archive_id,
            session_id=self.session_id,
            task_id=task_id,
            user_request=user_request,
            final_answer=final_answer,
            trace=trace,
            image_paths=image_paths,
            metadata={
                "latency_seconds": task_result.get("latency_seconds"),
                "estimated_response_tokens": task_result.get("estimated_response_tokens"),
                "success": task_result.get("success"),
                "error": task_result.get("error"),
            },
        )
        self.storage.append_archive(archive_entry)
        self.storage.add_artifacts(task_id=task_id, image_paths=image_paths, archive_id=archive_id)

        # Adapter extracts hard evidence candidates.
        candidates = self.trace_parser.parse_turn(
            task_id=task_id,
            archive_id=archive_id,
            user_request=user_request,
            final_answer=final_answer,
            trace=trace,
            image_paths=image_paths,
        )

        # Brain 2 recommends memory decisions from compact candidate briefings.
        sieve_result = self.sieve_worker.decide(
            user_request=user_request,
            final_answer=final_answer,
            candidates=candidates,
            prior_carry_packet=prior_carry_packet,
            recent_ledger=self.storage.read_recent_ledger(limit=12),
            alias_memory=self.storage.read_alias_memory(),
            correction_registry=self.storage.read_correction_registry(),
        )

        # Persist compact Brain 2 diagnostics.
        # Full evidence remains in evidence_archive.jsonl.
        # Final committed memory remains in memory_events.jsonl.
        # This log is only for inspecting the memory judge itself.
        parsed_worker_output = sieve_result.get("parsed_worker_output") or {}
        parsed_decisions = parsed_worker_output.get("decisions") or []

        candidate_briefings_preview = []
        for row in (sieve_result.get("candidate_briefings") or [])[:40]:
            candidate_briefings_preview.append({
                "key": row.get("key"),
                "candidate_type": row.get("candidate_type"),
                "generic_role": row.get("generic_role"),
                "summary_preview": str(row.get("summary", ""))[:280],
                "source_type": row.get("source_type"),
                "evidence_ref": row.get("evidence_ref"),
            })

        decision_preview = []
        for row in parsed_decisions[:60]:
            decision_preview.append({
                "key": row.get("key"),
                "status": row.get("status"),
                "reason_preview": str(row.get("reason", ""))[:240],
                "carry_text_preview": str(row.get("carry_text", ""))[:240] if row.get("carry_text") else None,
                "kv_future_hint": row.get("kv_future_hint"),
            })

        self.storage.append_sieve_worker_log({
            "session_id": self.session_id,
            "task_id": task_id,
            "archive_id": archive_id,
            "candidate_count": len(candidates),
            "decision_count": sieve_result.get("decision_count"),
            "parse_success": sieve_result.get("parse_success"),
            "fallback_used": sieve_result.get("fallback_used"),
            "worker_error": sieve_result.get("worker_error"),
            "task_summary": parsed_worker_output.get("task_summary"),
            "required_cautions": parsed_worker_output.get("required_cautions", []),
            "candidate_briefings_preview": candidate_briefings_preview,
            "decision_preview": decision_preview,
            "raw_worker_output_preview": str(sieve_result.get("raw_worker_output", ""))[:4000],
        })

        # Python governor enforces invariants.
        events = self.governor.validate(
            task_id=task_id,
            candidates=candidates,
            decisions=sieve_result["decisions"],
        )

        for event in events:
            self.storage.append_memory_event(event)

            # v1: store alias memory when an alias learning survived.
            candidate = event.candidate
            if candidate.metadata.get("learning_type") == "alias_normalization":
                self.storage.add_alias(
                    alias=str(candidate.metadata.get("alias")),
                    canonical=str(candidate.metadata.get("canonical")),
                    reason=event.decision.reason,
                    evidence_ref=candidate.evidence_ref,
                )

        # Store correction records recommended by Brain 2.
        for row in sieve_result.get("correction_records", []) or []:
            try:
                record = CorrectionRecord(
                    correction_id=new_id("corr"),
                    session_id=self.session_id,
                    correction_type=row.get("correction_type", "supersession"),
                    old_state=row.get("old_state"),
                    new_state=str(row.get("new_state", "")),
                    reason=str(row.get("reason", "")),
                    affected_task_ids=[task_id],
                    evidence_refs=[archive_id],
                )
                self.storage.add_correction(record)
            except Exception:
                # Do not let malformed correction suggestions break task flow.
                pass

        # Compact task ledger entry.
        ledger_entry = self._build_ledger_entry(
            task_id=task_id,
            user_request=user_request,
            archive_id=archive_id,
            final_answer=final_answer,
            task_summary=str(sieve_result.get("task_summary", "ChronoSieve turn processed.")),
            events=events,
        )
        self.storage.append_ledger(ledger_entry)

        # Rebuild carry packet after writing memory state.
        carry_packet = self.carry_builder.build(
            session_id=self.session_id,
            recent_events=self.storage.read_recent_events(limit=60),
            new_events=[],
            alias_memory=self.storage.read_alias_memory(),
            correction_registry=self.storage.read_correction_registry(),
        )
        self.storage.write_carry_packet(carry_packet)

        # B-lite answer mode: only append caution when governance flags are present.
        cautions = []
        cautions.extend(sieve_result.get("required_cautions", []) or [])
        cautions.extend(self.governor.required_cautions_from_events(events))
        cautions = self._dedupe(cautions)
        calibrated_answer = self._append_caution_block(final_answer, cautions)

        return {
            "answer": calibrated_answer,
            "raw_answer": final_answer,
            "trace": trace,
            "image_paths": image_paths,
            "latency_seconds": task_result.get("latency_seconds"),
            "estimated_response_tokens": task_result.get("estimated_response_tokens"),
            "success": task_result.get("success"),
            "error": task_result.get("error"),
            "task_id": task_id,
            "archive_id": archive_id,
            "candidate_count": len(candidates),
            "memory_event_count": len(events),
            "carry_packet_token_estimate": carry_packet.token_estimate,
            "storage_dir": str(self.storage.session_dir),
            # Rehydration diagnostics
            "rehydration_triggered": rehydration_bundle is not None,
            "rehydrated_archive_refs": (
                rehydration_bundle.archive_refs if rehydration_bundle else []
            ),
            
            "sieve_raw_output": sieve_result.get("raw_worker_output"),
            "sieve_parsed_output": sieve_result.get("parsed_worker_output"),
            "sieve_parse_success": sieve_result.get("parse_success"),
            "sieve_fallback_used": sieve_result.get("fallback_used"),
            "cautions": cautions,
        }

    def _build_brain1_history(
        self,
        *,
        carry_packet: str,
        rehydrated_evidence: str | None,
        recent_display_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []

        if rehydrated_evidence and rehydrated_evidence.strip():
            history.append({
                "role": "system",
                "content": (
                    "ChronoSieve has rehydrated archived evidence for this user request. "
                    "For audit/recall questions, this archived evidence is the source of truth. "
                    "If it directly answers the request, answer from it and do not call tools just to rediscover the same evidence.\n\n"
                    + rehydrated_evidence
                ),
            })
        if carry_packet.strip():
            history.append({
                "role": "system",
                "content": (
                    "ChronoSieve Carry Packet for continuity. This is selected memory, not a full transcript.\n\n"
                    + carry_packet
                ),
            })

        # Only keep most recent display context, not full Streamlit history.
        for msg in recent_display_history[-4:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role in {"user", "assistant", "system"} and content:
                history.append({"role": role, "content": content})

        return history

    def _build_ledger_entry(
        self,
        *,
        task_id: str,
        user_request: str,
        archive_id: str,
        final_answer: str,
        task_summary: str,
        events: list,
    ) -> TaskLedgerEntry:
        active_findings: list[str] = []
        caveats: list[str] = []
        uncertainties: list[str] = []
        non_authoritative_items: list[str] = []

        for event in events:
            status = event.decision.status
            ctype = event.candidate.candidate_type
            text = event.decision.carry_text or event.candidate.content
            text = " ".join(str(text).split())[:700]

            if status in {"fossil", "active"} and ctype != "caveats":
                active_findings.append(text)
            if ctype == "caveats" and status in {"fossil", "active"}:
                caveats.append(text)
            if ctype == "uncertainties":
                uncertainties.append(text)
            if status == "non_authoritative":
                non_authoritative_items.append(text)

        return TaskLedgerEntry(
            task_id=task_id,
            session_id=self.session_id,
            user_request=user_request,
            task_summary=task_summary,
            status="completed",
            evidence_refs=[archive_id],
            active_findings=self._dedupe(active_findings)[-20:],
            caveats=self._dedupe(caveats)[-20:],
            uncertainties=self._dedupe(uncertainties)[-20:],
            non_authoritative_items=self._dedupe(non_authoritative_items)[-20:],
            metadata={
                "final_answer_preview": final_answer[:1000],
            },
        )

    @staticmethod
    def _append_caution_block(answer: str, cautions: list[str]) -> str:
        if not cautions:
            return answer

        lines = [answer.strip(), "", "---", "", "**ChronoSieve caution**"]
        for caution in cautions[:5]:
            lines.append(f"- {caution}")
        return "\n".join(lines).strip()

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
