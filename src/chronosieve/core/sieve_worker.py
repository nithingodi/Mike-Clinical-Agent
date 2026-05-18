from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .schemas import MemoryCandidate, SieveDecision, new_id
from .utils import extract_json_object, safe_str
from .role_briefing import CHRONOSIEVE_ROLE_BRIEFING

VALID_STATUSES = {
    "fossil",
    "active",
    "archived",
    "decay",
    "invalidated",
    "non_authoritative",
}


class GemmaSieveWorker:
    """
    Brain 2: Gemma-guided memory governance.

    Fix in this version:
    - Brain 2 receives compact generic candidate briefings, not raw trace-heavy objects.
    - Candidates are keyed as C1/C2/C3 so Gemma does not need to copy fragile UUIDs.
    - Raw/parsed worker output is returned for audit logging.
    - If Gemma fails to classify candidates, fallback is typed, not archive-everything.
    - Output is constrained to compact JSON to reduce truncation risk on large candidate sets.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    def decide(
        self,
        *,
        user_request: str,
        final_answer: str,
        candidates: list[MemoryCandidate],
        prior_carry_packet: str,
        recent_ledger: list[dict[str, Any]] | None = None,
        alias_memory: dict[str, Any] | None = None,
        correction_registry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        briefing_rows, key_to_candidate_id = self._build_candidate_briefings(candidates)

        payload = self._build_payload(
            user_request=user_request,
            final_answer=final_answer,
            candidate_briefings=briefing_rows,
            prior_carry_packet=prior_carry_packet,
            recent_ledger=recent_ledger or [],
            alias_memory=alias_memory or {"aliases": []},
            correction_registry=correction_registry or {"corrections": []},
        )

        system = SystemMessage(content=self._system_prompt())
        human = HumanMessage(content=json.dumps(payload, indent=2, ensure_ascii=False, default=str))

        try:
            response = self.llm.invoke([system, human])
            raw = getattr(response, "content", str(response))
            worker_error = None
        except Exception as exc:
            raw = json.dumps({
                "task_summary": "Sieve worker failed; using typed fallback decisions.",
                "required_cautions": [f"Sieve worker error: {exc}"],
                "decisions": [],
            })
            worker_error = str(exc)

        parsed = extract_json_object(raw)
        decisions = self._parse_decisions(
            parsed=parsed,
            candidates=candidates,
            key_to_candidate_id=key_to_candidate_id,
        )

        parse_success = bool(parsed) and bool(parsed.get("decisions"))
        fallback_used = not parse_success or not decisions

        if fallback_used:
            decisions = self._typed_fallback_decisions(candidates)

        return {
            "task_summary": parsed.get("task_summary") or "ChronoSieve turn processed.",
            "decisions": decisions,
            "required_cautions": parsed.get("required_cautions") or [],
            "correction_records": parsed.get("correction_records") or [],
            "raw_worker_output": raw,
            "parsed_worker_output": parsed,
            "candidate_briefings": briefing_rows,
            "parse_success": parse_success,
            "fallback_used": fallback_used,
            "worker_error": worker_error,
            "decision_count": len(decisions),
        }

    def _build_payload(
        self,
        *,
        user_request: str,
        final_answer: str,
        candidate_briefings: list[dict[str, Any]],
        prior_carry_packet: str,
        recent_ledger: list[dict[str, Any]],
        alias_memory: dict[str, Any],
        correction_registry: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "chronosieve_role": "memory_governance_not_task_solving",
            "user_request": safe_str(user_request, max_chars=1200),
            "task_agent_final_answer": safe_str(final_answer, max_chars=1800),
            "prior_carry_packet": safe_str(prior_carry_packet, max_chars=2500),
            "recent_ledger": recent_ledger[-6:],
            "alias_memory": alias_memory,
            "correction_registry": correction_registry,
            "candidate_briefings": candidate_briefings,
            "allowed_statuses": sorted(VALID_STATUSES),
            "decision_goal": (
                "For each compact candidate briefing, decide what future attention should do with it. "
                "Use the candidate key C1/C2/etc. Do not copy internal UUIDs. Do not solve the original task."
            ),
            "output_constraints": {
                "reason_max_words": 12,
                "carry_text_max_chars": 180,
                "rehydration_triggers_max_items": 2,
                "omit_carry_text_when_status_in": ["archived", "decay"],
                "prefer_empty_rehydration_triggers_unless_audit_value_is_high": True,
            },
        }

    def _build_candidate_briefings(
        self,
        candidates: list[MemoryCandidate],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        rows: list[dict[str, Any]] = []
        key_to_candidate_id: dict[str, str] = {}

        for idx, c in enumerate(candidates, start=1):
            key = f"C{idx}"
            key_to_candidate_id[key] = c.candidate_id
            generic_role = self._infer_generic_role(c)

            rows.append({
                "key": key,
                "candidate_type": c.candidate_type,
                "generic_role": generic_role,
                "summary": self._generic_summary(c, generic_role),
                "source_type": c.source_type,
                "evidence_ref": c.evidence_ref,
                "score_hints": {
                    "evidence_strength": c.evidence_strength,
                    "future_utility": c.future_utility,
                    "audit_value": c.audit_value,
                    "correction_value": c.correction_value,
                    "risk_if_wrong": c.risk_if_wrong,
                    "token_cost": c.token_cost,
                    "redundancy": c.redundancy,
                },
                "metadata_summary": self._metadata_summary(c.metadata),
            })

        return rows, key_to_candidate_id

    def _infer_generic_role(self, c: MemoryCandidate) -> str:
        metadata = c.metadata or {}

        if c.candidate_type == "task_intent":
            return "requested_intent"
        if c.candidate_type == "tools_used":
            return "tool_sequence"
        if c.candidate_type == "artifacts":
            return "external_evidence_artifact"
        if c.candidate_type == "caveats":
            return "evidence_limitation"
        if c.candidate_type == "uncertainties":
            if metadata.get("proxy_risk"):
                return "proxy_or_substitute_result"
            return "uncertainty"
        if c.candidate_type == "learnings":
            return "procedural_learning"
        if c.candidate_type == "corrections":
            return "correction"
        if c.candidate_type == "claims":
            return "task_agent_claim"
        if c.candidate_type == "interpretations":
            return "model_interpretation"
        if c.candidate_type == "observations":
            field = str(metadata.get("field", ""))
            if field in {"canonical_id", "patient_id", "entity_id"}:
                return "resolved_entity"
            if field or "value" in metadata:
                return "resolved_parameter"
            return "model_or_tool_observation"
        if c.candidate_type == "tool_outputs":
            if metadata.get("is_error"):
                return "tool_failure"
            return "tool_result_summary"
        return "memory_candidate"

    def _generic_summary(self, c: MemoryCandidate, generic_role: str) -> str:
        metadata = c.metadata or {}

        # Avoid sending raw long payload to Brain 2. Keep the memory consequence visible.
        if generic_role == "external_evidence_artifact":
            artifact_type = metadata.get("artifact_type", "artifact")
            return f"An external evidence artifact was produced or selected. Artifact type: {artifact_type}. Full value is archived."

        if generic_role == "resolved_entity":
            value = metadata.get("value") or safe_str(c.content, 120)
            return f"A canonical or task-relevant entity was resolved: {value}."

        if generic_role == "resolved_parameter":
            field = metadata.get("field", "parameter")
            value = metadata.get("value")
            source_bits = []
            
            for key in ["resource_id", "source_ref", "evidence_id", "turn_id", "session_date_time", "timestamp", "speaker"]:
                if metadata.get(key):
                    source_bits.append(f"{key}={metadata.get(key)}")
                source_note = ""
                if source_bits:
                    source_note = " Source metadata: " + "; ".join(source_bits) + "."
                return (f"A task-specific source-grounded parameter/observation was resolved by evidence. "
                f"Field: {field}. Value preview: {value}.{source_note}"
                )

        if generic_role == "tool_sequence":
            tools = metadata.get("tools_used")
            return f"Tool sequence used by the task agent: {tools}. Useful for audit, usually not active memory."

        if generic_role == "tool_result_summary":
            tool = metadata.get("tool")
            is_error = metadata.get("is_error")
            return f"A tool returned a result. Tool: {tool}. Error: {is_error}. Raw output is archived."

        if generic_role == "tool_failure":
            tool = metadata.get("tool")
            return f"A tool failed or returned an error. Tool: {tool}. Failure details are archived."

        if generic_role == "procedural_learning":
            return safe_str(c.content, 500)

        if generic_role == "proxy_or_substitute_result":
            return safe_str(c.content, 500)

        if generic_role == "evidence_limitation":
            return safe_str(c.content, 500)

        if generic_role in {"task_agent_claim", "model_interpretation", "model_or_tool_observation", "requested_intent", "uncertainty", "correction"}:
            return safe_str(c.content, 500)

        return safe_str(c.content, 400)

    @staticmethod
    def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
        if not metadata:
            return {}
        allowed_keys = [
    "field",
    "value",
    "tool",
    "step_index",
    "is_error",
    "artifact_type",
    "learning_type",
    "proxy_risk",
    "requested_concept",
    "proxy_concept",

    # Generic source/evidence metadata.
    "resource_id",
    "resource_uri",
    "resource_type",
    "source_id",
    "source_ref",
    "evidence_id",
    "turn_id",
    "timestamp",
    "date",
    "session_date_time",
    "speaker",
    "path",

    # Generic salience hints from adapters.
    "salient_conversation_turn",
    "contains_temporal_expression",
]
        return {k: metadata.get(k) for k in allowed_keys if k in metadata}

    def _system_prompt(self) -> str:
        return (
        CHRONOSIEVE_ROLE_BRIEFING
        + """

# Your Current Role: Brain 2 / Sieve Worker

You are Actor 6: the memory-governance judge.

You are not Brain 1.
You are not the ResourceAdapter.
You are not the PolicyGovernor.
You are not the MemoryAccessController.
You are not the AnswerSynthesizer.
You do not answer the user's task.

Your job is to inspect compact MemoryCandidate briefings and decide what future attention should do with each candidate.

## Inputs You Receive

You receive:
1. user_request
   The original task or broad goal.

2. task_agent_final_answer
   Brain 1's answer or the current observation summary.
   This may contain useful claims, but Brain 1 claims are not automatically truth.

3. prior_carry_packet
   Existing governed memory selected from earlier turns.
   This is not a transcript.

4. recent_ledger
   Compact task-level history.

5. alias_memory
   Validated aliases or procedural mappings.

6. correction_registry
   Corrections, invalidations, supersessions, and blocked states.

7. candidate_briefings
   Compact candidate records keyed as C1, C2, C3...
   These are proposed memory items, not final memory.

## Your Output

For every candidate, assign exactly one memory status:

- fossil
  Durable protected memory. Use for stable goals, validated rules, high-value corrections,
  safety caveats, alias/procedural learnings, or durable facts that should survive long horizon.

- active
  Useful working memory for future task continuity. Use for important facts, entities,
  dates, preferences, plans, source-grounded observations, unresolved questions, and
  bounded details likely to be needed again.

- archived
  Keep externally for audit/source recall, but do not carry in active context.

- decay
  Low-value, noisy, redundant, or temporary residue. It may remain in archive but should
  not consume future attention.

- invalidated
  Contradicted, corrected, or superseded memory. It must not be reused as truth.

- non_authoritative
  Useful as a weak claim, model interpretation, proxy result, or warning-only item.
  It must not be treated as authoritative truth without revalidation.

## Authority Rules

Use these authority rules carefully:

1. Deterministic observations with explicit evidence/source refs can become active if useful.
   They do not need to be fossil unless they are durable anchors, corrections, rules, or guardrails.

2. Brain 1 claims and interpretations are not automatically authoritative.
   They should usually be non_authoritative unless separately grounded by deterministic evidence.

3. Artifacts and raw payloads are usually archived or external-only.
   Carry only compact handles or facts derived from them, not large raw payloads.

4. Caveats that prevent future overconfidence should survive.
   High-risk caveats should be active or fossil.

5. Proxy/substitute results must not become original-task truth.
   They should be non_authoritative or invalidated depending on context.

6. Corrections, supersessions, and invalidations are high-value.
   Preserve the new state and block the old unsafe state.

7. Stable aliases, canonical IDs, procedural learnings, and validated normalization rules
   can become fossil because they prevent repeated future errors.

8. Do not drop useful source-grounded details only because they are early or small.
   Long-horizon work often depends on small facts, timestamps, exact values, and evidence handles.

9. Do not carry everything.
   If a fact is useful only for audit, mark it archived with good rehydration triggers.

## Carry Text Rules

carry_text should be:
- short
- self-contained
- useful to a future model
- authority-aware
- source-aware when possible

Good carry_text examples:
- "Source-grounded observation: entity X had baseline date 2023-12-15. Evidence: archive_abc."
- "Warning: metric Y was only a proxy for requested metric X; do not treat it as X."
- "Correction: old state A was superseded by verified state B."

Bad carry_text examples:
- huge raw JSON payloads
- entire transcripts
- vague text like "important info"
- unsupported conclusions
- artifact paths without why they matter

## Rehydration Trigger Rules

rehydration_triggers should tell future ChronoSieve when archived evidence should be temporarily loaded.

Good triggers:
- "when asked for exact source/evidence"
- "when asked about this entity/date/value"
- "when resolving a contradiction"
- "when user asks why/how this was concluded"
- "when exact artifact/path/timestamp is needed"

## KV Future Hint Rules

Use backend-neutral hints. Good examples:
- protected_reanchor
- keep_current
- external_only
- keep_until_resolved
- keep_until_claim_resolved
- block_truth
- load_with_warning
- do_not_fossilize
- evict_when_irrelevant
- temporary_only

## Decision Standard

For each candidate, ask:

1. Is this deterministic evidence, model interpretation, artifact, caveat, correction, or noise?
2. Would losing this harm future continuity, auditability, safety, or correctness?
3. Should it be active memory, protected memory, external archive, warning-only, blocked, or decayed?
4. What exact future question would need this?
5. What runtime action should this imply later?

## Forbidden Behavior

- Do not solve the user's task again.
- Do not answer benchmark questions.
- Do not invent facts.
- Do not invent candidate keys.
- Do not invent archive ids.
- Do not promote weak interpretations into truth.
- Do not mark raw payloads as fossil.
- Do not ignore correction_registry or invalidation context.

## Required JSON Shape

Return JSON only.

Use candidate keys like C1, C2, C3. Do not copy or invent internal UUIDs.

{
  "task_summary": "compact summary of what happened in this turn",
  "required_cautions": ["caution to preserve, if any"],
  "correction_records": [
    {
      "correction_type": "factual_correction|reference_frame_change|task_scope_change|preference_correction|invalidation|supersession|conceptual_correction|architectural_correction",
      "old_state": "optional old state",
      "new_state": "new corrected state",
      "reason": "why this correction matters",
      "affected_keys": ["C1"]
    }
  ],
  "decisions": [
    {
      "key": "C1",
      "status": "fossil|active|archived|decay|invalidated|non_authoritative",
      "reason": "why this memory status is appropriate",
      "carry_text": "short text allowed into future carry/runtime memory, or null",
      "rehydration_triggers": ["when to bring archived evidence back"],
      "kv_future_hint": "future runtime behavior hint",
      "confidence": 0.0
    }
  ]
}
""".strip()
    )

    def _parse_decisions(
        self,
        *,
        parsed: dict[str, Any],
        candidates: list[MemoryCandidate],
        key_to_candidate_id: dict[str, str],
    ) -> list[SieveDecision]:
        candidate_by_id = {c.candidate_id: c for c in candidates}
        seen_candidate_ids: set[str] = set()
        decisions: list[SieveDecision] = []

        for row in parsed.get("decisions", []) or []:
            key = row.get("key")
            candidate_id = key_to_candidate_id.get(str(key))

            # Backward-compatible: allow candidate_id if Gemma returned it anyway.
            if not candidate_id:
                raw_candidate_id = row.get("candidate_id")
                if raw_candidate_id in candidate_by_id:
                    candidate_id = raw_candidate_id

            if not candidate_id or candidate_id not in candidate_by_id:
                continue

            status = row.get("status", "archived")
            if status not in VALID_STATUSES:
                status = "archived"

            candidate = candidate_by_id[candidate_id]
            decisions.append(SieveDecision(
                decision_id=new_id("dec"),
                candidate_id=candidate_id,
                status=status,  # type: ignore[arg-type]
                reason=str(row.get("reason") or "Gemma Sieve Brain recommended this status."),
                carry_text=row.get("carry_text"),
                archive_ref=candidate.evidence_ref,
                rehydration_triggers=list(row.get("rehydration_triggers") or []),
                kv_future_hint=row.get("kv_future_hint"),
                confidence=self._safe_confidence(row.get("confidence")),
            ))
            seen_candidate_ids.add(candidate_id)

        return decisions

    def _typed_fallback_decisions(self, candidates: list[MemoryCandidate]) -> list[SieveDecision]:
        """
        Safety fallback when Brain 2 returns malformed/empty decisions.
        This is generic and conservative, not domain-specific.
        """
        decisions: list[SieveDecision] = []

        for c in candidates:
            status = "archived"
            carry_text = None
            triggers = ["audit request", "user asks why/how this was concluded"]
            kv_hint = "external_only"
            reason = "Typed fallback: archived conservatively."

            if c.candidate_type == "learnings":
                status = "fossil"
                carry_text = safe_str(c.content, 500)
                kv_hint = "protected_reanchor"
                reason = "Typed fallback: procedural learning should survive to prevent repeated mistakes."

            elif c.candidate_type == "caveats":
                status = "active"
                carry_text = safe_str(c.content, 500)
                kv_hint = "keep_until_claim_resolved"
                reason = "Typed fallback: caveat/limitation should survive to prevent overconfidence."

            elif c.candidate_type == "uncertainties":
                if c.metadata.get("proxy_risk"):
                    status = "non_authoritative"
                    kv_hint = "block_truth"
                    reason = "Typed fallback: proxy/substitute risk must not become authoritative truth."
                else:
                    status = "active"
                    kv_hint = "keep_until_resolved"
                    reason = "Typed fallback: unresolved uncertainty should remain visible."
                carry_text = safe_str(c.content, 500)

            elif c.candidate_type == "observations" and (c.future_utility or 0) >= 3:
                status = "active"
                carry_text = safe_str(c.content, 400)
                kv_hint = "keep_current"
                reason = "Typed fallback: high-utility resolved observation should remain active."

            elif c.candidate_type == "claims":
                status = "non_authoritative"
                carry_text = "Task-agent final-answer claim is non-authoritative unless revalidated against deterministic evidence."
                kv_hint = "do_not_fossilize"
                reason = "Typed fallback: final-answer claim is not automatically authoritative without Sieve Brain judgment."

            elif c.candidate_type == "artifacts":
                status = "archived"
                carry_text = None
                kv_hint = "external_only"
                reason = "Typed fallback: artifacts are audit evidence; full values stay archived."

            elif c.candidate_type == "tool_outputs":
                status = "archived"
                carry_text = None
                kv_hint = "external_only"
                reason = "Typed fallback: raw tool output is payload; archive, do not carry."

            elif c.candidate_type == "task_intent":
                status = "active"
                carry_text = safe_str(c.content, 350)
                kv_hint = "keep_current"
                reason = "Typed fallback: task intent helps current continuity."

            decisions.append(SieveDecision(
                decision_id=new_id("dec"),
                candidate_id=c.candidate_id,
                status=status,  # type: ignore[arg-type]
                reason=reason,
                carry_text=carry_text,
                archive_ref=c.evidence_ref,
                rehydration_triggers=triggers,
                kv_future_hint=kv_hint,
                confidence=0.45,
            ))

        return decisions

    @staticmethod
    def _safe_confidence(value: Any) -> float | None:
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except Exception:
            return None
