from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.chronosieve.core.resource_adapter import ResourceRef, ObservationPacket
from src.chronosieve.core.schemas import MemoryCandidate, new_id
from src.chronosieve.core.utils import estimate_tokens, safe_str


@dataclass
class SyntheticRecord:
    record_id: str
    text: str
    metadata: dict[str, Any]


class SyntheticEvidenceAdapter:
    """
    Tiny generic adapter for smoke-testing the autonomous executor without DICOM.

    It intentionally includes:
    - stable facts
    - duplicate/noisy records
    - a correction
    - a proxy-risk record
    - an audit target

    This lets us test the loop before wiring local patient files.
    """

    adapter_name = "synthetic_evidence_adapter_v0"

    def __init__(self, records: list[SyntheticRecord] | None = None):
        self.records = records or self.default_records()
        self.record_by_id = {r.record_id: r for r in self.records}

    def list_resources(self, *, goal: str) -> list[ResourceRef]:
        return [
            ResourceRef(
                resource_id=r.record_id,
                uri=f"synthetic://{r.record_id}",
                resource_type="synthetic_record",
                title=r.metadata.get("title", r.record_id),
                metadata=r.metadata,
                cost_hint=1,
                priority_hint=int(r.metadata.get("priority_hint", 0)),
            )
            for r in self.records
        ]

    def describe_capabilities(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "inspect_resource",
                "purpose": "Read one synthetic evidence record and emit generic observation.",
                "cost_class": "low",
                "risk_class": "low",
            }
        ]

    def inspect_resource(self, *, resource: ResourceRef, instruction: str) -> ObservationPacket:
        record = self.record_by_id[resource.resource_id]
        warnings: list[str] = []

        if record.metadata.get("proxy_risk"):
            warnings.append("Proxy/substitute result risk present in this record.")
        if record.metadata.get("correction"):
            warnings.append("This record corrects or supersedes an earlier claim.")

        summary = f"Synthetic evidence record {record.record_id}: {record.text}"
        return ObservationPacket(
            resource_id=resource.resource_id,
            action="inspect_resource",
            summary=summary,
            raw_text=record.text,
            structured={
                "record_id": record.record_id,
                "metadata": record.metadata,
            },
            artifact_refs=[resource.uri],
            warnings=warnings,
            error=None,
            token_estimate=estimate_tokens(summary + json.dumps(record.metadata, default=str)),
        )

    def candidates_from_observation(
        self,
        *,
        goal: str,
        task_id: str,
        archive_id: str,
        observation: ObservationPacket,
        resource: ResourceRef,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        metadata = observation.structured.get("metadata", {}) or {}

        candidates.append(MemoryCandidate(
            candidate_id=new_id("cand"),
            candidate_type="artifacts",
            content=f"Evidence artifact inspected: {resource.uri}",
            source_type="deterministic_extractor",
            evidence_ref=archive_id,
            task_id=task_id,
            metadata={"artifact_type": resource.resource_type, "path": resource.uri},
            audit_value=3,
            token_cost=estimate_tokens(resource.uri),
        ))

        if metadata.get("fact_key"):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="observations",
                content=f"Resolved fact {metadata.get('fact_key')}: {metadata.get('fact_value')}",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={
                    "field": metadata.get("fact_key"),
                    "value": metadata.get("fact_value"),
                    "resource_id": resource.resource_id,
                },
                evidence_strength=3,
                future_utility=3,
                audit_value=3,
                token_cost=estimate_tokens(str(metadata.get("fact_value"))),
            ))

        if metadata.get("correction"):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="corrections",
                content=(
                    f"Correction observed: {metadata.get('old_state')} -> {metadata.get('new_state')}. "
                    f"Reason: {metadata.get('reason', '')}"
                ),
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={
                    "old_state": metadata.get("old_state"),
                    "new_state": metadata.get("new_state"),
                    "reason": metadata.get("reason"),
                },
                evidence_strength=3,
                correction_value=3,
                future_utility=3,
                audit_value=3,
                token_cost=estimate_tokens(observation.summary),
            ))

        if metadata.get("proxy_risk"):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="uncertainties",
                content=(
                    "Possible non-equivalent proxy substitution: requested concept "
                    f"{metadata.get('requested_concept')} but observed proxy concept "
                    f"{metadata.get('proxy_concept')}. Must not become authoritative for original task."
                ),
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={
                    "proxy_risk": True,
                    "requested_concept": metadata.get("requested_concept"),
                    "proxy_concept": metadata.get("proxy_concept"),
                },
                evidence_strength=2,
                future_utility=3,
                audit_value=3,
                correction_value=3,
                risk_if_wrong=3,
                token_cost=estimate_tokens(observation.summary),
            ))

        if metadata.get("claim"):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="claims",
                content=f"Record claim: {metadata.get('claim')}",
                source_type="task_answer",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"resource_id": resource.resource_id},
                evidence_strength=1,
                future_utility=2,
                audit_value=2,
                risk_if_wrong=2,
                token_cost=estimate_tokens(str(metadata.get("claim"))),
            ))

        if observation.warnings:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="caveats",
                content="; ".join(observation.warnings),
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"warnings": observation.warnings},
                evidence_strength=2,
                future_utility=3,
                audit_value=3,
                risk_if_wrong=3,
                token_cost=estimate_tokens("; ".join(observation.warnings)),
            ))

        if not candidates:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="tool_outputs",
                content=safe_str(observation.to_dict(), 1500),
                source_type="tool_trace",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"resource_id": resource.resource_id},
                audit_value=2,
                token_cost=observation.token_estimate,
            ))

        return candidates

    @staticmethod
    def default_records() -> list[SyntheticRecord]:
        return [
            SyntheticRecord(
                record_id="rec_001",
                text="Case ALPHA has canonical entity ID PATIENT-ALPHA and baseline date 2024-01-01.",
                metadata={
                    "title": "stable entity fact",
                    "fact_key": "baseline_date",
                    "fact_value": "2024-01-01",
                    "priority_hint": 3,
                },
            ),
            SyntheticRecord(
                record_id="rec_002",
                text="A model interpretation says the case is improving, but this is based on one weak observation.",
                metadata={
                    "title": "weak interpretation",
                    "claim": "Case ALPHA is improving based on one weak observation.",
                    "priority_hint": 1,
                },
            ),
            SyntheticRecord(
                record_id="rec_003",
                text="Tool failed for requested metric X, but returned proxy metric Y = 42.",
                metadata={
                    "title": "proxy trap",
                    "proxy_risk": True,
                    "requested_concept": "metric X",
                    "proxy_concept": "metric Y",
                    "claim": "Metric X is 42.",
                    "priority_hint": 3,
                },
            ),
            SyntheticRecord(
                record_id="rec_004",
                text="Correction: baseline date 2024-01-01 is superseded by verified baseline date 2023-12-15.",
                metadata={
                    "title": "correction",
                    "correction": True,
                    "old_state": "baseline date = 2024-01-01",
                    "new_state": "baseline date = 2023-12-15",
                    "reason": "Verified source supersedes earlier record.",
                    "fact_key": "baseline_date",
                    "fact_value": "2023-12-15",
                    "priority_hint": 3,
                },
            ),
            SyntheticRecord(
                record_id="rec_005",
                text="Duplicate low-value note repeats that Case ALPHA exists.",
                metadata={
                    "title": "duplicate noise",
                    "claim": "Case ALPHA exists.",
                    "priority_hint": 0,
                },
            ),
        ]
