from __future__ import annotations

import json
import re
from typing import Any

from src.chronosieve.core.schemas import MemoryCandidate, new_id
from src.chronosieve.core.utils import estimate_tokens, safe_str


PNG_RE = re.compile(r"[\w/\\.-]*scan_[a-f0-9]{6,16}\.png")
DCM_RE = re.compile(r"[\w/\\.-]+\.dcm")
PATIENT_RE = re.compile(r"UPENN-GBM-\d{5}")
DATE_RE = re.compile(r"\b(20\d{6}|19\d{6})\b")


class MikeTraceParser:
    """
    Mike-specific adapter.

    This file is allowed to know about Mike trace shapes, DICOM paths, PNG paths,
    patient IDs, modalities, and Z-coordinates.

    It converts Mike-specific outputs into generic ChronoSieve MemoryCandidate
    objects. The ChronoSieve core should remain domain-neutral.
    """

    def parse_turn(
        self,
        *,
        task_id: str,
        archive_id: str,
        user_request: str,
        final_answer: str,
        trace: list[dict[str, Any]],
        image_paths: list[str] | None = None,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        image_paths = image_paths or []

        candidates.append(MemoryCandidate(
            candidate_id=new_id("cand"),
            candidate_type="task_intent",
            content=f"User requested: {user_request}",
            source_type="user_request",
            evidence_ref=archive_id,
            task_id=task_id,
            metadata={"raw_user_request": user_request},
            token_cost=estimate_tokens(user_request),
        ))

        if final_answer:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="claims",
                content=f"Task agent final answer: {final_answer}",
                source_type="task_answer",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"answer_char_count": len(final_answer)},
                token_cost=estimate_tokens(final_answer),
            ))

        tool_names = [str(step.get("tool", "unknown")) for step in trace]
        if tool_names:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="tools_used",
                content="Tools used in order: " + " -> ".join(tool_names),
                source_type="tool_trace",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"tools_used": tool_names},
                token_cost=estimate_tokens(" ".join(tool_names)),
            ))

        for idx, step in enumerate(trace, start=1):
            candidates.extend(self._parse_step(
                task_id=task_id,
                archive_id=archive_id,
                step_index=idx,
                step=step,
            ))

        for path in image_paths:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="artifacts",
                content=f"Generated or referenced image artifact: {path}",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"artifact_type": "png", "path": path},
                audit_value=3,
                token_cost=estimate_tokens(path),
            ))

        candidates.extend(self._detect_alias_learnings(
            task_id=task_id,
            archive_id=archive_id,
            trace=trace,
        ))

        candidates.extend(self._detect_proxy_risk(
            task_id=task_id,
            archive_id=archive_id,
            user_request=user_request,
            final_answer=final_answer,
            trace=trace,
        ))

        candidates.extend(self._detect_caveat_candidates(
            task_id=task_id,
            archive_id=archive_id,
            final_answer=final_answer,
            trace=trace,
        ))

        return candidates

    def _parse_step(
        self,
        *,
        task_id: str,
        archive_id: str,
        step_index: int,
        step: dict[str, Any],
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        tool_name = str(step.get("tool", "unknown_tool"))
        tool_input = step.get("tool_input", {})
        observation = str(step.get("observation", ""))
        obs_json = self._try_json(observation)

        candidates.append(MemoryCandidate(
            candidate_id=new_id("cand"),
            candidate_type="tool_outputs",
            content=(
                f"Tool step {step_index}: {tool_name}\n"
                f"Input: {safe_str(tool_input, 1200)}\n"
                f"Observation: {safe_str(observation, 1800)}"
            ),
            source_type="tool_trace",
            evidence_ref=archive_id,
            task_id=task_id,
            metadata={
                "step_index": step_index,
                "tool": tool_name,
                "tool_input": tool_input,
                "observation_json": obs_json,
                "is_error": observation.lower().startswith("error") or "error:" in observation.lower(),
            },
            evidence_strength=3,
            audit_value=3,
            token_cost=estimate_tokens(observation),
        ))

        # Extract artifact/file-level evidence from observations.
        for dcm_path in sorted(set(DCM_RE.findall(observation))):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="artifacts",
                content=f"DICOM artifact path observed: {dcm_path}",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"artifact_type": "dicom", "path": dcm_path, "step_index": step_index},
                audit_value=3,
                token_cost=estimate_tokens(dcm_path),
            ))

        for png_path in sorted(set(PNG_RE.findall(observation))):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="artifacts",
                content=f"PNG artifact path observed: {png_path}",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"artifact_type": "png", "path": png_path, "step_index": step_index},
                audit_value=3,
                token_cost=estimate_tokens(png_path),
            ))

        # Extract structured fields if tool returned JSON.
        if isinstance(obs_json, dict):
            if "canonical_id" in obs_json:
                candidates.append(MemoryCandidate(
                    candidate_id=new_id("cand"),
                    candidate_type="observations",
                    content=f"Canonical entity ID resolved: {obs_json.get('canonical_id')}",
                    source_type="deterministic_extractor",
                    evidence_ref=archive_id,
                    task_id=task_id,
                    metadata={"field": "canonical_id", "value": obs_json.get("canonical_id"), "step_index": step_index},
                    evidence_strength=3,
                    future_utility=3,
                    audit_value=2,
                    token_cost=estimate_tokens(str(obs_json.get("canonical_id"))),
                ))

            if "z_coordinate" in obs_json:
                candidates.append(MemoryCandidate(
                    candidate_id=new_id("cand"),
                    candidate_type="observations",
                    content=f"Physical Z-coordinate selected: {obs_json.get('z_coordinate')}",
                    source_type="deterministic_extractor",
                    evidence_ref=archive_id,
                    task_id=task_id,
                    metadata={"field": "z_coordinate", "value": obs_json.get("z_coordinate"), "step_index": step_index},
                    evidence_strength=3,
                    future_utility=3,
                    audit_value=3,
                    token_cost=estimate_tokens(str(obs_json.get("z_coordinate"))),
                ))

            if "file_path" in obs_json:
                candidates.append(MemoryCandidate(
                    candidate_id=new_id("cand"),
                    candidate_type="artifacts",
                    content=f"File path selected by tool: {obs_json.get('file_path')}",
                    source_type="deterministic_extractor",
                    evidence_ref=archive_id,
                    task_id=task_id,
                    metadata={"artifact_type": "dicom", "path": obs_json.get("file_path"), "step_index": step_index},
                    evidence_strength=3,
                    audit_value=3,
                    token_cost=estimate_tokens(str(obs_json.get("file_path"))),
                ))

            if "png_path" in obs_json:
                candidates.append(MemoryCandidate(
                    candidate_id=new_id("cand"),
                    candidate_type="artifacts",
                    content=f"PNG path generated by tool: {obs_json.get('png_path')}",
                    source_type="deterministic_extractor",
                    evidence_ref=archive_id,
                    task_id=task_id,
                    metadata={"artifact_type": "png", "path": obs_json.get("png_path"), "step_index": step_index},
                    evidence_strength=3,
                    audit_value=3,
                    token_cost=estimate_tokens(str(obs_json.get("png_path"))),
                ))

        # Extract patient IDs and dates even when not JSON.
        for patient_id in sorted(set(PATIENT_RE.findall(observation))):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="observations",
                content=f"Patient/entity ID mentioned in trace: {patient_id}",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"field": "patient_id", "value": patient_id, "step_index": step_index},
                token_cost=estimate_tokens(patient_id),
            ))

        for date in sorted(set(DATE_RE.findall(observation))):
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="observations",
                content=f"Study/date-like value mentioned in trace: {date}",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"field": "date", "value": date, "step_index": step_index},
                token_cost=estimate_tokens(date),
            ))

        return candidates

    def _detect_alias_learnings(
        self,
        *,
        task_id: str,
        archive_id: str,
        trace: list[dict[str, Any]],
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        failed_modalities: set[str] = set()
        successful_modalities: set[str] = set()

        for step in trace:
            tool = str(step.get("tool", ""))
            tool_input = step.get("tool_input", {}) or {}
            obs = str(step.get("observation", ""))
            modality = None
            if isinstance(tool_input, dict):
                modality = tool_input.get("modality")

            if tool == "get_anatomically_aligned_slice" and modality:
                if "Error:" in obs or "not found" in obs.lower():
                    failed_modalities.add(str(modality))
                else:
                    successful_modalities.add(str(modality))

        # v1 practical alias inference. Generic as "procedure/learning" but Mike-specific metadata.
        if "T1-Post" in failed_modalities and "T1-Post Contrast" in successful_modalities:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="learnings",
                content="Validated procedural learning: normalize modality alias 'T1-Post' to canonical 'T1-Post Contrast' for this dataset/tool layer.",
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={
                    "learning_type": "alias_normalization",
                    "alias": "T1-Post",
                    "canonical": "T1-Post Contrast",
                    "failed_modalities": sorted(failed_modalities),
                    "successful_modalities": sorted(successful_modalities),
                },
                evidence_strength=3,
                future_utility=3,
                correction_value=3,
                audit_value=2,
                token_cost=24,
            ))

        return candidates

    def _detect_proxy_risk(
        self,
        *,
        task_id: str,
        archive_id: str,
        user_request: str,
        final_answer: str,
        trace: list[dict[str, Any]],
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        request_l = user_request.lower()
        answer_l = final_answer.lower()
        trace_text = "\n".join(str(step.get("observation", "")) for step in trace).lower()

        # v1 detection for the known baseline failure pattern, stored in generic terms.
        if "motor cortex" in request_l and "perf" in (answer_l + trace_text):
            failed_requested = "error" in trace_text or "failed" in answer_l or "sequence mismatch" in trace_text
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="uncertainties",
                content=(
                    "Possible non-equivalent proxy substitution: user requested motor cortex displacement, "
                    "but trace/answer references perfusion/perf displacement. Proxy result must not become authoritative for the requested task."
                ),
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={
                    "proxy_risk": True,
                    "requested_concept": "motor cortex displacement",
                    "proxy_concept": "perfusion/perf displacement",
                    "requested_task_failed": failed_requested,
                },
                evidence_strength=2,
                future_utility=3,
                audit_value=3,
                correction_value=3,
                risk_if_wrong=3,
                token_cost=45,
            ))

        return candidates

    def _detect_caveat_candidates(
        self,
        *,
        task_id: str,
        archive_id: str,
        final_answer: str,
        trace: list[dict[str, Any]],
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        trace_text = "\n".join(str(step.get("observation", "")) for step in trace)
        combined_l = (final_answer + "\n" + trace_text).lower()

        caveat_markers = [
            "single slice",
            "full-series",
            "full series",
            "volumetric",
            "requires",
            "specialized radiology",
            "not equivalent",
            "not sufficient",
            "cannot automatically",
        ]

        matched = [m for m in caveat_markers if m in combined_l]
        if matched:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="caveats",
                content=(
                    "Potential caveat/limitation detected in trace or answer. "
                    f"Matched markers: {', '.join(matched)}. Preserve relevant limitations if future claims depend on this task."
                ),
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={"matched_caveat_markers": matched},
                evidence_strength=2,
                future_utility=3,
                audit_value=3,
                risk_if_wrong=3,
                token_cost=35,
            ))

        return candidates

    @staticmethod
    def _try_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None
