from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

from src.chronosieve.core.resource_adapter import ResourceRef, ObservationPacket
from src.chronosieve.core.schemas import MemoryCandidate, new_id
from src.chronosieve.core.utils import estimate_tokens, safe_str


class DicomResourceAdapter:
    """
    Generic ResourceAdapter for DICOM files.

    Important boundary:
    - This adapter knows how to discover/read DICOM metadata.
    - It converts each DICOM into generic ResourceRef / ObservationPacket /
      MemoryCandidate objects.
    - It does NOT decide what should survive long-term memory. ChronoSieve
      Sieve + Governor decide that later.

    v0 scope:
    - metadata-only inspection
    - no pixel/vision analysis yet
    - balanced sampling for small investor demos such as 20 files
    """

    adapter_name = "dicom_resource_adapter_v0"

    def __init__(
        self,
        *,
        patient_root: str | Path,
        limit: int | None = 20,
        selection_strategy: str = "balanced",
    ):
        self.patient_root = Path(patient_root)
        self.limit = limit
        self.selection_strategy = selection_strategy
        self._resource_cache: list[ResourceRef] | None = None

    def list_resources(self, *, goal: str) -> list[ResourceRef]:
        if self._resource_cache is not None:
            return self._resource_cache

        if not self.patient_root.exists():
            raise FileNotFoundError(f"Patient root not found: {self.patient_root}")

        dcm_paths = sorted(self.patient_root.rglob("*.dcm"))
        resources: list[ResourceRef] = []

        for idx, path in enumerate(dcm_paths, start=1):
            metadata = self._read_metadata(path)
            resource_id = f"dcm_{idx:05d}"
            title = self._title_from_metadata(metadata, fallback=path.name)

            resources.append(ResourceRef(
                resource_id=resource_id,
                uri=str(path),
                resource_type="dicom_file",
                title=title,
                metadata=metadata,
                cost_hint=1,
                priority_hint=self._priority_hint(metadata),
            ))

        selected = self._select_resources(resources)
        self._resource_cache = selected
        return selected

    def describe_capabilities(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "inspect_resource",
                "purpose": "Read one DICOM file header and emit generic metadata observations.",
                "cost_class": "low",
                "risk_class": "low",
                "notes": "v0 is metadata-only; pixel/vision analysis is intentionally not used.",
            }
        ]

    def inspect_resource(self, *, resource: ResourceRef, instruction: str) -> ObservationPacket:
        path = Path(resource.uri)

        try:
            metadata = self._read_metadata(path)
            summary = self._summary_from_metadata(metadata, path)
            warnings: list[str] = []

            if metadata.get("read_error"):
                warnings.append(f"DICOM metadata read error: {metadata['read_error']}")
            if metadata.get("z_coordinate") is None:
                warnings.append("No usable z-coordinate found in SliceLocation or ImagePositionPatient.")
            if not metadata.get("series_description"):
                warnings.append("SeriesDescription missing or empty.")

            return ObservationPacket(
                resource_id=resource.resource_id,
                action="inspect_resource",
                summary=summary,
                raw_text=safe_str(metadata, 2000),
                structured={
                    "metadata": metadata,
                    "instruction": instruction,
                },
                artifact_refs=[str(path)],
                warnings=warnings,
                error=metadata.get("read_error"),
                token_estimate=estimate_tokens(summary + safe_str(metadata, 2000)),
            )

        except Exception as exc:
            summary = f"Failed to inspect DICOM resource {resource.resource_id}: {exc}"
            return ObservationPacket(
                resource_id=resource.resource_id,
                action="inspect_resource",
                summary=summary,
                raw_text="",
                structured={"path": str(path)},
                artifact_refs=[str(path)],
                warnings=[summary],
                error=str(exc),
                token_estimate=estimate_tokens(summary),
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
        metadata = observation.structured.get("metadata", {}) or {}
        candidates: list[MemoryCandidate] = []

        # Exact source path should remain audit evidence, not necessarily active truth.
        candidates.append(MemoryCandidate(
            candidate_id=new_id("cand"),
            candidate_type="artifacts",
            content=f"DICOM evidence artifact inspected: {resource.uri}",
            source_type="deterministic_extractor",
            evidence_ref=archive_id,
            task_id=task_id,
            metadata={
                "artifact_type": "dicom",
                "path": resource.uri,
                "resource_id": resource.resource_id,
            },
            evidence_strength=3,
            audit_value=3,
            token_cost=estimate_tokens(resource.uri),
        ))

        # Compact metadata observation. Survival decision belongs to Sieve/Governor.
        candidates.append(MemoryCandidate(
            candidate_id=new_id("cand"),
            candidate_type="observations",
            content=self._candidate_text(metadata, resource),
            source_type="deterministic_extractor",
            evidence_ref=archive_id,
            task_id=task_id,
            metadata={
                "field": "dicom_metadata",
                "value": {
                    "patient_id": metadata.get("patient_id"),
                    "study_date": metadata.get("study_date"),
                    "series_description": metadata.get("series_description"),
                    "modality": metadata.get("modality"),
                    "z_coordinate": metadata.get("z_coordinate"),
                    "resource_id": resource.resource_id,
                    "path": resource.uri,
                },
                "resource_id": resource.resource_id,
            },
            evidence_strength=3,
            future_utility=2,
            audit_value=3,
            token_cost=estimate_tokens(observation.summary),
        ))

        if observation.warnings or observation.error:
            candidates.append(MemoryCandidate(
                candidate_id=new_id("cand"),
                candidate_type="caveats",
                content="; ".join(observation.warnings) or str(observation.error),
                source_type="deterministic_extractor",
                evidence_ref=archive_id,
                task_id=task_id,
                metadata={
                    "warnings": observation.warnings,
                    "resource_id": resource.resource_id,
                    "path": resource.uri,
                },
                evidence_strength=2,
                future_utility=2,
                audit_value=3,
                risk_if_wrong=2,
                token_cost=estimate_tokens("; ".join(observation.warnings)),
            ))

        return candidates

    def _select_resources(self, resources: list[ResourceRef]) -> list[ResourceRef]:
        if self.limit is None or self.limit <= 0 or len(resources) <= self.limit:
            return resources

        if self.selection_strategy != "balanced":
            return resources[: self.limit]

        groups: dict[tuple[str, str], list[ResourceRef]] = defaultdict(list)
        for r in resources:
            key = (
                str(r.metadata.get("study_date") or "unknown_date"),
                str(r.metadata.get("series_description") or "unknown_series"),
            )
            groups[key].append(r)

        # Sort within group anatomically when possible.
        for key in groups:
            groups[key].sort(key=lambda r: self._z_sort_key(r))

        ordered_group_keys = sorted(groups.keys())
        selected: list[ResourceRef] = []
        seen: set[str] = set()

        # Pick middle, quartiles, then edges from each series/date group.
        round_fractions = [0.50, 0.25, 0.75, 0.00, 1.00]
        for frac in round_fractions:
            for key in ordered_group_keys:
                group = groups[key]
                if not group:
                    continue
                idx = int(round(frac * (len(group) - 1)))
                ref = group[idx]
                if ref.resource_id in seen:
                    continue
                selected.append(ref)
                seen.add(ref.resource_id)
                if len(selected) >= self.limit:
                    return selected

        # Fill remaining deterministically if limit not reached.
        for r in resources:
            if r.resource_id not in seen:
                selected.append(r)
                seen.add(r.resource_id)
                if len(selected) >= self.limit:
                    break

        return selected

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any]:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)

            ipp = DicomResourceAdapter._as_list(getattr(ds, "ImagePositionPatient", None))
            z_from_ipp = None
            if ipp and len(ipp) >= 3:
                z_from_ipp = DicomResourceAdapter._safe_float(ipp[2])

            slice_location = DicomResourceAdapter._safe_float(getattr(ds, "SliceLocation", None))
            z_coordinate = slice_location if slice_location is not None else z_from_ipp

            return {
                "path": str(path),
                "patient_id": DicomResourceAdapter._safe_str(getattr(ds, "PatientID", None)),
                "study_date": DicomResourceAdapter._safe_str(getattr(ds, "StudyDate", None)),
                "study_time": DicomResourceAdapter._safe_str(getattr(ds, "StudyTime", None)),
                "modality": DicomResourceAdapter._safe_str(getattr(ds, "Modality", None)),
                "series_description": DicomResourceAdapter._safe_str(getattr(ds, "SeriesDescription", None)),
                "sequence_name": DicomResourceAdapter._safe_str(getattr(ds, "SequenceName", None)),
                "scanning_sequence": DicomResourceAdapter._safe_str(getattr(ds, "ScanningSequence", None)),
                "series_instance_uid": DicomResourceAdapter._safe_str(getattr(ds, "SeriesInstanceUID", None)),
                "study_instance_uid": DicomResourceAdapter._safe_str(getattr(ds, "StudyInstanceUID", None)),
                "sop_instance_uid": DicomResourceAdapter._safe_str(getattr(ds, "SOPInstanceUID", None)),
                "instance_number": DicomResourceAdapter._safe_int(getattr(ds, "InstanceNumber", None)),
                "slice_location": slice_location,
                "image_position_patient": ipp,
                "z_coordinate": z_coordinate,
                "pixel_spacing": DicomResourceAdapter._as_list(getattr(ds, "PixelSpacing", None)),
                "slice_thickness": DicomResourceAdapter._safe_float(getattr(ds, "SliceThickness", None)),
                "rows": DicomResourceAdapter._safe_int(getattr(ds, "Rows", None)),
                "columns": DicomResourceAdapter._safe_int(getattr(ds, "Columns", None)),
                "read_error": None,
            }
        except Exception as exc:
            return {
                "path": str(path),
                "read_error": str(exc),
            }

    @staticmethod
    def _title_from_metadata(metadata: dict[str, Any], *, fallback: str) -> str:
        date = metadata.get("study_date") or "unknown_date"
        series = metadata.get("series_description") or "unknown_series"
        z = metadata.get("z_coordinate")
        if z is None:
            return f"{date} | {series} | {fallback}"
        return f"{date} | {series} | z={z}"

    @staticmethod
    def _summary_from_metadata(metadata: dict[str, Any], path: Path) -> str:
        if metadata.get("read_error"):
            return f"DICOM metadata read failed for {path}: {metadata['read_error']}"

        return (
            "DICOM metadata inspected: "
            f"patient={metadata.get('patient_id')}, "
            f"date={metadata.get('study_date')}, "
            f"modality={metadata.get('modality')}, "
            f"series={metadata.get('series_description')}, "
            f"z={metadata.get('z_coordinate')}, "
            f"rows={metadata.get('rows')}, cols={metadata.get('columns')}, "
            f"path={path}"
        )

    @staticmethod
    def _candidate_text(metadata: dict[str, Any], resource: ResourceRef) -> str:
        return (
            "DICOM metadata observation: "
            f"resource_id={resource.resource_id}; "
            f"patient_id={metadata.get('patient_id')}; "
            f"study_date={metadata.get('study_date')}; "
            f"modality={metadata.get('modality')}; "
            f"series_description={metadata.get('series_description')}; "
            f"z_coordinate={metadata.get('z_coordinate')}; "
            f"path={resource.uri}"
        )

    @staticmethod
    def _priority_hint(metadata: dict[str, Any]) -> int:
        # Keep this generic-ish: prioritize complete metadata and mid-level utility,
        # not medical survival logic.
        score = 0
        for key in ["study_date", "series_description", "z_coordinate", "patient_id"]:
            if metadata.get(key) is not None:
                score += 1
        return score

    @staticmethod
    def _z_sort_key(resource: ResourceRef) -> tuple[int, float, str]:
        z = resource.metadata.get("z_coordinate")
        if z is None:
            return (1, 0.0, resource.uri)
        return (0, float(z), resource.uri)

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _as_list(value: Any) -> list[Any] | None:
        if value is None:
            return None
        try:
            return [DicomResourceAdapter._safe_float(v) for v in list(value)]
        except Exception:
            return [str(value)]
