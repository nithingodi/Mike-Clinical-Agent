from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Protocol

from src.chronosieve.core.schemas import MemoryCandidate


@dataclass
class ResourceRef:
    """
    A generic pointer to one evidence/resource item.

    Examples:
    - DICOM file
    - PDF page
    - code file
    - support ticket
    - game log
    - financial filing section

    ChronoSieve core should not need to know which domain this came from.
    """

    resource_id: str
    uri: str
    resource_type: str = "generic"
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    cost_hint: int = 1
    priority_hint: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationPacket:
    """
    Output of inspecting exactly one resource/action.

    This is not automatically authoritative memory. It is raw/structured
    evidence that must pass through candidate extraction + Sieve + Governor.
    """

    resource_id: str
    action: str
    summary: str
    raw_text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceAdapter(Protocol):
    """
    Generic adapter interface for autonomous long-horizon work.

    ChronoSieve core should depend only on this interface, not on DICOM,
    legal, code, finance, sports, or support-ticket domain specifics.
    """

    adapter_name: str

    def list_resources(self, *, goal: str) -> list[ResourceRef]:
        """
        Return the evidence/resource universe available for a broad goal.

        The returned refs should be lightweight pointers/previews, not full
        resource contents.
        """
        ...

    def describe_capabilities(self) -> list[dict[str, Any]]:
        """
        Return the actions this adapter can perform.

        Example actions:
        - inspect_resource
        - summarize_resource
        - extract_metadata
        - render_preview
        """
        ...

    def inspect_resource(
        self,
        *,
        resource: ResourceRef,
        instruction: str,
    ) -> ObservationPacket:
        """
        Inspect exactly one resource using one bounded instruction.

        The implementation can call tools, read files, render images, query a DB,
        or perform any domain-specific work. The output must be converted into a
        generic ObservationPacket.
        """
        ...

    def candidates_from_observation(
        self,
        *,
        goal: str,
        task_id: str,
        archive_id: str,
        observation: ObservationPacket,
        resource: ResourceRef,
    ) -> list[MemoryCandidate]:
        """
        Convert the adapter observation into generic ChronoSieve candidates.

        Domain-specific extraction is allowed here, but long-term memory survival
        is not decided here. Survival is decided later by Sieve + Governor.
        """
        ...
