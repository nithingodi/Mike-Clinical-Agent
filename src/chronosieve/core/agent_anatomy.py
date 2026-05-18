from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCapability:
    """
    A generic description of a tool/action the autonomous executor may use.

    This is intentionally domain-neutral. A DICOM adapter, code adapter,
    legal-doc adapter, finance adapter, sports adapter, or support-ticket
    adapter can expose different capabilities through the same shape.
    """

    name: str
    purpose: str
    input_schema_hint: str = ""
    output_schema_hint: str = ""
    cost_class: str = "low"  # low | medium | high
    risk_class: str = "low"  # low | medium | high

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentAnatomy:
    """
    What the model should know about its own operating body.

    This does not make the system domain-specific. It tells the task brain
    what context size, carry limits, archive/rehydration abilities, runtime
    backend, and operating rules exist so it can plan bounded long-horizon
    work instead of trying to stuff all evidence into a single prompt.
    """

    model_name: str = "gemma-4-31B-it-Q4_K_M.gguf"
    ctx_size: int = 32768
    safe_context_budget: int = 26000
    max_output_tokens: int = 4096
    carry_soft_limit: int = 4000
    carry_hard_limit: int = 6000
    archive_available: bool = True
    rehydration_available: bool = True
    runtime_backend: str = "prompt_cache_runtime_v0"
    kv_direct_available: bool = False

    memory_statuses: list[str] = field(default_factory=lambda: [
        "fossil",
        "active",
        "archived",
        "decay",
        "invalidated",
        "non_authoritative",
    ])

    operating_rules: list[str] = field(default_factory=lambda: [
        "Do not append full task history indefinitely.",
        "Process long evidence streams through bounded steps.",
        "Archive raw evidence externally.",
        "Carry only governed memory into future loops.",
        "Use rehydration for old evidence instead of trusting stale recall.",
        "Separate deterministic facts from model interpretations.",
        "Do not promote proxy/substitute results as authoritative truth.",
    ])

    tools: list[ToolCapability] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tools"] = [tool.to_dict() for tool in self.tools]
        return data

    def compact_briefing(self) -> dict[str, Any]:
        """
        Short version safe to send to the planner each loop.
        """
        return {
            "model_name": self.model_name,
            "ctx_size": self.ctx_size,
            "safe_context_budget": self.safe_context_budget,
            "max_output_tokens": self.max_output_tokens,
            "carry_soft_limit": self.carry_soft_limit,
            "carry_hard_limit": self.carry_hard_limit,
            "archive_available": self.archive_available,
            "rehydration_available": self.rehydration_available,
            "runtime_backend": self.runtime_backend,
            "kv_direct_available": self.kv_direct_available,
            "memory_statuses": self.memory_statuses,
            "operating_rules": self.operating_rules,
            "tools": [tool.to_dict() for tool in self.tools],
        }
