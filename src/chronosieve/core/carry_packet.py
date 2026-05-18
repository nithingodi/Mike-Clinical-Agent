from __future__ import annotations

from .schemas import CarryPacket, MemoryEvent
from .utils import estimate_tokens
import re

PATH_LIKE_RE = re.compile(r"(?:[\w.\-]+[/\\])+[\w.\-]+\.[A-Za-z0-9]{2,8}")

class CarryPacketBuilder:
    """
    Builds the clean briefing note that Brain 1 receives next turn.

    This should replace full raw chat history as the main continuity mechanism.
    """

    def __init__(self, soft_limit_tokens: int = 4000, hard_limit_tokens: int = 6000):
        self.soft_limit_tokens = soft_limit_tokens
        self.hard_limit_tokens = hard_limit_tokens

    def build(
        self,
        *,
        session_id: str,
        recent_events: list[dict],
        new_events: list[MemoryEvent],
        alias_memory: dict,
        correction_registry: dict,
    ) -> CarryPacket:
        sections: list[str] = []
        included_event_ids: list[str] = []
        included_archive_refs: list[str] = []

        sections.append("# ChronoSieve Carry Packet")
        sections.append("")
        sections.append("This is a clean briefing note selected by ChronoSieve. It is not a transcript.")
        sections.append("")

        fossils = []
        active = []
        cautions = []
        non_authoritative = []
        invalidated = []
        audit_refs = []

        all_events = list(recent_events) + [event.to_dict() for event in new_events]

        for row in all_events[-80:]:
            decision = row.get("decision", {})
            candidate = row.get("candidate", {})
            status = decision.get("status")
            ctype = candidate.get("candidate_type")
            carry_text = decision.get("carry_text") or candidate.get("content")
            event_id = row.get("event_id")
            archive_ref = decision.get("archive_ref") or candidate.get("evidence_ref")

            if not carry_text:
                continue

            carry_text = self._compact_text(str(carry_text))

            if status == "fossil":
                fossils.append(carry_text)
                if event_id:
                    included_event_ids.append(event_id)
            elif status == "active":
                active.append(carry_text)
                if event_id:
                    included_event_ids.append(event_id)
            elif status == "non_authoritative":
                non_authoritative.append(carry_text)
                if event_id:
                    included_event_ids.append(event_id)
            elif status == "invalidated":
                invalidated.append(carry_text)
                if event_id:
                    included_event_ids.append(event_id)

            if ctype == "caveats" and status in {"fossil", "active"}:
                cautions.append(carry_text)

            # Keep audit refs only for memory that actually survived or must constrain future truth.
            # Do not list every decayed raw trace in the carry packet.
            if archive_ref and status in {"fossil", "active", "non_authoritative", "invalidated"}:
                audit_refs.append(str(archive_ref))
                included_archive_refs.append(str(archive_ref))

        aliases = alias_memory.get("aliases", []) if isinstance(alias_memory, dict) else []
        corrections = correction_registry.get("corrections", []) if isinstance(correction_registry, dict) else []

        if aliases:
            sections.append("## Validated Procedural / Alias Memory")
            for alias in aliases[-20:]:
                sections.append(f"- {alias.get('alias')} → {alias.get('canonical')} ({alias.get('reason')})")
            sections.append("")

        if fossils:
            sections.append("## Fossils / Protected Memory")
            for item in self._dedupe(fossils)[-30:]:
                sections.append(f"- {item}")
            sections.append("")

        if active:
            sections.append("## Active Working Memory")
            for item in self._dedupe(active)[-30:]:
                sections.append(f"- {item}")
            sections.append("")

        if cautions:
            sections.append("## Required Caveats")
            for item in self._dedupe(cautions)[-20:]:
                sections.append(f"- {item}")
            sections.append("")

        if non_authoritative:
            sections.append("## Non-Authoritative / Proxy-Only Items")
            sections.append("Do not treat these as authoritative truth for the original requested task.")
            for item in self._dedupe(non_authoritative)[-20:]:
                sections.append(f"- {item}")
            sections.append("")

        if invalidated:
            sections.append("## Invalidated / Superseded Items")
            sections.append("These must not be reused as truth unless explicitly revalidated.")
            for item in self._dedupe(invalidated)[-20:]:
                sections.append(f"- {item}")
            sections.append("")

        if corrections:
            sections.append("## Correction Registry Summary")
            for corr in corrections[-20:]:
                sections.append(
                    f"- {corr.get('correction_type')}: {corr.get('old_state')} → {corr.get('new_state')} | {corr.get('reason')}"
                )
            sections.append("")

        if audit_refs:
            sections.append("## Audit / Rehydration References")
            for ref in self._dedupe(audit_refs)[-30:]:
                sections.append(f"- {ref}")
            sections.append("")

        content = "\n".join(sections).strip() + "\n"
        content = self._truncate_to_hard_limit(content)

        return CarryPacket(
            session_id=session_id,
            content_md=content,
            token_estimate=estimate_tokens(content),
            included_event_ids=self._dedupe(included_event_ids),
            included_archive_refs=self._dedupe(included_archive_refs),
        )

    def _truncate_to_hard_limit(self, content: str) -> str:
        tokens = estimate_tokens(content)
        if tokens <= self.hard_limit_tokens:
            return content

        # v1 rough truncation. Keep beginning, which contains fossils/cautions.
        max_chars = self.hard_limit_tokens * 4
        return content[:max_chars] + "\n\n...[ChronoSieve carry packet truncated to hard token limit]\n"

    
    @staticmethod
    def _compact_text(text: str, max_chars: int = 700) -> str:
        clean = " ".join(text.split())

        path_count = len(set(PATH_LIKE_RE.findall(clean)))
        if path_count and len(clean) > 260:
            return (
                f"Archived artifact/path payload omitted from carry packet "
                f"({path_count} path-like value(s)); use audit/rehydration references for exact values."
            )

        if len(clean) > max_chars:
            return clean[:max_chars] + "...[truncated]"
        return clean

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
