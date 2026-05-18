from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .utils import estimate_tokens


PATH_LIKE_RE = re.compile(r"(?:[\w.\-]+[/\\])+[\w.\-]+\.[A-Za-z0-9]{2,8}")


@dataclass
class RehydrationBundle:
    content_md: str
    archive_refs: list[str]
    trigger_reason: str
    token_estimate: int


class RehydrationPolicy:
    """
    Generic archive recall policy.

    Detects audit/evidence/path/trace recall requests and injects compact
    archived evidence before Brain 1 runs.

    Domain-neutral:
    - no patient logic
    - no DICOM logic
    - no clinical rules
    """

    TEMPORAL_TERMS = {
        "earlier", "previous", "previously", "prior", "last",
        "before", "go back", "from earlier", "from before",
    }

    AUDIT_TERMS = {
        "exact", "evidence", "source", "sources", "trace", "tool", "tools",
        "observation", "observations", "file", "files", "path", "paths",
        "artifact", "artifacts", "retrieved", "used", "which", "what",
        "why", "how", "audit", "archive",
    }

    STOPWORDS = {
        "the", "and", "for", "with", "from", "that", "this", "what", "which",
        "when", "where", "were", "was", "are", "you", "did", "have", "their",
        "about", "into", "between", "both", "task", "earlier", "previous",
        "previously", "prior", "last", "go", "back",
    }

    def __init__(self, max_archives: int = 2, max_bundle_chars: int = 5000):
        self.max_archives = max_archives
        self.max_bundle_chars = max_bundle_chars

    def build_bundle(
        self,
        *,
        user_request: str,
        recent_archives: list[dict[str, Any]],
        recent_ledger: list[dict[str, Any]] | None = None,
        artifacts_index: dict[str, Any] | None = None,
    ) -> RehydrationBundle | None:
        trigger_reason = self._detect_trigger(user_request)
        if not trigger_reason:
            return None

        scored: list[tuple[int, dict[str, Any], list[str]]] = []

        for archive in recent_archives:
            evidence_items = self._extract_evidence_items(archive)
            score = self._score_archive(user_request, archive, evidence_items)
            if score > 0:
                scored.append((score, archive, evidence_items))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[: self.max_archives]

        sections: list[str] = []
        archive_refs: list[str] = []

        sections.append("# ChronoSieve Rehydrated Evidence Bundle")
        sections.append("")
        sections.append(
            "This is selected archived evidence for the current audit/recall request. "
            "If this evidence directly answers the user, answer from it and do not rerun tools "
            "merely to rediscover the same evidence. If insufficient, say what is missing."
        )
        sections.append("")
        sections.append(f"Trigger: {trigger_reason}")
        sections.append("")

        for score, archive, evidence_items in selected:
            archive_id = str(archive.get("archive_id", "unknown_archive"))
            archive_refs.append(archive_id)

            sections.append(f"## Archive: {archive_id}")
            sections.append(f"Task ID: {archive.get('task_id')}")
            sections.append(f"Relevance score: {score}")
            sections.append("Original request:")
            sections.append("- " + self._compact(str(archive.get("user_request", "")), 700))

            tool_names = [str(step.get("tool", "unknown")) for step in archive.get("trace", []) or []]
            if tool_names:
                sections.append("Tool sequence:")
                sections.append("- " + " -> ".join(tool_names))

            if evidence_items:
                sections.append("Extracted archived evidence:")
                for item in evidence_items[:16]:
                    sections.append("- " + self._compact(item, 900))

            sections.append("")

        content = "\n".join(sections).strip() + "\n"
        if len(content) > self.max_bundle_chars:
            content = content[: self.max_bundle_chars] + "\n...[rehydrated evidence bundle truncated]\n"

        return RehydrationBundle(
            content_md=content,
            archive_refs=self._dedupe(archive_refs),
            trigger_reason=trigger_reason,
            token_estimate=estimate_tokens(content),
        )

    def _detect_trigger(self, user_request: str) -> str | None:
        text = " ".join(str(user_request).lower().split())

        has_temporal = any(term in text for term in self.TEMPORAL_TERMS)
        has_audit = any(term in text for term in self.AUDIT_TERMS)

        if has_temporal and has_audit:
            return "temporal audit/recall intent detected"

        strong_phrases = {
            "which exact", "what exact", "show the exact",
            "what evidence", "which evidence",
            "what sources", "which sources", "audit trail",
        }

        if any(phrase in text for phrase in strong_phrases):
            return "explicit evidence/source/path recall intent detected"

        return None

    def _score_archive(
        self,
        user_request: str,
        archive: dict[str, Any],
        evidence_items: list[str],
    ) -> int:
        query_tokens = self._tokens(user_request)

        archive_text = "\n".join([
            str(archive.get("user_request", "")),
            str(archive.get("final_answer", ""))[:1500],
            "\n".join(evidence_items[:20]),
        ])

        archive_tokens = self._tokens(archive_text)
        score = len(query_tokens & archive_tokens)

        query_l = user_request.lower()

        if any(x in query_l for x in {"path", "paths", "file", "files", "artifact", "artifacts"}):
            if any(PATH_LIKE_RE.search(item) for item in evidence_items):
                score += 10

        if any(x in query_l for x in {"trace", "tool", "tools", "observation", "evidence"}):
            if archive.get("trace"):
                score += 5

        original_request_tokens = self._tokens(str(archive.get("user_request", "")))
        score += 2 * len(query_tokens & original_request_tokens)

        return score

    def _extract_evidence_items(self, archive: dict[str, Any]) -> list[str]:
        items: list[str] = []

        for idx, step in enumerate(archive.get("trace", []) or [], start=1):
            tool = str(step.get("tool", "unknown_tool"))
            observation = str(step.get("observation", ""))
            obs_json = self._try_json(observation)

            if isinstance(obs_json, dict):
                for key in ["canonical_id", "file_path", "png_path", "z_coordinate"]:
                    if key in obs_json:
                        items.append(
                            f"tool step {idx} ({tool}) observation {key}: {obs_json.get(key)}"
                        )

            for path in sorted(set(PATH_LIKE_RE.findall(observation))):
                items.append(f"tool step {idx} ({tool}) observed path: {path}")

        final_answer = str(archive.get("final_answer", ""))
        for path in sorted(set(PATH_LIKE_RE.findall(final_answer))):
            items.append(f"final answer referenced path: {path}")

        return self._dedupe(items)

    @staticmethod
    def _try_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _tokens(self, text: str) -> set[str]:
        raw = re.findall(r"[a-zA-Z0-9_\-]+", str(text).lower())
        return {t for t in raw if len(t) > 2 and t not in self.STOPWORDS}

    @staticmethod
    def _compact(text: str, max_chars: int) -> str:
        clean = " ".join(str(text).split())
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