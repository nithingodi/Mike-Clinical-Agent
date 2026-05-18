from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import (
    EvidenceArchiveEntry,
    MemoryEvent,
    TaskLedgerEntry,
    CorrectionRecord,
    CarryPacket,
    utc_now,
)
from .utils import append_jsonl, read_jsonl, read_json, write_json


class ChronoSieveStorage:
    """
    File-backed v1 storage.

    This is intentionally simple:
    - no database
    - no vector store
    - no LangGraph
    - no KV internals

    Storage is the external temporal substrate.
    """

    def __init__(
        self,
        session_id: str,
        root_dir: str | Path = "storage/chronosieve_sessions",
        adapter_name: str = "generic",
        backend_model: str = "unknown",
    ):
        self.session_id = session_id
        self.root_dir = Path(root_dir)
        self.session_dir = self.root_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.session_dir / "session_manifest.json"
        self.ledger_path = self.session_dir / "ledger.jsonl"
        self.archive_path = self.session_dir / "evidence_archive.jsonl"
        self.events_path = self.session_dir / "memory_events.jsonl"
        self.sieve_worker_logs_path = self.session_dir / "sieve_worker_logs.jsonl"
        self.alias_memory_path = self.session_dir / "alias_memory.json"
        self.correction_registry_path = self.session_dir / "correction_registry.json"
        self.artifacts_index_path = self.session_dir / "artifacts_index.json"
        self.carry_packet_path = self.session_dir / "carry_packet.md"

        self._init_manifest(adapter_name=adapter_name, backend_model=backend_model)
        self._init_json_files()

    def _init_manifest(self, adapter_name: str, backend_model: str) -> None:
        if self.manifest_path.exists():
            return
        write_json(self.manifest_path, {
            "session_id": self.session_id,
            "created_at": utc_now(),
            "adapter_name": adapter_name,
            "backend_model": backend_model,
            "phase": "phase_1_python_controller",
            "ctx_size": 32768,
            "safe_budget": 26000,
            "carry_packet_target": "4000-6000 tokens",
        })

    def _init_json_files(self) -> None:
        if not self.alias_memory_path.exists():
            write_json(self.alias_memory_path, {"aliases": []})
        if not self.correction_registry_path.exists():
            write_json(self.correction_registry_path, {"corrections": []})
        if not self.artifacts_index_path.exists():
            write_json(self.artifacts_index_path, {"artifacts": []})
        if not self.carry_packet_path.exists():
            self.carry_packet_path.write_text(
                "# ChronoSieve Carry Packet\n\nNo prior memory yet.\n",
                encoding="utf-8",
            )

    def append_archive(self, entry: EvidenceArchiveEntry) -> None:
        append_jsonl(self.archive_path, entry.to_dict())

    def append_ledger(self, entry: TaskLedgerEntry) -> None:
        append_jsonl(self.ledger_path, entry.to_dict())

    def append_memory_event(self, event: MemoryEvent) -> None:
        append_jsonl(self.events_path, event.to_dict())

    def append_sieve_worker_log(self, record: dict[str, Any]) -> None:
        """
        Audit log for Brain 2 itself.
        This lets us inspect whether the Sieve Brain returned malformed JSON,
        wrong keys, no decisions, or useful decisions.
        """
        record = dict(record)
        record.setdefault("created_at", utc_now())
        append_jsonl(self.sieve_worker_logs_path, record)

    def add_correction(self, record: CorrectionRecord) -> None:
        data = read_json(self.correction_registry_path, {"corrections": []})
        data.setdefault("corrections", []).append(record.to_dict())
        write_json(self.correction_registry_path, data)

    def add_artifacts(self, task_id: str, image_paths: list[str], archive_id: str) -> None:
        if not image_paths:
            return
        data = read_json(self.artifacts_index_path, {"artifacts": []})
        for path in image_paths:
            data.setdefault("artifacts", []).append({
                "task_id": task_id,
                "archive_id": archive_id,
                "path": path,
                "created_at": utc_now(),
            })
        write_json(self.artifacts_index_path, data)

    def add_alias(self, alias: str, canonical: str, reason: str, evidence_ref: str | None = None) -> None:
        data = read_json(self.alias_memory_path, {"aliases": []})
        existing = data.setdefault("aliases", [])

        for row in existing:
            if row.get("alias") == alias and row.get("canonical") == canonical:
                return

        existing.append({
            "alias": alias,
            "canonical": canonical,
            "reason": reason,
            "evidence_ref": evidence_ref,
            "created_at": utc_now(),
        })
        write_json(self.alias_memory_path, data)

    def write_carry_packet(self, packet: CarryPacket) -> None:
        self.carry_packet_path.write_text(packet.content_md, encoding="utf-8")

    def read_carry_packet_text(self) -> str:
        if not self.carry_packet_path.exists():
            return ""
        return self.carry_packet_path.read_text(encoding="utf-8")

    def read_recent_events(self, limit: int = 30) -> list[dict[str, Any]]:
        rows = read_jsonl(self.events_path)
        return rows[-limit:]

    def read_recent_ledger(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = read_jsonl(self.ledger_path)
        return rows[-limit:]
    
    def read_recent_archives(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = read_jsonl(self.archive_path)
        return rows[-limit:]

    def read_artifacts_index(self) -> dict[str, Any]:
        return read_json(self.artifacts_index_path, {"artifacts": []})

    def read_alias_memory(self) -> dict[str, Any]:
        return read_json(self.alias_memory_path, {"aliases": []})

    def read_correction_registry(self) -> dict[str, Any]:
        return read_json(self.correction_registry_path, {"corrections": []})
