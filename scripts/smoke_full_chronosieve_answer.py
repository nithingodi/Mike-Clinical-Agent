from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.state_loop import llm
from src.chronosieve.core.memory_access import ChronoSieveMemoryAccessController
from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.utils import estimate_tokens, read_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--session-id", required=True)
    p.add_argument(
        "--question",
        default="When did Caroline go to the LGBTQ support group?",
    )
    return p.parse_args()


def main():
    args = parse_args()

    storage = ChronoSieveStorage(session_id=args.session_id)

    events = read_jsonl(storage.events_path)
    archives = read_jsonl(storage.archive_path)
    ledger = read_jsonl(storage.ledger_path)
    carry = storage.read_carry_packet_text()

    print("\n=== FULL CHRONOSIEVE SMOKE TEST ===")
    print(f"session_id: {args.session_id}")
    print(f"question: {args.question}")
    print(f"memory_events: {len(events)}")
    print(f"archives: {len(archives)}")
    print(f"ledger_rows: {len(ledger)}")
    print(f"carry_tokens: {estimate_tokens(carry)}")

    controller = ChronoSieveMemoryAccessController(
        storage=storage,
        llm=llm,
        max_event_inventory=80,
        max_archive_inventory=80,
        max_selected_archives=5,
        max_selected_events=8,
        max_temporary_evidence_chars=9000,
        runtime_segment_chars=12000,
    )

    result = controller.answer(args.question)
    decision = result.memory_access_decision

    print("\n=== MEMORY ACCESS DECISION ===")
    print(f"selector_used: {decision.selector_used}")
    print(f"needs_temporary_evidence: {decision.needs_temporary_evidence}")
    print(f"confidence: {decision.confidence}")
    print(f"selected_archive_refs: {decision.selected_archive_refs}")
    print(f"selected_event_ids: {decision.selected_event_ids}")
    print(f"selected_ledger_task_ids: {decision.selected_ledger_task_ids}")
    print(f"reason: {decision.reason}")

    print("\n=== RUNTIME BUILD ===")
    print(f"prompt_tokens: {result.prompt_tokens}")
    print(f"runtime_context_tokens: {result.runtime_context_tokens}")
    print(f"temporary_evidence_tokens: {result.temporary_evidence_tokens}")
    print("kv_action_summary:")
    print(json.dumps(result.kv_action_summary, indent=2))
    print("runtime_segments:")
    print(json.dumps(result.runtime_segments, indent=2))

    print("\n=== ANSWER ===")
    print(result.answer)

    print("\n=== PASS CRITERIA ===")
    print("- No Python/import/runtime errors")
    print("- MemoryAccessController selected reasonable handles or decided memory was enough")
    print("- Runtime context was built with KV actions + segments")
    print("- AnswerSynthesizer returned an answer or UNKNOWN from governed context")


if __name__ == "__main__":
    main()