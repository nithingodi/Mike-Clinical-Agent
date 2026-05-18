from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chronosieve.adapters.mike.session_factory import create_mike_chronosieve_session


if __name__ == "__main__":
    session = create_mike_chronosieve_session(session_id="demo_mike_session")

    prompt = (
        "Calculate the exact functional shift in millimeters of the motor cortex between the two scan sessions for UPENN-GBM-00045."
    )

    result = session.handle_turn(prompt)

    print("\n=== CHRONOSIEVE-WRAPPED ANSWER ===\n")
    print(result["answer"])

    print("\n=== CHRONOSIEVE METADATA ===")
    print(f"task_id: {result['task_id']}")
    print(f"archive_id: {result['archive_id']}")
    print(f"candidate_count: {result['candidate_count']}")
    print(f"memory_event_count: {result['memory_event_count']}")
    print(f"carry_packet_token_estimate: {result['carry_packet_token_estimate']}")
    print(f"storage_dir: {result['storage_dir']}")
    print(f"rehydration_triggered: {result.get('rehydration_triggered')}")
    print(f"rehydrated_archive_refs: {result.get('rehydrated_archive_refs')}")

    if result.get("cautions"):
        print("\n=== CAUTIONS ===")
        for caution in result["cautions"]:
            print(f"- {caution}")