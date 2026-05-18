from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.kv_sync import KVSyncPlanner
from src.chronosieve.core.runtime_context import RuntimeContextBuilder
from src.chronosieve.backends.kv_direct_contract import KVDirectContractPlanner
from src.chronosieve.backends.mock_kv_direct_runtime import MockKVDirectRuntime


if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "demo_mike_session"

    storage = ChronoSieveStorage(session_id=session_id)

    kv_plan = KVSyncPlanner().plan(
        session_id=session_id,
        events=storage.read_recent_events(limit=500),
        carry_packet_text=storage.read_carry_packet_text(),
        correction_registry=storage.read_correction_registry(),
    )

    runtime_package = RuntimeContextBuilder().build(kv_plan)
    direct_plan = KVDirectContractPlanner().plan(runtime_package)

    runtime = MockKVDirectRuntime()
    result = runtime.apply_plan(direct_plan)
    state = result["runtime_state"]

    print("# Mock KV-Direct Runtime Test")
    print()
    print(f"session_id: {session_id}")
    print(f"operation_count: {result['operation_count']}")
    print()

    print("## Segment counts")
    for key in [
        "pinned_segments",
        "active_segments",
        "warning_segments",
        "blocked_truth",
        "external_refs",
        "temporary_segments",
        "evicted_segments",
    ]:
        print(f"- {key}: {len(state.get(key, {}))}")

    print()
    print("## Applied operations")
    for row in result["applied"]:
        print(f"- {row['operation_type']} · {row['segment_type']} · {row['status']}")


    print()
    print("## Blocked truth")
    for seg in state.get("blocked_truth", {}).values():
        print("-", (seg.get("content") or "")[:800])

    print()
    print("## Warning segments")
    for seg in state.get("warning_segments", {}).values():
        print("-", (seg.get("content") or "")[:1000])

    print()
    print("## Full runtime state JSON preview")
    print(json.dumps(state, indent=2, ensure_ascii=False)[:5000])
