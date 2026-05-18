from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.kv_sync import KVSyncPlanner
from src.chronosieve.core.runtime_context import RuntimeContextBuilder
from src.chronosieve.backends.kv_direct_contract import KVDirectContractPlanner


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

    print(direct_plan.as_markdown())
