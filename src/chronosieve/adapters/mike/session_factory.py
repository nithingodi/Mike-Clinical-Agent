from __future__ import annotations

from src.agent.state_loop import invoke_mike_with_trace, llm
from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.sieve_worker import GemmaSieveWorker
from src.chronosieve.core.session import ChronoSieveSession
from src.chronosieve.adapters.mike.trace_parser import MikeTraceParser


def create_mike_chronosieve_session(
    session_id: str = "demo_mike_session",
    storage_root: str = "storage/chronosieve_sessions",
) -> ChronoSieveSession:
    """
    Factory for ChronoSieve + existing Mike.

    Brain 1:
        invoke_mike_with_trace()

    Brain 2:
        same Gemma 4 31B llm object, used as a restricted Sieve Worker.

    Storage:
        file-backed ChronoSieve session directory.
    """
    storage = ChronoSieveStorage(
        session_id=session_id,
        root_dir=storage_root,
        adapter_name="mike_clinical_trace_adapter_v1",
        backend_model=getattr(llm, "model_name", "gemma-4-31b-llama-cpp"),
    )

    sieve_worker = GemmaSieveWorker(llm=llm)

    return ChronoSieveSession(
        session_id=session_id,
        task_agent_callable=invoke_mike_with_trace,
        sieve_worker=sieve_worker,
        storage=storage,
        trace_parser=MikeTraceParser(),
    )
