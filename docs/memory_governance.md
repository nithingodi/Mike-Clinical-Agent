# Memory Governance Contract

ChronoSieve separates memory into several layers:

- Evidence archive: full raw task evidence and traces.
- Task ledger: compact task-level summaries.
- Memory events: committed memory after model recommendation and Python validation.
- Carry packet: bounded working memory passed to the task agent.
- Correction registry: reference-frame changes, factual corrections, and invalidations.
- Non-authoritative items: claims that must not be treated as verified truth.

The design goal is to keep the active context small while preserving auditability. The model should not carry every token forward. It should carry what matters, with the correct authority level.
