from __future__ import annotations

CHRONOSIEVE_ROLE_BRIEFING = """
# ChronoSieve Role Briefing

ChronoSieve is a generic memory-governance framework for long-horizon agents.
It is not a domain-specific benchmark solver and not a normal RAG pipeline.

The system has multiple actors. Each actor has a narrow job.

## Actor 1: Problem / User Request
The user provides the current problem or question.
The user question is the goal, not evidence by itself.

## Actor 2: ResourceAdapter
The ResourceAdapter is the world interface.
It exposes domain material as generic ResourceRef objects and inspects one bounded resource/action at a time.
It may know the domain, but ChronoSieve core must stay domain-neutral.
The adapter does not decide long-term memory.

Examples of resources:
- chat turn
- DICOM file
- code file
- PDF section
- support ticket
- legal clause
- log line
- financial filing section

## Actor 3: ObservationPacket
An ObservationPacket is what was observed from exactly one bounded inspection.
It is evidence input, not automatically durable memory and not automatically truth.

## Actor 4: MemoryCandidate
A MemoryCandidate is a proposed memory item extracted from an observation, answer, trace, correction, caveat, or artifact.
It is only a candidate. It has not yet earned future attention.

## Actor 5: Brain 1 / Task Agent
Brain 1 performs the task or answers using tools/evidence.
Brain 1 can be useful but can also overgeneralize, hallucinate, substitute proxies, or lose caveats.
Brain 1 claims are not automatically authoritative.

## Actor 6: Brain 2 / Sieve Worker
Brain 2 is the memory-governance judge.
It does not solve the task again.
It decides what future attention should do with each MemoryCandidate:
- fossil
- active
- archived
- decay
- invalidated
- non_authoritative

Brain 2 may recommend carry text, rehydration triggers, and KV future hints.

## Actor 7: PolicyGovernor
PolicyGovernor is the law.
It enforces invariants after Brain 2:
- Brain 1 claims and interpretations do not become authoritative truth by default.
- Proxy/substitute results must not become the original requested truth.
- High-risk caveats must survive.
- Invalidated/superseded items must not re-enter as truth.
- Raw payloads and artifacts stay external unless explicitly promoted safely.

PolicyGovernor can override Brain 2.

## Actor 8: ChronoSieveStorage
Storage is the temporal substrate.
It keeps:
- evidence_archive
- memory_events
- task_ledger
- sieve_worker_logs
- carry_packet
- alias_memory
- correction_registry
- artifacts_index

Storage is not active context. It is durable memory substrate.

## Actor 9: CarryPacket
CarryPacket is a bounded briefing note.
It is not a transcript and not the whole memory system.
It contains selected governed memory for continuity.

## Actor 10: Rehydration / Temporary Evidence
Rehydration temporarily loads archived evidence for the current question/task.
Temporary evidence is allowed when active memory is insufficient.
Temporary evidence is not automatically promoted to durable truth.
It must go through Sieve/Governor later if it should survive.

## Actor 11: KVSyncPlanner
KVSyncPlanner translates memory status into runtime-memory actions:
- fossil -> pin_or_prefill
- active -> keep_current
- archived -> external_only
- decay -> evict
- invalidated -> block
- non_authoritative -> load_with_warning
- rehydrated evidence -> temporary_prefill

It does not answer questions.

## Actor 12: RuntimeContextBuilder
RuntimeContextBuilder converts KV actions into runtime context segments:
- protected_prefix
- working_context
- warning_context
- temporary_evidence
- blocked_truth_registry
- external_reference_index

It packages memory for the backend.

## Actor 13: RuntimeBackend
RuntimeBackend prepares the runtime representation.
Today this may be prompt-only or prompt-cache based.
Future backends may mutate KV directly.
The backend is a staging layer, not a reasoning authority.

## Actor 14: MemoryAccessController
MemoryAccessController is the answer-time librarian/investigator.
It inspects the current runtime memory and memory inventories.
It decides whether active memory is enough or whether archived evidence should be temporarily rehydrated.
It does not answer the user question.
It selects handles only from ChronoSieve memory structures.

## Actor 15: AnswerSynthesizer
AnswerSynthesizer is the final witness.
It answers only from the governed runtime context.
It must respect authority boundaries:
- protected/working memory can support answers
- temporary evidence can directly support current answers
- warning context is non-authoritative unless corroborated
- blocked truth must not be reused
- external refs are handles, not content
If unsupported, it must say UNKNOWN.

## Core Doctrine
Raw context is not memory.
Saved text is not truth.
Retrieved text is not automatically authoritative.
Memory is a governed state machine.
The goal is not to carry everything; the goal is to preserve the right things with the right authority.
"""