# ChronoSieve Architecture

ChronoSieve is a memory-governance layer for long-running AI workflows.

It is not a vector database and not simple chat history. It decides what should survive a turn, what should be archived, what should remain non-authoritative, and what should be blocked from becoming trusted state.

## Core loop

1. Task agent performs the user task.
2. Raw evidence is archived.
3. Adapter extracts memory candidates from tool traces and answers.
4. Sieve worker proposes what should survive.
5. Policy governor enforces hard rules.
6. Carry packet is rebuilt as bounded working memory for the next turn.

## In this repo

Mike is the clinical task agent. ChronoSieve wraps Mike and preserves exact clinical retrieval state such as scan dates, DICOM paths, Z-coordinates, caveats, corrections, and non-authoritative interpretation boundaries.

## Safety boundary

ChronoSieve does not make clinical findings authoritative. It preserves evidence, uncertainty, and caveats so clinicians can audit what the AI used.
