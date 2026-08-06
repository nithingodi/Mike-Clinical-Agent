# Mike + ChronoSieve: Local Gemma 4 Clinical Evidence Navigator

**Mike** is a local Gemma 4 clinical evidence navigator for longitudinal MRI review.
**ChronoSieve** is the memory-governance layer that helps long-running AI workflows preserve evidence, archive raw traces, track corrections, maintain caveats, and avoid turning weak/proxy claims into authoritative truth.

> Research/demo only. This project is not a diagnostic medical device, treatment recommendation system, or replacement for a radiologist, oncologist, neurosurgeon, clinician, or clinical team.

---

## One-Sentence Summary

Mike helps clinicians navigate large, private, multimodal patient evidence locally with Gemma 4, while ChronoSieve manages what the agent should remember, archive, rehydrate, correct, or block across long workflows.

---

## Why This Matters

Clinical review is not one prompt. A clinician may need to move across:

* scan dates,
* DICOM folders,
* MRI modalities,
* physical slice coordinates,
* rendered images,
* prior findings,
* tool outputs,
* caveats,
* corrections,
* and audit evidence.

For one UPENN-GBM patient case used in this demo:

| Item                                    |  Scale |
| --------------------------------------- | -----: |
| DICOM files                             |  3,002 |
| Scan sessions                           |      2 |
| MRI series                              |     12 |
| Naive all-pixel + metadata token burden | ~1.77M |
| Active turn context target              |   <32K |

This cannot be solved by simply pasting the patient record into a chat window. The system needs selective evidence retrieval, local inference, visible tool traces, and governed memory.

---

## What Is Mike?

Mike is the clinical task agent.

It uses local Gemma 4 to:

* understand clinical-style questions,
* decide when to call tools,
* query patient timelines,
* inspect available MRI modalities,
* retrieve anatomically aligned DICOM slices by physical Z-coordinate,
* convert selected DICOM slices into PNG previews,
* perform research-assistive multimodal review,
* and expose intermediate tool traces for auditability.

Example question:

```text
I need to compare the tumor margins for UPENN-GBM-00045.
Fetch the T1 Post-Contrast slice at exactly Z = -15.0mm for both
the November 2005 and March 2006 sessions.
```

Mike retrieves the relevant scan dates, resolves modality names, selects the closest slices, returns file paths and Z-coordinates, renders images, and provides a trace.

---

## What Is ChronoSieve?

ChronoSieve is a generic memory-governance layer for long-running AI workflows.

It is not just chat history.
It is not a vector database.
It is not a bigger context window.

ChronoSieve asks:

* What should survive this turn?
* What should be archived but not carried?
* What is a model interpretation rather than deterministic evidence?
* What is a caveat?
* What was corrected or invalidated?
* What should be rehydrated later?
* What must not become authoritative truth?

In this clinical demo, ChronoSieve helps preserve:

* scan dates,
* DICOM paths,
* target and actual Z-coordinates,
* modality alias learnings,
* caveats,
* non-authoritative image interpretations,
* correction records,
* and audit references.

---

## High-Level Architecture

```text
User question
  ↓
Mike / Gemma 4 task agent
  ↓
Clinical tools
  ↓
Timeline / DICOM / image evidence
  ↓
Answer + tool trace
  ↓
ChronoSieve
  ↓
Evidence archive + memory candidates
  ↓
Gemma Sieve Worker
  ↓
Python Policy Governor
  ↓
Ledger + carry packet + correction registry
  ↓
Next grounded turn
```

The task agent performs the work. ChronoSieve decides what future attention should do with the result.

---

## Repository Structure

```text
.
├── app.py
├── patient_index.csv
├── requirements.txt
├── README.md
│
├── docs/
│   ├── demo_script.md
│   ├── safety_boundary.md
│   ├── chronosieve_architecture.md
│   └── memory_governance.md
│
├── scripts/
│   ├── run_chronosieve_mike_turn.py
│   ├── smoke_full_chronosieve_answer.py
│   ├── test_mock_kv_direct_runtime.py
│   ├── inspect_runtime_context.py
│   └── inspect_kv_direct_contract.py
│
└── src/
    ├── agent/
    │   └── state_loop.py
    │
    ├── tools/
    │   ├── data_parser.py
    │   ├── eloquent_tools.py
    │   └── sequence_mapper.py
    │
    ├── indexer.py
    │
    └── chronosieve/
        ├── core/
        │   ├── schemas.py
        │   ├── storage.py
        │   ├── carry_packet.py
        │   ├── sieve_worker.py
        │   ├── policy_governor.py
        │   ├── session.py
        │   ├── memory_access.py
        │   ├── rehydration.py
        │   ├── runtime_context.py
        │   ├── runtime_backend.py
        │   ├── kv_sync.py
        │   ├── autonomous_executor.py
        │   ├── autonomous_state.py
        │   ├── resource_adapter.py
        │   ├── role_briefing.py
        │   └── utils.py
        │
        ├── adapters/
        │   ├── mike/
        │   │   ├── trace_parser.py
        │   │   └── session_factory.py
        │   ├── dicom/
        │   │   └── dicom_resource_adapter.py
        │   └── generic/
        │       └── synthetic_resource_adapter.py
        │
        └── backends/
            ├── kv_direct_contract.py
            ├── mock_kv_direct_runtime.py
            ├── llamacpp_backend.py
            ├── llamacpp_prompt_cache_backend.py
            └── llamacpp_prompt_cache_live.py
```

---

## Requirements

Recommended:

* Python 3.10+
* Streamlit
* LangChain
* langchain-openai
* pandas
* pydicom
* numpy
* Pillow
* llama.cpp server with OpenAI-compatible `/v1` endpoint
* Gemma 4 GGUF model
* Gemma multimodal projector if using image tools

Install:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a local `.env` file. Do not commit it.

```bash
cp .env.example .env
```

Example:

```bash
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=local-dummy-key
LLM_MODEL=gemma-4-31B-it-Q4_K_M.gguf
```

The local key is only for the OpenAI-compatible llama.cpp endpoint. It is not an OpenAI key.

---

## Data Setup

This repository does not include raw DICOM files or model weights.

The demo uses the public, de-identified UPENN-GBM dataset from The Cancer Imaging Archive.

`patient_index.csv` is a lightweight local index used by the tools. If your DICOM files live in a different folder, regenerate or edit the file paths before running the app.

Expected columns include:

```text
PatientID
StudyDate
Modality
Z_Coordinate
FilePath
```

Do not commit raw DICOMs, generated patient images, private logs, `.env` files, or model weights.

---

## Run the Streamlit App

Start your local llama.cpp server first, then run:

```bash
streamlit run app.py
```

The app exposes:

* doctor-facing chat,
* tool-grounded answers,
* extracted MRI images,
* execution trace,
* tool inputs,
* tool observations,
* token estimates.

---

## Run a Basic LLM Sanity Check

```bash
python - <<'PY'
from src.agent.state_loop import llm
print(llm.invoke("Reply with exactly: server ok").content)
PY
```

Expected:

```text
server ok
```

---

## Run ChronoSieve With Mike

This wraps the existing Mike task agent with ChronoSieve memory governance:

```bash
python scripts/run_chronosieve_mike_turn.py
```

Expected output includes:

```text
CHRONOSIEVE-WRAPPED ANSWER
task_id
archive_id
candidate_count
memory_event_count
carry_packet_token_estimate
storage_dir
cautions
```

This demonstrates the core loop:

```text
Mike answer + trace
  → evidence archive
  → memory candidates
  → Sieve Worker decision
  → Policy Governor validation
  → ledger + carry packet
```

---

## Run ChronoSieve Runtime / KV Contract Checks

ChronoSieve includes planned runtime and KV-direct contract layers. These do not mutate llama.cpp KV cache directly yet; they define and simulate how memory segments should be staged in future runtimes.

Inspect runtime context:

```bash
python scripts/inspect_runtime_context.py
```

Inspect KV-direct contract:

```bash
python scripts/inspect_kv_direct_contract.py
```

Test mock KV runtime:

```bash
python scripts/test_mock_kv_direct_runtime.py
```

---

## How ChronoSieve Works

ChronoSieve separates memory into multiple layers:

| Layer                   | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| Evidence archive        | Stores raw task evidence and traces externally |
| Task ledger             | Compact task-level state                       |
| Memory events           | Committed memory decisions                     |
| Carry packet            | Bounded working memory for the next turn       |
| Alias memory            | Procedural/canonical-name learnings            |
| Correction registry     | Reference-frame changes and invalidations      |
| Non-authoritative items | Useful claims that must not become truth       |

The important principle:

```text
Raw context is not memory.
A context window is a workspace.
ChronoSieve governs what deserves future attention.
```

---

## Memory Status Types

ChronoSieve can classify memory as:

| Status              | Meaning                               |
| ------------------- | ------------------------------------- |
| `active`            | Useful for near-future continuity     |
| `archived`          | Stored for audit, not carried forward |
| `fossil`            | Durable protected memory              |
| `non_authoritative` | Useful but not trusted as truth       |
| `invalidated`       | Must not be reused as truth           |
| `decay`             | Low-value or redundant residue        |

---

## The Sieve Worker + Policy Governor Split

ChronoSieve uses two layers:

### 1. Gemma Sieve Worker

The Sieve Worker reviews compact candidate briefings and recommends what should happen to each memory candidate.

It does not solve the user task again. It only decides memory consequences.

### 2. Python Policy Governor

The Policy Governor enforces hard safety invariants.

Examples:

* Brain-generated claims are not automatically authoritative.
* Proxy results must not become truth for the original task.
* Large raw payloads stay archived.
* High-risk caveats survive.
* Validated alias learnings can be promoted.

This split lets the model provide judgment while Python enforces rules.

---

## Adapters

ChronoSieve is designed to be domain-neutral. Adapters convert a task agent’s outputs into generic memory candidates.

### Mike Adapter

Located at:

```text
src/chronosieve/adapters/mike/
```

The Mike adapter parses LangChain-style tool traces and extracts:

* DICOM file paths,
* PNG artifact paths,
* patient IDs,
* study dates,
* selected Z-coordinates,
* modality alias learnings,
* caveats,
* tool errors,
* proxy-risk patterns,
* task-agent claims.

### DICOM Adapter

Located at:

```text
src/chronosieve/adapters/dicom/
```

Used for resource-level DICOM evidence processing and sequential file workflows.

### Generic Adapter

Located at:

```text
src/chronosieve/adapters/generic/
```

Used for non-clinical or synthetic resource workflows.

---

## Use ChronoSieve With Your Own Agent

ChronoSieve can wrap any agent that returns a dictionary with an answer and trace.

Minimal shape:

```python
def my_task_agent(user_request: str, history: list[dict[str, str]]) -> dict:
    return {
        "answer": "Final answer from your agent.",
        "trace": [
            {
                "tool": "example_tool",
                "tool_input": {"query": "example"},
                "observation": "Tool output here."
            }
        ],
        "image_paths": [],
        "latency_seconds": 1.23,
        "estimated_response_tokens": 100,
        "success": True,
        "error": None,
    }
```

Then create a ChronoSieve session:

```python
from src.chronosieve.core.storage import ChronoSieveStorage
from src.chronosieve.core.sieve_worker import GemmaSieveWorker
from src.chronosieve.core.session import ChronoSieveSession
from src.chronosieve.adapters.mike.trace_parser import MikeTraceParser
from src.agent.state_loop import llm

storage = ChronoSieveStorage(
    session_id="my_session",
    root_dir="storage/chronosieve_sessions",
    adapter_name="my_adapter",
    backend_model="local-gemma"
)

session = ChronoSieveSession(
    session_id="my_session",
    task_agent_callable=my_task_agent,
    sieve_worker=GemmaSieveWorker(llm=llm),
    storage=storage,
    trace_parser=MikeTraceParser(),  # replace with your own adapter/parser for non-Mike agents
)

result = session.handle_turn("Analyze this task.")
print(result["answer"])
print(result["carry_packet_token_estimate"])
print(result["storage_dir"])
```

For a non-clinical agent, implement a parser that converts your tool traces into `MemoryCandidate` objects.

---

## What Was Demonstrated

### Clinical workflow demo

The Streamlit demo shows:

* patient timeline reconstruction,
* baseline/follow-up scan date detection,
* DTI availability checks,
* modality recovery,
* T1 post-contrast slice retrieval near target Z,
* DICOM-to-PNG rendering,
* multimodal MRI review,
* exact file-path recall,
* correction handling,
* metadata vs image-observation separation,
* safety caveats.

### Memory continuity proof

A 40-turn memory test produced:

| Metric                   |      Result |
| ------------------------ | ----------: |
| Dialogue turns processed |          40 |
| Evidence archive rows    |          40 |
| Governed memory events   |          87 |
| Ledger rows              |          40 |
| Active carry packet      | ~513 tokens |

### DICOM evidence crawl proof

A 20-file DICOM autonomous crawl produced:

| Metric                   |     Result |
| ------------------------ | ---------: |
| Selected DICOM resources |         20 |
| Completed                |         20 |
| Failed                   |          0 |
| Skipped                  |          0 |
| Memory events            |         40 |
| Archive rows             |         20 |
| Unique archive refs      |         20 |
| Carry packet size        | 804 tokens |
| KV-direct required       |         No |

---

## Safety Boundary

This project is for research and demonstration only.

Safe claims:

* Demonstrates local, tool-grounded clinical evidence navigation.
* Helps navigate timelines, metadata, image slices, and evidence paths.
* Preserves auditability, caveats, and corrections across long workflows.
* Uses Gemma 4 locally through llama.cpp.
* Shows memory governance for long-running AI tasks.

Do not use this project for:

* diagnosis,
* treatment recommendations,
* emergency decisions,
* replacing medical professionals,
* unsupervised patient care,
* production clinical deployment without validation, security review, and regulatory assessment.

All image interpretations are assistive and non-authoritative unless validated by qualified clinicians.

---

## Dataset Attribution

MRI data used in the demo comes from the public, de-identified UPENN-GBM collection hosted by The Cancer Imaging Archive.

Recommended citation:

```text
Bakas, S. et al. The UPenn-GBM data collection: Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the Hospital of the University of Pennsylvania. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.709X-DN49
```

Use of this repository should follow the dataset license and attribution requirements.

---

## Kaggle Submission Links


```text
Kaggle writeup:
https://kaggle.com/competitions/gemma-4-good-hackathon/writeups/chronosieve-memory-os-local-gemma-4-clinical-copi

Short demo video:
https://youtu.be/QxgZRKBtEQg

Live full demo:
https://youtu.be/rssFc_CLC-8

Pitch deck PDF:
https://drive.google.com/file/d/1yKHCjVWoqfojLpJ-njEpvtbrMIDMmYHk/view?usp=sharing
```

---

## Project Thesis

Bigger context windows alone do not solve long evidence workflows.

A long-horizon AI agent needs to know:

* what to remember,
* what to archive,
* what to rehydrate,
* what was corrected,
* what is only a model interpretation,
* and what must not become truth.

Mike is the clinical wedge.
ChronoSieve is the memory-governance layer.

Together, they demonstrate a path toward local, private, auditable AI agents that can work across many files and many turns without losing the evidence trail.
