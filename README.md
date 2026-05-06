# Mike: Local Gemma 4 Clinical Copilot

**A privacy-first clinical research copilot for longitudinal MRI review, powered by local Gemma 4, tool calling, multimodal reasoning, and memory continuity.**

> Research/demo only. This project is not a diagnostic medical device, treatment recommendation system, or replacement for a radiologist, oncologist, neurosurgeon, or clinical team.

---

## Overview

Doctors and clinical researchers do not review complex patients in one clean prompt. They move across scan dates, DICOM folders, MRI modalities, image slices, notes, tool outputs, prior findings, hypotheses, and corrections.

This project, **Mike**, is a local Gemma 4 clinical copilot designed to help users ask natural-language questions across longitudinal patient MRI data. The current demo focuses on UPENN-GBM neuro-oncology MRI workflows.

Mike can:

* query a patient’s scan timeline,
* identify available sessions and modalities,
* retrieve anatomically aligned DICOM slices by physical Z-coordinate,
* render DICOM slices into viewable PNG previews,
* use Gemma 4 multimodal reasoning for image review,
* expose tool traces and evidence paths for auditability,
* preserve useful memory and caveats across follow-up questions.

The goal is not to automate diagnosis. The goal is to reduce the time clinicians and researchers spend navigating evidence.

---

## Why Gemma 4

This project uses Gemma 4 as the local intelligence layer for:

* **Reasoning:** decomposing clinical-style questions into tool calls.
* **Function calling:** choosing timeline, DICOM retrieval, image processing, and analysis tools.
* **Structured JSON handling:** reading and routing tool outputs.
* **Multimodal understanding:** reviewing rendered MRI slice images.
* **Local/private deployment:** running through llama.cpp so sensitive data does not need to leave the local environment.

The working setup uses a Gemma 4 31B GGUF model served through a llama.cpp OpenAI-compatible endpoint.

---

## Architecture

```text
Doctor / researcher question
  ↓
Gemma 4 task agent
  ↓
Clinical tools
  ↓
Timeline / DICOM / image evidence
  ↓
Grounded answer + visible execution trace
```

The broader research layer behind the demo is **ChronoSieve**, a memory-continuity architecture for long evidence workflows.

```text
Tool trace / evidence / answer
  ↓
Evidence archive
  ↓
Memory candidate extraction
  ↓
Memory governance
  ↓
Bounded carry/runtime context
  ↓
Future grounded answer
```

ChronoSieve treats the context window as a workspace, not a database. Instead of stuffing every previous message and tool output into the next prompt, it decides what should remain active, what should be archived, what should be treated as non-authoritative, and what should be rehydrated only when needed.

---

## Demo Capabilities

### 1. Patient timeline retrieval

Mike can answer questions such as:

```text
What scan sessions are available for UPENN-GBM-00045?
```

It returns scan dates, modality availability, and longitudinal context.

### 2. DICOM slice retrieval

Mike can retrieve MRI slices by physical Z-coordinate:

```text
Fetch the T1 Post-Contrast slice closest to Z = -15mm for the baseline and follow-up sessions.
```

It identifies the closest available DICOM files and returns their paths, dates, modalities, and Z-coordinates.

### 3. DICOM-to-image rendering

Selected DICOM files can be converted into PNG previews for visual inspection.

### 4. Multimodal image review

Gemma 4 can inspect rendered MRI slices and provide research-assistive observations with safety caveats.

### 5. Evidence trace visibility

The app exposes intermediate tool calls, inputs, observations, and evidence paths so users can audit how an answer was produced.

### 6. Memory continuity

The system can preserve important findings and caveats across follow-up questions without carrying the entire transcript.

---

## Memory Continuity Stress Test

To validate long-horizon behavior, I replayed 40 turns of a longitudinal conversation one turn at a time through the memory-governance loop.

| Metric                   |      Result |
| ------------------------ | ----------: |
| Dialogue turns processed |          40 |
| Evidence archive rows    |          40 |
| Governed memory events   |          87 |
| Ledger rows              |          40 |
| Active carry packet      | ~513 tokens |

Then I asked exact recall and temporal reasoning questions through the full runtime path:

```text
MemoryAccessController
  ↓
KV/runtime memory plan
  ↓
Governed runtime context
  ↓
AnswerSynthesizer
```

Example results:

* “When did Caroline go to the support group?” → **7 May 2023**, resolved from “yesterday” anchored to 8 May 2023.
* “What did Caroline research?” → **adoption agencies**.
* “When is Melanie planning on going camping?” → **June 2023**.
* “When did Melanie run a charity race?” → The system rehydrated archived evidence and answered **May 20, 2023**, because the raw dialogue said “last Saturday” from a 25 May 2023 session.

This matters for clinical workflows because doctors ask questions across time. The system must preserve evidence trails, caveats, and source grounding rather than producing plausible one-turn answers.

---

## DICOM Evidence Crawl Proof

A 20-file DICOM autonomous evidence crawl was run on patient `UPENN-GBM-00045`.

| Metric                   |                                               Result |
| ------------------------ | ---------------------------------------------------: |
| Selected DICOM resources |                                                   20 |
| Completed                |                                                   20 |
| Failed                   |                                                    0 |
| Skipped                  |                                                    0 |
| Memory events            |                                                   40 |
| Archive rows             |                                                   20 |
| Unique archive refs      |                                                   20 |
| Carry packet size        |                                           804 tokens |
| KV-direct required       |                                                   No |
| Dates covered            |                                   20051130, 20060323 |
| Series covered           | T2, DTI, perfusion, T1 axial, T1 stealth-post, FLAIR |

This validates the core loop:

```text
DICOM resource
  → metadata observation
  → memory candidates
  → memory governance
  → archive + ledger
  → bounded carry packet
```

---

## Repository Layout

```text
app.py                         # Streamlit app
patient_index.csv              # Patient index / demo metadata
requirements.txt               # Python dependencies

src/
  agent/
    state_loop.py              # Gemma 4 clinical task agent

  tools/
    data_parser.py             # Dataset / DICOM parsing helpers
    eloquent_tools.py          # Clinical tools used by the agent
    sequence_mapper.py         # Modality / sequence matching helpers

  indexer.py                   # Dataset indexing utilities

  memory_continuity/
    reference_memory_layer.py  # Public reference scaffold for memory continuity

notebooks/
  01_clinical_eda.ipynb
  patient_token_budget_results.json

docs/
  architecture.md
  safety_boundary.md
  kaggle_writeup.md
  demo_script.md

assets/
  architecture_diagram.png
  screenshots/
```

---

## Requirements

Recommended environment:

* Python 3.10+
* Streamlit
* LangChain / langchain-openai / langchain-core
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

## llama.cpp Server

The local app expects a llama.cpp OpenAI-compatible endpoint.

Example sanity check:

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

## Run the App

```bash
streamlit run app.py
```

---

## Safety Boundary

This project is for research and demonstration only.

Safe claims:

* Demonstrates local/tool-grounded clinical-data workflows.
* Helps navigate timelines, scan metadata, image slices, and evidence paths.
* Preserves auditability and caveats across long workflows.
* Uses Gemma 4 locally through llama.cpp.

Do not use this project for:

* clinical diagnosis,
* treatment recommendations,
* emergency decisions,
* replacing qualified medical professionals,
* unsupervised patient care.

All image interpretations should be treated as assistive and non-authoritative unless validated by appropriate clinicians.

---

## Public / Private Boundary

This public repository contains the Gemma 4 clinical copilot demo and a reference memory-continuity scaffold.

Some research internals are intentionally not fully open-sourced in this competition repository, including advanced memory-governance prompts, runtime-memory controller internals, and experimental KV-direct work. The public code demonstrates the Gemma 4 app, tool use, evidence handling, and user-facing clinical workflow.

---

## Project Thesis

Bigger context windows alone do not solve long clinical workflows.

Doctors need systems that can inspect evidence, preserve what matters, archive full traces, remember caveats, and rehydrate exact proof when challenged.

**Mike** demonstrates how local Gemma 4 can help clinicians and researchers navigate patient histories and MRI evidence faster, more privately, and with better auditability.
