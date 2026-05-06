from __future__ import annotations

import os
from typing import Any

import streamlit as st

from src.agent.state_loop import invoke_mike_with_trace
from src.chronosieve.adapters.mike.session_factory import create_mike_chronosieve_session


# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Mike A/B: Baseline vs ChronoSieve",
    page_icon="🧠",
    layout="wide",
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def render_images(image_paths: list[str]) -> None:
    for img_path in image_paths or []:
        if img_path and os.path.exists(img_path):
            st.image(
                img_path,
                caption=f"Extracted: {os.path.basename(img_path)}",
                width=330,
            )


def compact_text(value: Any, max_chars: int = 3500) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated for demo view]"


def render_tool_trace(trace: list[dict[str, Any]] | None, *, title: str) -> None:
    """
    Doctor-facing tool trace.

    This intentionally shows only the clinical work path:
    - which Mike tools were called
    - tool inputs
    - tool observations

    It does NOT expose ChronoSieve internals such as memory events, carry-packet
    building, Sieve decisions, policy-governor rules, archive structure, or KV/runtime
    details. That keeps the demo understandable without giving away the memory OS moat.
    """
    trace = trace or []
    with st.expander(title, expanded=False):
        if not trace:
            st.caption("No visible tool calls were captured for this answer.")
            return

        for i, step in enumerate(trace, start=1):
            tool_name = step.get("tool", "unknown_tool")
            st.markdown(f"**Step {i}: `{tool_name}`**")

            st.markdown("**Tool input**")
            tool_input = step.get("tool_input", {})
            if isinstance(tool_input, dict):
                st.json(tool_input)
            else:
                st.code(compact_text(tool_input), language="text")

            st.markdown("**Tool observation**")
            st.code(compact_text(step.get("observation", "")), language="text")

            token_estimate = step.get("observation_token_estimate")
            if token_estimate is not None:
                st.caption(f"Estimated observation tokens: {token_estimate}")

            if i != len(trace):
                st.divider()


def init_state() -> None:
    if "compare_turns" not in st.session_state:
        st.session_state.compare_turns = []

    if "baseline_history" not in st.session_state:
        st.session_state.baseline_history = [
            {
                "role": "assistant",
                "content": "Hello Doctor. I am baseline Mike. Which patient file should we review today?",
            }
        ]

    if "chronosieve_history" not in st.session_state:
        st.session_state.chronosieve_history = [
            {
                "role": "assistant",
                "content": "Hello Doctor. I am Mike with memory continuity enabled. Which patient file should we review today?",
            }
        ]


def reset_all() -> None:
    st.session_state.compare_turns = []
    st.session_state.baseline_history = [
        {
            "role": "assistant",
            "content": "Hello Doctor. I am baseline Mike. Which patient file should we review today?",
        }
    ]
    st.session_state.chronosieve_history = [
        {
            "role": "assistant",
            "content": "Hello Doctor. I am Mike with memory continuity enabled. Which patient file should we review today?",
        }
    ]

    for key in list(st.session_state.keys()):
        if key.startswith("chronosieve_session::"):
            del st.session_state[key]


def get_chronosieve_session(session_id: str):
    key = f"chronosieve_session::{session_id}"
    if key not in st.session_state:
        st.session_state[key] = create_mike_chronosieve_session(session_id=session_id)
    return st.session_state[key]


def display_answer_card(title: str, result: dict[str, Any] | None, *, trace_title: str) -> None:
    st.markdown(f"### {title}")

    if result is None:
        st.info("No response yet.")
        return

    if result.get("error_display"):
        st.error(result["error_display"])
        return

    st.markdown(result.get("answer", ""))
    render_images(result.get("image_paths", []))
    render_tool_trace(result.get("trace", []), title=trace_title)


def build_history_for_baseline() -> list[dict[str, str]]:
    return [
        {"role": row["role"], "content": row["content"]}
        for row in st.session_state.baseline_history
        if row.get("role") in {"user", "assistant", "system"}
    ]


def build_recent_history_for_chronosieve() -> list[dict[str, str]]:
    # ChronoSieve owns durable continuity internally. The UI only passes recent chat
    # context for conversational smoothness.
    return [
        {"role": row["role"], "content": row["content"]}
        for row in st.session_state.chronosieve_history[-4:]
        if row.get("role") in {"user", "assistant", "system"}
    ]


# ─────────────────────────────────────────────
# State + sidebar
# ─────────────────────────────────────────────

init_state()

st.sidebar.title("🧠 Mike A/B Demo")

run_mode = st.sidebar.radio(
    "Demo mode",
    ["Compare both", "Baseline only", "ChronoSieve only"],
    index=0,
)

session_id = st.sidebar.text_input(
    "Memory-continuity session",
    value="doctor_ab_demo_chronosieve",
)

show_trace = st.sidebar.toggle(
    "Show expandable tool traces",
    value=True,
    help="Shows Mike's clinical tool calls. Does not expose ChronoSieve memory internals.",
)

if st.sidebar.button("Reset demo chat", use_container_width=True):
    reset_all()
    st.rerun()

st.sidebar.markdown(
    """
Doctor-facing comparison UI.

Use the same prompts on both agents and compare the answers.  
Expandable traces show only clinical tool work, not memory-system internals.
    """.strip()
)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("🧠 Mike: Baseline vs Memory-Continuity")
st.markdown(
    "A doctor-facing A/B chat demo over longitudinal UPENN-GBM data. "
    "Type one prompt and compare the original Mike agent against the memory-governed version."
)

st.divider()


# ─────────────────────────────────────────────
# Opening greeting
# ─────────────────────────────────────────────

if not st.session_state.compare_turns:
    if run_mode == "Compare both":
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Baseline Mike")
            st.chat_message("assistant").markdown(
                "Hello Doctor. I am baseline Mike. Which patient file should we review today?"
            )
        with col_b:
            st.markdown("### Memory-Continuity Mike")
            st.chat_message("assistant").markdown(
                "Hello Doctor. I am Mike with memory continuity enabled. Which patient file should we review today?"
            )
    elif run_mode == "Baseline only":
        st.chat_message("assistant").markdown(
            "Hello Doctor. I am baseline Mike. Which patient file should we review today?"
        )
    else:
        st.chat_message("assistant").markdown(
            "Hello Doctor. I am Mike with memory continuity enabled. Which patient file should we review today?"
        )


# ─────────────────────────────────────────────
# Render prior turns
# ─────────────────────────────────────────────

for turn in st.session_state.compare_turns:
    st.chat_message("user").markdown(turn["prompt"])

    if run_mode == "Compare both":
        col_a, col_b = st.columns(2)
        with col_a:
            with st.chat_message("assistant"):
                result = turn.get("baseline")
                if not show_trace and result:
                    result = {**result, "trace": []}
                display_answer_card(
                    "Baseline Mike",
                    result,
                    trace_title="🔎 Baseline Mike execution trace",
                )
        with col_b:
            with st.chat_message("assistant"):
                result = turn.get("chronosieve")
                if not show_trace and result:
                    result = {**result, "trace": []}
                display_answer_card(
                    "Memory-Continuity Mike",
                    result,
                    trace_title="🔎 Memory-Continuity Mike execution trace",
                )
    elif run_mode == "Baseline only":
        with st.chat_message("assistant"):
            result = turn.get("baseline")
            if not show_trace and result:
                result = {**result, "trace": []}
            display_answer_card(
                "Baseline Mike",
                result,
                trace_title="🔎 Baseline Mike execution trace",
            )
    else:
        with st.chat_message("assistant"):
            result = turn.get("chronosieve")
            if not show_trace and result:
                result = {**result, "trace": []}
            display_answer_card(
                "Memory-Continuity Mike",
                result,
                trace_title="🔎 Memory-Continuity Mike execution trace",
            )

    st.divider()


# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────

prompt = st.chat_input("Ask the same clinical workflow prompt to both agents...")

if prompt:
    turn_record: dict[str, Any] = {
        "prompt": prompt,
        "baseline": None,
        "chronosieve": None,
    }

    st.chat_message("user").markdown(prompt)

    if run_mode == "Compare both":
        col_a, col_b = st.columns(2)
    else:
        col_a = st.container()
        col_b = st.container()

    # Baseline path
    if run_mode in {"Compare both", "Baseline only"}:
        with col_a:
            with st.chat_message("assistant"):
                st.markdown("### Baseline Mike")
                try:
                    with st.spinner("Baseline Mike is thinking..."):
                        baseline_history = build_history_for_baseline()
                        baseline_result = invoke_mike_with_trace(prompt, baseline_history)

                    baseline_answer = baseline_result.get("answer", "")
                    baseline_images = baseline_result.get("image_paths", [])
                    baseline_trace = baseline_result.get("trace", [])

                    st.markdown(baseline_answer)
                    render_images(baseline_images)
                    if show_trace:
                        render_tool_trace(
                            baseline_trace,
                            title="🔎 Baseline Mike execution trace",
                        )

                    turn_record["baseline"] = {
                        "answer": baseline_answer,
                        "image_paths": baseline_images,
                        "trace": baseline_trace,
                    }

                    st.session_state.baseline_history.append({"role": "user", "content": prompt})
                    st.session_state.baseline_history.append({"role": "assistant", "content": baseline_answer})

                except Exception as exc:
                    msg = f"Baseline Mike failed: {exc}"
                    st.error(msg)
                    turn_record["baseline"] = {"error_display": msg, "trace": []}

    # ChronoSieve / memory-continuity path
    if run_mode in {"Compare both", "ChronoSieve only"}:
        with col_b:
            with st.chat_message("assistant"):
                st.markdown("### Memory-Continuity Mike")
                try:
                    with st.spinner("Memory-Continuity Mike is thinking..."):
                        chrono_session = get_chronosieve_session(session_id)
                        recent_history = build_recent_history_for_chronosieve()
                        chrono_result = chrono_session.handle_turn(
                            prompt,
                            recent_display_history=recent_history,
                        )

                    chrono_answer = chrono_result.get("answer", "")
                    chrono_images = chrono_result.get("image_paths", [])
                    chrono_trace = chrono_result.get("trace", [])

                    st.markdown(chrono_answer)
                    render_images(chrono_images)
                    if show_trace:
                        render_tool_trace(
                            chrono_trace,
                            title="🔎 Memory-Continuity Mike execution trace",
                        )

                    turn_record["chronosieve"] = {
                        "answer": chrono_answer,
                        "image_paths": chrono_images,
                        "trace": chrono_trace,
                    }

                    st.session_state.chronosieve_history.append({"role": "user", "content": prompt})
                    st.session_state.chronosieve_history.append({"role": "assistant", "content": chrono_answer})

                except Exception as exc:
                    msg = f"Memory-Continuity Mike failed: {exc}"
                    st.error(msg)
                    turn_record["chronosieve"] = {"error_display": msg, "trace": []}

    st.session_state.compare_turns.append(turn_record)
    st.rerun()
