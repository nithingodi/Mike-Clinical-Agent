import os
import uuid
import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import langchain
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

from src.tools.data_parser import extract_image_from_dicom
from src.tools.eloquent_tools import compute_eloquent_displacement


# ==========================================
# 0. LOCAL ENV / CONFIG
# ==========================================

load_dotenv()

# Never hardcode ngrok URLs, API keys, or cloud keys in source code.
# Put local values in .env:
# LLM_BASE_URL=http://localhost:8000/v1
# LLM_API_KEY=sk-local-run
# LLM_MODEL=gemma-4-31B-it-Q4_K_M.gguf
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-local-run")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-31B-it-Q4_K_M.gguf")

# Public/demo default: debug off.
# Enable locally with LANGCHAIN_DEBUG=true
langchain.debug = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

DB_PATH = os.getenv("PATIENT_INDEX_PATH", "patient_index.csv")
BASELINE_LOG_DIR = Path(os.getenv("BASELINE_LOG_DIR", "baseline_logs"))
BASELINE_LOG_DIR.mkdir(exist_ok=True)


# ==========================================
# 1. UTILS
# ==========================================

def estimate_tokens(text: str) -> int:
    """
    Rough proxy token estimator.
    Good enough for baseline trend tracking.
    Later this can be replaced with a tokenizer endpoint.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def history_to_text(history_array: list) -> str:
    parts = []
    for m in history_array:
        role = m.get("role", "unknown")
        content = str(m.get("content", ""))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def append_baseline_log(record: dict, path: str | None = None) -> None:
    log_path = Path(path) if path else BASELINE_LOG_DIR / "baseline_turns.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==========================================
# 2. LLM — defined early so tools can reference it
# ==========================================

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=0.3,
    top_p=0.9,
    presence_penalty=0.6,
    frequency_penalty=0.6,
    max_tokens=4096,
    model_kwargs={
        "tool_choice": "auto"
    },
    # Expanded stop sequence catch-all for local LangChain agents
    stop=["<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<eos>", "Observation:"]
)


# ==========================================
# 3. LOAD CLINICAL DATABASE
# ==========================================

def _load_db() -> pd.DataFrame:
    expected_cols = ["PatientID", "StudyDate", "Modality", "Z_Coordinate", "FilePath"]

    if not os.path.exists(DB_PATH):
        print(f"⚠️  Warning: {DB_PATH} not found. Tools will return empty results.")
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(DB_PATH)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Warning: {DB_PATH} is missing columns: {missing}")

    if "StudyDate" in df.columns:
        df["StudyDate"] = df["StudyDate"].astype(str).str.replace("-", "", regex=False)

    return df


clinical_db = _load_db()


# ==========================================
# 4. TOOLS
# ==========================================

@tool
def query_patient_timeline(patient_id: str) -> str:
    """
    ALWAYS call this first. Resolves partial/shorthand patient IDs to the
    canonical ID and returns their full scan timeline grouped by date and modality.
    """
    if clinical_db.empty:
        return "Error: Clinical database is not loaded."

    if "PatientID" not in clinical_db.columns:
        return "Error: Clinical database does not contain PatientID column."

    matches = clinical_db[
        clinical_db["PatientID"].astype(str).str.contains(patient_id, na=False, regex=False)
    ]["PatientID"].unique()

    if len(matches) == 0:
        return f"Error: No patient matching '{patient_id}' found in the database."

    canonical_id = matches[0]
    df = clinical_db[clinical_db["PatientID"] == canonical_id]

    summary = (
        df.groupby(["StudyDate", "Modality"])
        .size()
        .reset_index(name="SliceCount")
    )

    return json.dumps({
        "canonical_id": canonical_id,
        "timeline": summary.to_dict(orient="records")
    }, indent=2)


@tool
def get_anatomically_aligned_slice(
    patient_id: str,
    date: str,
    modality: str,
    target_z: float = None
) -> str:
    """
    Retrieves the DICOM file path for a specific scan. Matches the closest
    axial slice to target_z, or uses the median slice if target_z is omitted.
    Returns canonical patient ID, file path, and Z coordinate.
    """
    if clinical_db.empty:
        return "Error: Clinical database is not loaded."

    required = ["PatientID", "StudyDate", "Modality", "Z_Coordinate", "FilePath"]
    missing = [c for c in required if c not in clinical_db.columns]
    if missing:
        return f"Error: Clinical database missing required columns: {missing}"

    normalised_date = str(date).replace("-", "")

    mask = (
        clinical_db["PatientID"].astype(str).str.contains(patient_id, na=False, regex=False)
        & (clinical_db["StudyDate"].astype(str) == normalised_date)
        & (clinical_db["Modality"].astype(str) == modality)
    )

    df = clinical_db[mask]

    if df.empty:
        return (
            f"Error: No {modality} scan found for patient matching '{patient_id}' "
            f"on {date}. Verify the date and modality with query_patient_timeline first."
        )

    df = df.copy()
    df["Z_Coordinate"] = pd.to_numeric(df["Z_Coordinate"], errors="coerce")
    df = df.dropna(subset=["Z_Coordinate"])

    if df.empty:
        return (
            f"Error: Matching {modality} scan exists, but no usable Z_Coordinate "
            f"values were found for patient '{patient_id}' on {date}."
        )

    target = float(target_z) if target_z is not None else float(df["Z_Coordinate"].median())
    idx = (df["Z_Coordinate"] - target).abs().idxmin()
    selected = df.loc[idx]

    return json.dumps({
        "canonical_id": selected["PatientID"],
        "file_path": selected["FilePath"],
        "z_coordinate": float(selected["Z_Coordinate"])
    }, indent=2)


@tool
def process_mri_slice(file_path: str) -> str:
    """
    Converts a DICOM file to a PNG image for visual analysis.
    Input must be a file_path returned by get_anatomically_aligned_slice.
    Do NOT construct paths manually.
    Returns the path to the saved PNG.
    """
    if not os.path.exists(file_path):
        return (
            f"Error: DICOM file not found at '{file_path}'. "
            "Use get_anatomically_aligned_slice to get a valid path."
        )

    unique_filename = f"scan_{uuid.uuid4().hex[:8]}.png"

    try:
        saved_path = extract_image_from_dicom(file_path, output_filename=unique_filename)
        return json.dumps({"png_path": saved_path})
    except Exception as e:
        return f"Error processing DICOM: {str(e)}"


@tool
def visually_analyze_mri(image_paths: list[str], clinical_question: str) -> str:
    """
    Sends one or more PNG images to the vision model for clinical analysis.
    For longitudinal comparison, pass both baseline and follow-up paths together.
    image_paths must be png_path values returned by process_mri_slice.
    """
    valid_paths = [p for p in image_paths if os.path.exists(p)]

    if not valid_paths:
        return "Error: None of the provided image paths exist. Run process_mri_slice first."

    contents = [{"type": "text", "text": clinical_question}]

    for path in valid_paths:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"}
        })

    try:
        # Direct LLM call — bypasses the agent loop for vision/multimodal review.
        response = llm.invoke([HumanMessage(content=contents)])
        return str(getattr(response, "content", response))
    except Exception as e:
        return f"Vision analysis failed: {str(e)}"


@tool
def calculate_functional_shift(
    patient_id: str,
    structure_tag: str,
    baseline_date: str,
    followup_date: str
) -> str:
    """
    Calculates the spatial displacement of eloquent brain structures
    such as language cortex or motor cortex between two scan dates.
    Use when the user asks about shift, displacement, or movement of functional areas.
    """
    try:
        return compute_eloquent_displacement(
            patient_id, structure_tag, baseline_date, followup_date
        )
    except Exception as e:
        return f"Error computing functional shift: {str(e)}"


# ==========================================
# 5. AGENT CONFIGURATION
# ==========================================

tools = [
    query_patient_timeline,
    get_anatomically_aligned_slice,
    process_mri_slice,
    visually_analyze_mri,
    calculate_functional_shift,
]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are 'Mike', a deterministic Neuro-Oncology Copilot. "
        "You assist neurosurgeons and clinical researchers by retrieving and analyzing patient MRI data. "
        "You have no knowledge of patients outside the provided tools. Never guess or fabricate data.\n\n"

        "SAFETY BOUNDARY:\n"
        "- This is a research/demo assistant, not a diagnostic medical device.\n"
        "- Do not provide treatment recommendations.\n"
        "- Do not claim certainty from a single image slice.\n"
        "- Always preserve uncertainty and tool errors.\n"
        "- Final clinical interpretation must come from qualified clinicians.\n\n"

        "WORKFLOW RULES, follow in order:\n"
        "1. ALWAYS call `query_patient_timeline` first to resolve the patient ID and confirm available scans.\n"
        "2. Use `get_anatomically_aligned_slice` to retrieve the DICOM path for a specific scan.\n"
        "3. Use `process_mri_slice` to convert a DICOM to PNG before any visual analysis.\n"
        "4. Use `visually_analyze_mri` with the PNG path(s) for longitudinal or single-scan analysis.\n"
        "5. Use `calculate_functional_shift` only when displacement of eloquent areas is asked about.\n\n"

        "If a tool returns an error, report the error clearly and do not proceed with fabricated data.\n\n"

        "=== ONE-SHOT EXECUTION TRACE ===\n"
        "User: Pull the timeline for UPENN-GBM-00045.\n"
        "Thought: The user is asking for the MRI timeline for a specific patient. I must use the query_patient_timeline tool.\n"
        "Tool Call: `query_patient_timeline` with patient_id='UPENN-GBM-00045'.\n"
        "Tool Result: {{\"canonical_id\": \"UPENN-GBM-00045\", \"timeline\": [{{\"StudyDate\": \"20051130\", \"Modality\": \"T1-Post\"}}]}}\n"
        "Final Answer: The MRI timeline for UPENN-GBM-00045 includes a baseline scan on Nov 30, 2005, containing a T1-Post sequence.\n"
        "================================"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
    max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)


# ==========================================
# 6. RESPONSE / TRACE HELPERS
# ==========================================

def _strip_thinking_tokens(text: str) -> str:
    """
    Remove Gemma 4 thinking channel tokens that can leak through llama-server
    into the final output.
    """
    text = str(text)
    text = re.sub(r"<\|channel\>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.DOTALL)
    return text.strip()


def _convert_history(history_array: list) -> list:
    """
    Convert [{role, content}, ...] dicts to LangChain message objects
    for MessagesPlaceholder. Preserves proper turn structure.
    """
    mapping = {
        "human": HumanMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }

    messages = []

    for m in history_array:
        role = str(m.get("role", "human")).lower()
        cls = mapping.get(role, HumanMessage)
        messages.append(cls(content=str(m.get("content", ""))))

    return messages


def extract_png_paths_from_text(text: str) -> list[str]:
    """
    Extract generated PNG paths from arbitrary text/tool observations.
    Handles scan_<hex>.png and longer variants.
    """
    if not text:
        return []

    matches = re.findall(r'[\w/\\.-]*scan_[a-f0-9]{6,16}\.png', str(text))
    return sorted(set(matches))


def _final_text_from_agent_output(out) -> str:
    if isinstance(out, list):
        text_blocks = [
            _strip_thinking_tokens(
                item.get("text", "") if isinstance(item, dict) else str(item)
            )
            for item in out
        ]
        return next(
            (t for t in reversed(text_blocks) if t),
            "No response generated."
        )

    return _strip_thinking_tokens(str(out))


# ==========================================
# 7. PUBLIC INVOCATION FUNCTIONS
# ==========================================

def invoke_mike(user_prompt: str, history_array: list) -> str:
    started = time.time()

    history_text = history_to_text(history_array)

    baseline_record = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "user_prompt": user_prompt,
        "history_message_count": len(history_array),
        "history_char_count": len(history_text),
        "estimated_history_tokens": estimate_tokens(history_text),
        "user_prompt_char_count": len(user_prompt),
        "estimated_user_prompt_tokens": estimate_tokens(user_prompt),
        "estimated_total_input_tokens_proxy": estimate_tokens(history_text + "\n" + user_prompt),
        "success": None,
        "error": None,
    }

    try:
        response = agent_executor.invoke({
            "input": user_prompt,
            "chat_history": _convert_history(history_array),
        })

        final_text = _final_text_from_agent_output(response.get("output", ""))

        baseline_record.update({
            "success": True,
            "latency_seconds": round(time.time() - started, 3),
            "response_char_count": len(final_text),
            "estimated_response_tokens": estimate_tokens(final_text),
        })

        append_baseline_log(baseline_record)
        return final_text

    except Exception as e:
        baseline_record.update({
            "success": False,
            "latency_seconds": round(time.time() - started, 3),
            "error": str(e),
        })

        append_baseline_log(baseline_record)
        return f"⚠️ Agent error: {str(e)}"


def invoke_mike_with_trace(user_prompt: str, history_array: list) -> dict:
    """
    Returns final answer plus visible execution trace and image paths.
    Used by the Streamlit app/debug UI.
    """
    started = time.time()
    history_text = history_to_text(history_array)

    baseline_record = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "user_prompt": user_prompt,
        "history_message_count": len(history_array),
        "history_char_count": len(history_text),
        "estimated_history_tokens": estimate_tokens(history_text),
        "user_prompt_char_count": len(user_prompt),
        "estimated_user_prompt_tokens": estimate_tokens(user_prompt),
        "estimated_total_input_tokens_proxy": estimate_tokens(history_text + "\n" + user_prompt),
        "success": None,
        "error": None,
        "tool_trace": [],
        "image_paths": [],
    }

    try:
        response = agent_executor.invoke({
            "input": user_prompt,
            "chat_history": _convert_history(history_array),
        })

        final_text = _final_text_from_agent_output(response.get("output", ""))

        trace = []
        image_paths = set(extract_png_paths_from_text(final_text))

        for step in response.get("intermediate_steps", []):
            try:
                action, observation = step

                tool_name = getattr(action, "tool", "unknown_tool")
                tool_input = getattr(action, "tool_input", None)
                action_log = getattr(action, "log", "")

                obs_text = str(observation)
                image_paths.update(extract_png_paths_from_text(obs_text))

                trace.append({
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "observation": obs_text,
                    "action_log": _strip_thinking_tokens(str(action_log)),
                    "observation_token_estimate": estimate_tokens(obs_text),
                })

            except Exception as trace_error:
                trace.append({
                    "tool": "trace_parse_error",
                    "tool_input": None,
                    "observation": str(trace_error),
                    "action_log": "",
                    "observation_token_estimate": 0,
                })

        image_paths = sorted(p for p in image_paths if os.path.exists(p))

        baseline_record.update({
            "success": True,
            "latency_seconds": round(time.time() - started, 3),
            "response_char_count": len(final_text),
            "estimated_response_tokens": estimate_tokens(final_text),
            "tool_trace": trace,
            "image_paths": image_paths,
        })

        append_baseline_log(baseline_record)

        return {
            "answer": final_text,
            "trace": trace,
            "image_paths": image_paths,
            "latency_seconds": baseline_record["latency_seconds"],
            "estimated_response_tokens": baseline_record["estimated_response_tokens"],
            "success": True,
            "error": None,
        }

    except Exception as e:
        baseline_record.update({
            "success": False,
            "latency_seconds": round(time.time() - started, 3),
            "error": str(e),
        })

        append_baseline_log(baseline_record)

        return {
            "answer": f"⚠️ Agent error: {str(e)}",
            "trace": [],
            "image_paths": [],
            "latency_seconds": baseline_record["latency_seconds"],
            "estimated_response_tokens": 0,
            "success": False,
            "error": str(e),
        }