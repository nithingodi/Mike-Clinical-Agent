from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """
    Rough v1 estimator. Same idea as current Mike baseline:
    len(text)//4 is good enough for trend tracking.
    """
    if not text:
        return 0
    return max(1, len(str(text)) // 4)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_str(obj: Any, max_chars: int | None = None) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction for Gemma structured output.
    Handles code fences and extra text around JSON.
    """
    if not text:
        return {}

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return {}

    return {}
