# few_shot_definitions.py에서 few-shot 예시를 로드해 프롬프트 텍스트로 변환한다.
# routing_few_shot_block()은 라우팅 LLM 프롬프트에, answer_llm_few_shot_block()은 답변 LLM 프롬프트에 삽입된다.
"""Load versioned few-shot prompt assets from `src/services/few_shot_definitions.py`."""

from __future__ import annotations

import importlib
import os
from typing import Any

import services.few_shot_definitions as few_shot_definitions


def clear_few_shot_cache() -> None:
    """Reload `few_shot_definitions` so `data` reflects on-disk edits (e.g. tests)."""
    importlib.reload(few_shot_definitions)


def load_few_shot_definitions() -> dict[str, Any] | None:
    """Return parsed data dict from few_shot_definitions.py."""
    return few_shot_definitions.data


def get_task(task_id: str) -> dict[str, Any] | None:
    doc = load_few_shot_definitions()
    if not doc:
        return None
    tasks = doc.get("tasks")
    if not isinstance(tasks, dict):
        return None
    t = tasks.get(task_id)
    return t if isinstance(t, dict) else None


def get_task_examples(task_id: str) -> list[dict[str, Any]]:
    task = get_task(task_id)
    if not task:
        return []
    ex = task.get("examples")
    if not isinstance(ex, list):
        return []
    return [e for e in ex if isinstance(e, dict)]


def format_task_examples_text(
    task_id: str,
    *,
    max_examples: int | None = None,
    example_label: str = "예시",
) -> str:
    """Turn task examples into a single text block for a single-user-message prompt."""
    examples = get_task_examples(task_id)
    if max_examples is not None:
        examples = examples[: max(0, max_examples)]

    parts: list[str] = []
    for i, ex in enumerate(examples, start=1):
        eid = ex.get("id")
        header = f"### {example_label} {i}"
        if isinstance(eid, str) and eid.strip():
            header += f" (`{eid}`)"
        parts.append(header)
        messages = ex.get("messages")
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip() or "user"
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"[{role}]\n{content}")
        parts.append("")
    return "\n".join(parts).strip()


def routing_few_shot_block() -> str:
    """Few-shot text appended to the route LLM single-message prompt."""
    max_raw = (os.getenv("QNA_FEW_SHOT_ROUTING_MAX") or "").strip()
    max_examples: int | None
    if max_raw.isdigit():
        max_examples = int(max_raw)
    else:
        max_examples = None
    return format_task_examples_text("route_classification", max_examples=max_examples)


def answer_llm_few_shot_block() -> str:
    """Concatenated few-shots for future grounded answer LLM (db_api / style)."""
    chunks = [
        format_task_examples_text("db_user_answer_style", example_label="답변 스타일 예시"),
        format_task_examples_text("employee_dept_and_name_lookup", example_label="부서+사원 예시"),
        format_task_examples_text(
            "structured_db_input_normalization", example_label="정형 DB 입력 정규화(§8.2) 참고"
        ),
        format_task_examples_text("regulation_owner_paraphrase", example_label="규정 담당 부서 예시"),
        format_task_examples_text("policy_condition_calculation", example_label="조건 기반 계산 예시"),
    ]
    return "\n\n---\n\n".join(c for c in chunks if c.strip())
