# QnA 질의 로그(log_query)와 검색 평가 로그(log_retrieval_eval)를 저장·검색·삭제한다.
# 저장 방식은 환경변수(QNA_LOG_STORE)로 메모리 또는 JSONL 파일 중 선택한다.
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.qna_config import load_runtime_policy
from services.qna_retention import purge_old_records

QUERY_LOGS: list[dict[str, Any]] = []
RETRIEVAL_EVAL_LOGS: list[dict[str, Any]] = []
QUERY_LOG_FILE = "qna_query_logs.jsonl"
RETRIEVAL_LOG_FILE = "qna_retrieval_eval_logs.jsonl"


def reset_logs() -> None:
    QUERY_LOGS.clear()
    RETRIEVAL_EVAL_LOGS.clear()
    policy = load_runtime_policy()
    if policy.log_store != "jsonl":
        return
    query_path = policy.log_dir / QUERY_LOG_FILE
    retrieval_path = policy.log_dir / RETRIEVAL_LOG_FILE
    for path in (query_path, retrieval_path):
        if path.exists():
            path.unlink()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _matches(record: dict[str, Any], request_id: str | None, status: str | None, date_prefix: str | None) -> bool:
    if request_id and str(record.get("request_id")) != request_id:
        return False
    if status and str(record.get("status")) != status:
        return False
    if date_prefix:
        created_at = str(record.get("created_at", ""))
        if not created_at.startswith(date_prefix):
            return False
    return True


def _log(record: dict[str, Any], log_list: list, filename: str) -> None:
    payload = dict(record)
    payload.setdefault("created_at", datetime.now(UTC).isoformat())
    log_list.append(payload)
    policy = load_runtime_policy()
    if policy.log_store == "jsonl":
        _append_jsonl(policy.log_dir / filename, payload)


def log_query(record: dict[str, Any]) -> None:
    _log(record, QUERY_LOGS, QUERY_LOG_FILE)


def log_retrieval_eval(record: dict[str, Any]) -> None:
    _log(record, RETRIEVAL_EVAL_LOGS, RETRIEVAL_LOG_FILE)


def search_logs(
    *,
    kind: str = "query",
    request_id: str | None = None,
    status: str | None = None,
    date_prefix: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    policy = load_runtime_policy()
    kind = kind.strip().lower()
    if limit <= 0:
        return []
    if kind not in {"query", "retrieval"}:
        raise ValueError(f"Unsupported kind: {kind}")

    if policy.log_store == "jsonl":
        target = policy.log_dir / (QUERY_LOG_FILE if kind == "query" else RETRIEVAL_LOG_FILE)
        rows = _read_jsonl(target)
    else:
        rows = QUERY_LOGS if kind == "query" else RETRIEVAL_EVAL_LOGS

    filtered = [row for row in rows if _matches(row, request_id, status, date_prefix)]
    return filtered[-limit:]


def purge_logs(retention_days: int = 90, dry_run: bool = True) -> dict[str, int]:
    policy = load_runtime_policy()
    if retention_days <= 0:
        raise ValueError("retention_days must be > 0")

    if policy.log_store == "jsonl":
        query_path = policy.log_dir / QUERY_LOG_FILE
        retrieval_path = policy.log_dir / RETRIEVAL_LOG_FILE
        query_rows = _read_jsonl(query_path)
        retrieval_rows = _read_jsonl(retrieval_path)
        query_before = len(query_rows)
        retrieval_before = len(retrieval_rows)
        query_removed = purge_old_records(query_rows, retention_days=retention_days)
        retrieval_removed = purge_old_records(retrieval_rows, retention_days=retention_days)
        if not dry_run:
            _write_jsonl(query_path, query_rows)
            _write_jsonl(retrieval_path, retrieval_rows)
        return {
            "query_before": query_before,
            "query_removed": query_removed,
            "query_after": len(query_rows),
            "retrieval_before": retrieval_before,
            "retrieval_removed": retrieval_removed,
            "retrieval_after": len(retrieval_rows),
        }

    query_before = len(QUERY_LOGS)
    retrieval_before = len(RETRIEVAL_EVAL_LOGS)
    query_rows = list(QUERY_LOGS)
    retrieval_rows = list(RETRIEVAL_EVAL_LOGS)
    query_removed = purge_old_records(query_rows, retention_days=retention_days)
    retrieval_removed = purge_old_records(retrieval_rows, retention_days=retention_days)
    if not dry_run:
        QUERY_LOGS[:] = query_rows
        RETRIEVAL_EVAL_LOGS[:] = retrieval_rows
    return {
        "query_before": query_before,
        "query_removed": query_removed,
        "query_after": len(query_rows),
        "retrieval_before": retrieval_before,
        "retrieval_removed": retrieval_removed,
        "retrieval_after": len(retrieval_rows),
    }
