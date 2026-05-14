# 로그 보존 기간 관리 유틸.
# purge_old_records()는 created_at 기준으로 retention_days를 초과한 레코드를 in-place로 제거한다.
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def purge_old_records(records: list[dict[str, Any]], retention_days: int = 90) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    kept: list[dict[str, Any]] = []
    removed = 0
    for item in records:
        created_at = item.get("created_at")
        if not created_at:
            kept.append(item)
            continue
        try:
            dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            kept.append(item)
            continue
        if dt >= cutoff:
            kept.append(item)
        else:
            removed += 1
    records[:] = kept
    return removed
