# friday.topic_mapping 테이블로 질문을 업무 카테고리에 매핑해 검색어를 보강한다.
# ILIKE 토큰 매칭 → pgvector 유사도 순으로 시도하며, rag/hybrid에서는 keyword·semantic 쿼리에 topic을 추가한다.
"""Resolve user questions to friday.topic_mapping entries for query enrichment.

topic_mapping 테이블(friday 스키마)을 활용해 질문을 업무 카테고리(topic)로 매핑하고
검색 쿼리와 mobigen_task_owner 조회를 보강한다.

매핑 전략 (우선순위 순):
  1. ILIKE — topic 텍스트 토큰 매칭 (빠름, 정확한 키워드 필요)
  2. Semantic — keyword_embedding / description_embedding pgvector 유사도 (ILIKE 실패 시 fallback)
"""

from __future__ import annotations

import os
import re
from typing import Any

from db.connect_rdb import AsyncPostgresDB
from utils.logger import LOG
from services.qna_utils import _generate_embedding, run_async

_TOPIC_TABLE = "friday.topic_mapping"
_DEFAULT_TIMEOUT = 2.0
_SEM_TIMEOUT = float(os.getenv("QNA_TOPIC_SEM_TIMEOUT", "6.0"))  # OpenAI API 호출 포함
_SIM_THRESHOLD = float(os.getenv("QNA_TOPIC_SIM_THRESHOLD", "0.50"))

# topic/description 매칭에서 제외할 범용 불용어
_TOPIC_STOP: frozenset[str] = frozenset(
    {
        "규정", "사내", "담당", "담당자", "부서", "어떻게", "어디", "어느",
        "알려줘", "알려", "어떤", "무엇", "뭐야", "뭔지", "하려면", "하면",
        "쓰려면", "신청", "받고", "싶어", "있나요", "있어", "해야", "해요",
        "누구야", "누구", "누가", "뭐예요", "무엇인가요", "알고싶어",
    }
)


def _topic_mapper_enabled() -> bool:
    v = os.getenv("QNA_TOPIC_MAPPER", "1").strip().lower()
    return v not in ("0", "false", "off", "no")


def _topic_mapper_timeout() -> float:
    raw = os.getenv("QNA_TOPIC_MAPPER_TIMEOUT", "").strip()
    try:
        return float(raw) if raw else _DEFAULT_TIMEOUT
    except ValueError:
        return _DEFAULT_TIMEOUT


def _extract_tokens(question: str) -> list[str]:
    """한글/영숫자 2글자 이상 토큰을 추출하고 불용어를 제거한다."""
    tokens = re.findall(r"[\uac00-\ud7a3a-zA-Z0-9]{2,}", question)
    return [t for t in tokens if t not in _TOPIC_STOP]


async def _fetch_topic_rows_async(tokens: list[str]) -> list[dict[str, Any]]:
    if not tokens:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        params: list[str] = []
        conditions: list[str] = []
        for token in tokens[:6]:
            idx = len(params) + 1
            params.append(f"%{token}%")
            conditions.append(f"topic ILIKE ${idx}")
        sql = f"""
            SELECT id, topic, description
            FROM {_TOPIC_TABLE}
            WHERE {" OR ".join(conditions)}
            LIMIT 15
        """
        rows = await db.fetch(sql, *params)
        return [dict(r) for r in rows]
    finally:
        await db.close()


def _score_row(row: dict[str, Any], tokens: list[str]) -> int:
    topic = (row.get("topic") or "").lower()
    desc = (row.get("description") or "").lower()
    combined = f"{topic} {desc}"
    return sum(1 for t in tokens if t.lower() in combined)



async def _resolve_topic_semantic_async(query: str) -> dict[str, Any] | None:
    """keyword_embedding pgvector 유사도로 topic 매핑.

    사용자 질문은 키워드 나열에 가깝고 keyword 텍스트도 같은 형태라 벡터 공간이 일치한다.
    description_embedding은 문장 형태라 유사도가 낮게 나오므로 사용하지 않는다.
    """
    tokens = _extract_tokens(query)
    embed_text = " ".join(tokens) if tokens else query
    embedding = _generate_embedding(embed_text, "qna_topic_mapper")
    if embedding is None:
        return None
    vec_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
    db = AsyncPostgresDB()
    await db.connect()
    try:
        rows = await db.fetch(f"""
            SELECT id, topic, description,
                   (1 - (keyword_embedding <=> '{vec_literal}'::vector)) AS kw_sim
            FROM {_TOPIC_TABLE}
            ORDER BY kw_sim DESC
            LIMIT 3
        """)
        if not rows:
            return None
        best = dict(rows[0])
        if best["kw_sim"] < _SIM_THRESHOLD:
            LOG.debug(
                "qna_topic_mapper: semantic kw_sim=%.3f below threshold=%.2f, no match",
                best["kw_sim"], _SIM_THRESHOLD,
            )
            return None
        LOG.debug(
            "qna_topic_mapper: semantic matched topic=%r (kw_sim=%.3f)",
            best["topic"], best["kw_sim"],
        )
        return best
    finally:
        await db.close()


def _run_async(coro, timeout_sec: float) -> Any:
    try:
        return run_async(coro, timeout_sec)
    except Exception as exc:
        LOG.debug("qna_topic_mapper: async fetch failed (%s)", exc)
        return None


def resolve_topic(question: str, timeout_sec: float | None = None) -> dict[str, Any] | None:
    """질문에 가장 잘 매칭되는 topic_mapping 행을 반환한다. 없으면 None.

    1차: topic ILIKE 토큰 매칭
    2차: keyword_embedding / description_embedding pgvector 유사도 (ILIKE 실패 시)
    반환 dict 키: topic, description
    """
    if not _topic_mapper_enabled() or not question.strip():
        return None
    t = timeout_sec if timeout_sec is not None else _topic_mapper_timeout()

    tokens = _extract_tokens(question)
    if tokens:
        rows = _run_async(_fetch_topic_rows_async(tokens), t)
        if rows:
            scored = [(row, _score_row(row, tokens)) for row in rows]
            best_row, best_score = max(scored, key=lambda x: x[1])
            # 단일 토큰 매칭(score=1)은 false positive 위험이 있어 semantic에 위임
            if best_score >= 2:
                LOG.debug(
                    "qna_topic_mapper: ILIKE matched topic=%r (score=%d, tokens=%r)",
                    best_row.get("topic"), best_score, tokens,
                )
                return best_row

    # ILIKE 미매칭 또는 score < 2 → semantic fallback (OpenAI API 포함이라 별도 타임아웃)
    return _run_async(_resolve_topic_semantic_async(question), _SEM_TIMEOUT)


def topic_to_search_tokens(topic: str) -> list[str]:
    """'근무-휴가', '구매/재고관리-전산장비' 같은 topic 문자열에서 검색용 토큰을 분리한다."""
    parts = re.split(r"[-/\s]+", topic)
    return [p.strip() for p in parts if len(p.strip()) >= 2]


def topic_to_task_parts(topic: str) -> tuple[str, str | None]:
    """topic 텍스트를 mobigen_task_owner의 (task_name, task_detail)로 파싱한다.

    '교육-법정의무교육' → ('교육', '법정의무교육')
    '사무환경'          → ('사무환경', None)
    '사원증/발급/등록'   → ('사원증/발급/등록', None)
    """
    if "-" in topic:
        task_name, task_detail = topic.split("-", 1)
        return task_name.strip(), task_detail.strip() or None
    return topic.strip(), None


def set_matched_topic(pre_retrieval: dict[str, Any]) -> dict[str, Any]:
    """topic 매핑 결과를 matched_topic 키로만 반영한다. keyword_query·semantic_query는 변경하지 않는다.

    db_api 라우트에서 사용한다. 로깅·추적 목적으로 topic을 기록하되 쿼리 보강은 하지 않는다.
    """
    question = str(pre_retrieval.get("rewritten_query") or "")
    topic_row = resolve_topic(question)
    topic = (topic_row.get("topic") or "").strip() if topic_row else None
    return {**pre_retrieval, "matched_topic": topic or None}


def enrich_pre_retrieval_with_topic(pre_retrieval: dict[str, Any]) -> dict[str, Any]:
    """pre_retrieval dict에 topic 매칭 결과를 반영해 keyword_query·semantic_query를 보강한다.

    변경 없이 원본을 반환하거나 enriched shallow-copy를 반환한다.
    matched_topic 키를 추가한다(None 포함). rag·hybrid 라우트에서 사용한다.
    """
    question = str(pre_retrieval.get("rewritten_query") or "")
    topic_row = resolve_topic(question)
    if not topic_row:
        return {**pre_retrieval, "matched_topic": None}

    topic = (topic_row.get("topic") or "").strip()

    enriched = dict(pre_retrieval)
    enriched["matched_topic"] = topic or None

    if topic:
        kq = str(enriched.get("keyword_query") or "")
        if topic not in kq:
            enriched["keyword_query"] = f"{kq} {topic}".strip()
        sq = str(enriched.get("semantic_query") or "")
        if topic not in sq:
            enriched["semantic_query"] = f"{sq} {topic}".strip()

    return enriched
