# RAG 검색 전담 모듈.
# FTS(ILIKE)와 벡터 검색 결과를 RRF로 융합하고, 중복 제거·토큰 압축을 거쳐 최종 컨텍스트 청크를 반환한다.
# db_api/hybrid 라우트에서는 qna_structured_db.fetch_user_lookup_chunks()도 함께 호출한다.
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from db.connect_rdb import AsyncPostgresDB
from utils.logger import LOG
from services.qna_structured_db import fetch_user_lookup_chunks
from services.qna_utils import _generate_embedding, run_async
from services.qna_types import (
    CHANNEL_K_FTS,
    CHANNEL_K_VEC,
    DOC_TOKEN_CAP,
    FINAL_TOP_K,
    GLOBAL_THRESHOLD,
    MAX_CHUNKS_PER_DOC,
    MAX_CONTEXT_CHUNKS,
    MAX_CONTEXT_TOKENS,
    MIN_CANDIDATES,
    NEAR_DUP_SIMILARITY,
    MIN_CONTEXT_SCORE,
    RERANK_TOP_N,
    RRF_K,
    SIMILARITY_THRESHOLD,
    THRESHOLD_RELAX_MAX,
    THRESHOLD_RELAX_STEP,
    RetrievalChunk,
    RetrievalResult,
    RouteType,
)

# Top retrieved chunk score must be >= this to treat evidence as reliable (FR-016).
RELIABILITY_TOP_SCORE_THRESHOLD = MIN_CONTEXT_SCORE

# ──────────────────────────────────────────────
# 실 DB RAG 검색 (QNA_RAG_DB=1 일 때 활성화)
# ──────────────────────────────────────────────

_RAG_DOCS = "friday.documents"
_RAG_CHUNKS = "friday.document_chunks"

_SELECT_COLS = """
    dc.chunk_id::text        AS chunk_id,
    dc.document_id::text     AS document_id,
    COALESCE(dc.heading, d.title) AS heading,
    dc.content,
    COALESCE(dc.page_start, 1) AS page,
    dc.chunk_order,
    d.title,
    d.document_type,
    d.effective_date,
    d.is_latest,
    d.source_file_path,
    d.source_url
"""


def _rag_db_enabled() -> bool:
    return os.getenv("QNA_RAG_DB", "0").strip().lower() in ("1", "true", "yes")



def _build_doc_filter(metadata_filter: dict) -> tuple[str, list]:
    """metadata_filter → (WHERE 절 문자열, 파라미터 리스트)"""
    conditions = ["d.document_status = 'completed'"]
    params: list = []

    department = (str(metadata_filter.get("department") or "")).strip()
    if department:
        # documents.owner_department 컬럼 제거됨 — 제목·경로에서만 힌트 매칭
        params.append(f"%{department}%")
        n = len(params)
        conditions.append(
            f"(d.title ILIKE ${n} OR COALESCE(d.source_file_path, '') ILIKE ${n})"
        )

    time_scope = metadata_filter.get("time_scope")
    if time_scope in ("latest", "current"):
        conditions.append("d.is_latest = true")
    elif time_scope == "this_year":
        conditions.append("EXTRACT(YEAR FROM d.effective_date) = EXTRACT(YEAR FROM CURRENT_DATE)")
    elif time_scope == "custom_range":
        if date_from := metadata_filter.get("date_from"):
            params.append(date_from)
            conditions.append(f"d.effective_date >= ${len(params)}")
        if date_to := metadata_filter.get("date_to"):
            params.append(date_to)
            conditions.append(f"d.effective_date <= ${len(params)}")

    return " AND ".join(conditions), params


async def _fts_search_async(
    keyword_query: str,
    metadata_filter: dict,
    limit: int,
) -> list[tuple[dict, float]]:
    tokens = [t for t in keyword_query.split() if len(t) >= 2]
    if not tokens:
        return []

    doc_filter, params = _build_doc_filter(metadata_filter)
    offset = len(params)

    # 토큰별 ILIKE 파라미터 등록
    ilike_conds: list[str] = []
    for token in tokens:
        params.append(f"%{token}%")
        ilike_conds.append(f"dc.content ILIKE ${len(params)}")

    # 점수: 매칭 토큰 수 / 전체 토큰 수
    score_cases = " + ".join(
        f"(CASE WHEN dc.content ILIKE ${offset + i + 1} THEN 1.0 ELSE 0.0 END)"
        for i in range(len(tokens))
    )
    score_expr = f"({score_cases}) / {len(tokens)}.0"

    sql = f"""
        SELECT {_SELECT_COLS}, {score_expr} AS score
        FROM {_RAG_CHUNKS} dc
        JOIN {_RAG_DOCS} d ON dc.document_id = d.document_id
        WHERE {doc_filter}
          AND ({' OR '.join(ilike_conds)})
          AND dc.content IS NOT NULL
        ORDER BY score DESC, dc.chunk_order ASC
        LIMIT {limit}
    """
    db = AsyncPostgresDB()
    await db.connect()
    try:
        rows = await db.fetch(sql, *params)
        return [(dict(r), float(r["score"])) for r in rows]
    finally:
        await db.close()


async def _vec_search_async(
    embedding: list[float],
    metadata_filter: dict,
    limit: int,
) -> list[tuple[dict, float]]:
    doc_filter, params = _build_doc_filter(metadata_filter)
    # float 배열을 pgvector 리터럴로 변환 (사용자 입력이 아닌 모델 출력이므로 안전)
    vec_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"

    sql = f"""
        SELECT {_SELECT_COLS},
               (1 - (dc.embedding <=> '{vec_literal}'::vector)) AS score
        FROM {_RAG_CHUNKS} dc
        JOIN {_RAG_DOCS} d ON dc.document_id = d.document_id
        WHERE {doc_filter}
          AND dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> '{vec_literal}'::vector
        LIMIT {limit}
    """
    db = AsyncPostgresDB()
    await db.connect()
    try:
        rows = await db.fetch(sql, *params)
        return [(dict(r), float(r["score"])) for r in rows]
    finally:
        await db.close()


def _db_rows_to_stub_format(rows: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """DB 행을 _rank_channel 반환 포맷(dict, score)으로 변환."""
    result = []
    for row, score in rows:
        result.append(({
            "document_id": row["document_id"],
            "title": row.get("title", ""),
            "content": row.get("content", ""),
            "page": row.get("page", 1),
            "chunk_id": row.get("chunk_id"),
            "heading": row.get("heading"),
            "chunk_order": row.get("chunk_order", 0),
            "document_type": row.get("document_type"),
            "effective_date": str(row["effective_date"]) if row.get("effective_date") else None,
            "is_latest": row.get("is_latest"),
            "source_file_path": row.get("source_file_path"),
            "source_url": row.get("source_url"),
        }, score))
    return result


def _rag_rank_from_db(
    keyword_query: str,
    semantic_query: str,
    metadata_filter: dict,
) -> tuple[list[tuple[dict, float]], list[tuple[dict, float]]]:
    """실 DB에서 FTS + vector 후보를 가져온다. 실패 시 빈 리스트 반환."""
    timeout = float(os.getenv("QNA_RAG_DB_TIMEOUT", "10"))
    embedding = _generate_embedding(semantic_query, "qna_retriever")

    try:
        fts_rows = run_async(
            _fts_search_async(keyword_query, metadata_filter, CHANNEL_K_FTS), timeout
        ) or []
    except Exception as e:
        LOG.warning("qna_retriever: FTS search failed: %s", e)
        fts_rows = []

    vec_rows: list[tuple[dict, float]] = []
    if embedding:
        try:
            vec_rows = run_async(
                _vec_search_async(embedding, metadata_filter, CHANNEL_K_VEC), timeout
            ) or []
        except Exception as e:
            LOG.warning("qna_retriever: vector search failed: %s", e)
            vec_rows = []

    return _db_rows_to_stub_format(fts_rows), _db_rows_to_stub_format(vec_rows)


# ──────────────────────────────────────────────


def retrieval_from_db_chunks(
    normalized_query: str,
    route_type: RouteType,
    final_chunks: list[RetrievalChunk],
    metadata_filter: dict[str, object] | None = None,
) -> RetrievalResult:
    """동명이인 clarify 등 DB 청크만으로 RetrievalResult를 구성할 때 사용."""
    metadata_filter = metadata_filter or {}
    has_context = bool(
        final_chunks and final_chunks[0].score >= RELIABILITY_TOP_SCORE_THRESHOLD
    )
    return RetrievalResult(
        normalized_query=normalized_query,
        keyword_query=normalized_query,
        semantic_query=normalized_query,
        route_type=route_type,
        chunks=final_chunks,
        has_context=has_context,
        failure_reason=None if has_context else "no_result",
        metadata_filter=metadata_filter,
        document_candidates=[],
        chunk_candidates=[c.chunk_id for c in final_chunks],
        final_context_chunk_ids=[c.chunk_id for c in final_chunks],
        rerank_applied=False,
        dedup_applied=False,
        guardrail_relaxations=0,
    )


_CORPUS = [
    {
        "document_id": "doc-001",
        "title": "기숙사 신청 가이드",
        "content": "기숙사 신청 절차는 신청서 제출, 서류 검토, 배정 통보 순서로 진행됩니다.",
        "page": 3,
    },
    {
        "document_id": "doc-002",
        "title": "출장비 정산 규정",
        "content": "출장비 정산은 귀환 후 7일 이내에 증빙 제출이 필요하며 담당 부서는 피플팀입니다.",
        "page": 12,
    },
    {
        "document_id": "doc-003",
        "title": "법인카드 운영 규정",
        "content": "법인카드 사용 절차와 승인 기준, 담당 부서 연락처가 포함되어 있습니다.",
        "page": 8,
    },
    {
        "document_id": "doc-004",
        "title": "출장비 정산 가이드",
        "content": "출장비 정산 시 영수증과 일정 증빙을 함께 제출해야 하며 피플팀 검토 후 처리됩니다.",
        "page": 13,
    },
    {
        "document_id": "doc-005",
        "title": "기숙사 운영 규정",
        "content": "기숙사 운영 규정은 입사 기준, 배정 원칙, 퇴실 절차를 포함하고 최신 공지와 함께 확인해야 합니다.",
        "page": 4,
    },
]


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", text).strip().lower().split())


def _token_similarity(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return overlap / union


def _best_token_score(token: str, text_tokens: list[str], lowered_text: str) -> float:
    if token in text_tokens:
        return 1.0

    if len(token) >= 3:
        token_stem = token[:3]
        if any(word.startswith(token_stem) for word in text_tokens):
            return 0.7

    if token in lowered_text:
        return 0.5

    return 0.0


def _score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0

    text_tokens = _tokenize(text)
    lowered_text = text.lower()
    total = 0.0
    for token in q_tokens:
        total += _best_token_score(token, text_tokens, lowered_text)
    return total / len(q_tokens)


def _semantic_score(query: str, text: str) -> float:
    # Stub semantic score: same signal with slight smoothing for deterministic tests.
    return min(1.0, _score(query, text) * 0.95 + 0.03)


def _parse_object_uri(path: str) -> tuple[str, str] | None:
    parsed = urlparse(path)
    if parsed.scheme not in {"s3", "minio"}:
        return None
    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        return None
    return bucket, key


def _build_minio_endpoint() -> str | None:
    host = os.getenv("C_MINIO_CLIENT_HOST", "").strip()
    if not host:
        return None
    port = os.getenv("C_MINIO_CLIENT_PORT", "").strip()
    if host.startswith("http://") or host.startswith("https://"):
        endpoint = host
    else:
        endpoint = f"http://{host}"
    if port:
        base = endpoint.split("://", 1)[1]
        if ":" not in base:
            endpoint = f"{endpoint}:{port}"
    return endpoint


def _read_minio_object(bucket: str, key: str) -> str | None:
    try:
        import boto3
    except ModuleNotFoundError:
        return None

    client_kwargs: dict = {}
    endpoint = _build_minio_endpoint()
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint

    access_key = os.getenv("C_MINIO_ACCESS_KEY", "").strip()
    secret_key = os.getenv("C_MINIO_SECRET_KEY", "").strip()
    if access_key:
        client_kwargs["aws_access_key_id"] = access_key
    if secret_key:
        client_kwargs["aws_secret_access_key"] = secret_key

    try:
        s3 = boto3.client("s3", **client_kwargs)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8")
    except Exception:
        return None


def _load_runtime_corpus() -> list[dict]:
    path = os.getenv("QNA_CORPUS_PATH", "").strip()
    if not path:
        return _CORPUS

    object_ref = _parse_object_uri(path)
    if object_ref:
        text = _read_minio_object(*object_ref)
        if text is None:
            return _CORPUS
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _CORPUS
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return _CORPUS

    candidate = Path(path)
    if not candidate.exists():
        return _CORPUS
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _CORPUS
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return _CORPUS


def _rank_channel(query: str, corpus: list[dict], channel: str, limit: int) -> list[tuple[dict, float]]:
    scored: list[tuple[dict, float]] = []
    for item in corpus:
        score = _score(query, item["content"]) if channel == "fts" else _semantic_score(query, item["content"])
        if score >= SIMILARITY_THRESHOLD:
            scored.append((item, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def _apply_metadata_filter(rows: list[tuple[dict, float]], metadata_filter: dict[str, object]) -> list[tuple[dict, float]]:
    department = str(metadata_filter.get("department", "")).strip().lower()
    if not department:
        return rows
    return [(item, score) for item, score in rows if department in item["content"].lower()]


def _chunk_key(row: dict) -> str:
    """RRF 집계 키: chunk_id가 있으면 사용, 없으면(stub corpus) document_id 사용."""
    return row.get("chunk_id") or row["document_id"]


def _fuse_rrf(
    fts_rows: list[tuple[dict, float]],
    vec_rows: list[tuple[dict, float]],
) -> list[tuple[dict, float, int | None, int | None, float, float]]:
    # chunk_id 단위로 집계 — document_id로 집계하면 같은 문서의 여러 chunk가 덮어써짐
    fts_rank = {_chunk_key(row): idx + 1 for idx, (row, _) in enumerate(fts_rows)}
    vec_rank = {_chunk_key(row): idx + 1 for idx, (row, _) in enumerate(vec_rows)}
    fts_score = {_chunk_key(row): score for row, score in fts_rows}
    vec_score = {_chunk_key(row): score for row, score in vec_rows}
    chunks = {_chunk_key(row): row for row, _ in fts_rows + vec_rows}

    fused: list[tuple[dict, float, int | None, int | None, float, float]] = []
    for cid, row in chunks.items():
        rrf = 0.0
        fr = fts_rank.get(cid)
        vr = vec_rank.get(cid)
        if fr is not None:
            rrf += RRF_K / (RRF_K + fr)
        if vr is not None:
            rrf += RRF_K / (RRF_K + vr)
        fused.append((row, rrf, fr, vr, fts_score.get(cid, 0.0), vec_score.get(cid, 0.0)))

    # tie-break: vector rank -> FTS rank -> page -> chunk_id
    fused.sort(
        key=lambda x: (
            -x[1],
            x[3] if x[3] is not None else 9999,
            x[2] if x[2] is not None else 9999,
            -int(x[0].get("page", 0)),
            _chunk_key(x[0]),
        )
    )
    return fused


def _dedup_chunks(chunks: list[RetrievalChunk]) -> tuple[list[RetrievalChunk], bool]:
    changed = False
    by_doc: dict[str, list[RetrievalChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk)
    deduped: list[RetrievalChunk] = []
    for items in by_doc.values():
        items.sort(key=lambda c: c.score, reverse=True)
        deduped.extend(items[:MAX_CHUNKS_PER_DOC])
        changed = changed or len(items) > MAX_CHUNKS_PER_DOC

    hash_seen: set[str] = set()
    exact_filtered: list[RetrievalChunk] = []
    for chunk in deduped:
        h = _normalize_text(chunk.content)
        if h in hash_seen:
            changed = True
            continue
        hash_seen.add(h)
        exact_filtered.append(chunk)

    near_filtered: list[RetrievalChunk] = []
    for chunk in exact_filtered:
        if any(_token_similarity(chunk.content, seen.content) >= NEAR_DUP_SIMILARITY for seen in near_filtered):
            changed = True
            continue
        near_filtered.append(chunk)
    return near_filtered, changed


def _compress_context(chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    compressed: list[RetrievalChunk] = []
    total_tokens = 0
    per_doc_tokens: dict[str, int] = {}
    for chunk in chunks:
        token_count = len(chunk.content.split())
        if total_tokens + token_count > MAX_CONTEXT_TOKENS:
            continue
        doc_tokens = per_doc_tokens.get(chunk.document_id, 0)
        if doc_tokens + token_count > DOC_TOKEN_CAP:
            continue
        compressed.append(chunk)
        total_tokens += token_count
        per_doc_tokens[chunk.document_id] = doc_tokens + token_count
        if len(compressed) >= MAX_CONTEXT_CHUNKS:
            break
    return compressed


def retrieve(
    normalized_query: str,
    route_type: RouteType,
    metadata_filter: dict[str, object] | None = None,
    keyword_query: str | None = None,
    semantic_query: str | None = None,
) -> RetrievalResult:
    metadata_filter = metadata_filter or {}
    fts_query = keyword_query or normalized_query
    vec_query = semantic_query or normalized_query

    db_chunks: list[RetrievalChunk] = []
    if route_type in ("db_api", "hybrid"):
        db_chunks = fetch_user_lookup_chunks(normalized_query)

    # db_api에서 DB 조회 성공 시 RAG 검색 전체 스킵 (FTS + vector 불필요)
    skip_rag = route_type == "db_api" and bool(db_chunks)

    if not skip_rag:
        if _rag_db_enabled():
            fts_rows, vec_rows = _rag_rank_from_db(fts_query, vec_query, metadata_filter)
        else:
            corpus = _load_runtime_corpus()
            fts_rows = _apply_metadata_filter(_rank_channel(fts_query, corpus, "fts", CHANNEL_K_FTS), metadata_filter)
            vec_rows = _apply_metadata_filter(_rank_channel(vec_query, corpus, "vec", CHANNEL_K_VEC), metadata_filter)
    else:
        fts_rows, vec_rows = [], []

    fused = _fuse_rrf(fts_rows, vec_rows)

    threshold = GLOBAL_THRESHOLD
    relaxations = 0
    candidates = [row for row in fused if row[1] >= threshold]
    min_required = min(MIN_CANDIDATES, max(1, len(fused)))
    while len(candidates) < min_required and relaxations < THRESHOLD_RELAX_MAX:
        relaxations += 1
        threshold = max(0.0, threshold - THRESHOLD_RELAX_STEP)
        candidates = [row for row in fused if row[1] >= threshold]

    selected = candidates[:FINAL_TOP_K]
    chunks: list[RetrievalChunk] = []
    for idx, (item, fused_score, fr, vr, fs, vs) in enumerate(selected):
        chunks.append(
            RetrievalChunk(
                chunk_id=item.get("chunk_id") or f"chunk-{idx}",
                document_id=item["document_id"],
                title=item["title"],
                content=item["content"],
                page=item["page"],
                score=fused_score,
                metadata={
                    "fts_rank": fr,
                    "vector_rank": vr,
                    "fts_score": fs,
                    "vector_score": vs,
                    "source_url": item.get("source_url"),
                    "source_file_path": item.get("source_file_path"),
                },
            )
        )

    chunks, dedup_applied = _dedup_chunks(chunks)
    chunks.sort(key=lambda c: c.score, reverse=True)
    reranked = chunks[:RERANK_TOP_N]
    final_context = _compress_context(reranked)

    if route_type == "hybrid" and db_chunks:
        final_context = _compress_context(db_chunks + list(final_context))
    if route_type == "db_api" and db_chunks:
        # db_api는 DB 조회 결과만 사용. db_chunks가 없으면 RAG 결과를 fallback으로 유지.
        final_context = _compress_context(db_chunks)

    has_context = bool(final_context and final_context[0].score >= RELIABILITY_TOP_SCORE_THRESHOLD)
    failure_reason = None if has_context else "no_result"

    return RetrievalResult(
        normalized_query=normalized_query,
        keyword_query=fts_query,
        semantic_query=vec_query,
        route_type=route_type,
        chunks=final_context,
        has_context=has_context,
        failure_reason=failure_reason,
        metadata_filter=metadata_filter,
        document_candidates=[row[0]["document_id"] for row in selected],
        chunk_candidates=[chunk.chunk_id for chunk in chunks],
        final_context_chunk_ids=[chunk.chunk_id for chunk in final_context],
        rerank_applied=True,
        dedup_applied=dedup_applied,
        guardrail_relaxations=relaxations,
    )
