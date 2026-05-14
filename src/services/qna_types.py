# 프로젝트 전역 타입(RouteType, StatusType, FailureReason)과 RAG 검색 상수를 정의한다.
# RetrievalChunk(단일 청크)·RetrievalResult(검색 전체 결과) 데이터클래스도 여기서 선언한다.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RouteType = Literal["rag", "db_api", "hybrid", "unsupported"]
StatusType = Literal["success", "fallback", "error"]
FailureReason = Literal[
    "routing_failed",
    "no_result",
    "permission_denied",
    "llm_timeout",
    "llm_error",
    "unknown",
    "structured_person_not_found",
    "structured_clarification",
    "off_topic",
]

SIMILARITY_THRESHOLD = 0.4
MIN_CONTEXT_SCORE = 0.5
MAX_SUBQUERIES = 3
RRF_K = 60
CHANNEL_K_FTS = 30
CHANNEL_K_VEC = 30
FINAL_TOP_K = 10
GLOBAL_THRESHOLD = 1.0
MIN_CANDIDATES = 4
THRESHOLD_RELAX_STEP = 0.03
THRESHOLD_RELAX_MAX = 2
RERANK_TOP_N = 6
MAX_CONTEXT_CHUNKS = 4
NEAR_DUP_SIMILARITY = 0.92
MAX_CHUNKS_PER_DOC = 3
MAX_CONTEXT_TOKENS = 1800
DOC_TOKEN_CAP = 500


@dataclass
class RetrievalChunk:
    chunk_id: str
    document_id: str
    title: str
    content: str
    page: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    normalized_query: str
    keyword_query: str
    semantic_query: str
    route_type: RouteType
    chunks: list[RetrievalChunk]
    has_context: bool
    failure_reason: FailureReason | None = None
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    document_candidates: list[str] = field(default_factory=list)
    chunk_candidates: list[str] = field(default_factory=list)
    final_context_chunk_ids: list[str] = field(default_factory=list)
    rerank_applied: bool = False
    dedup_applied: bool = False
    guardrail_relaxations: int = 0
