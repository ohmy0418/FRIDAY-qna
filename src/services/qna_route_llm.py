# LLM으로 라우트를 1차 분류하고 규칙 기반 guardrail로 보정해 최종 RouteType을 확정한다.
# resolve_route_type()이 외부 진입점이며, LLM 실패 시 qna_router.classify_route()로 fallback한다.
"""LLM-first route classification with rule-based hints as guardrail (see specs/reference/plan.md Phase 3)."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from services.qna_config import QnaRuntimePolicy
from services.qna_llm import _build_llm_client
from services.qna_preprocess import decompose_subqueries, normalize_question
from services.qna_router import build_llm_routing_prompt, classify_route, extract_route_hints
from services.qna_structured_db import is_regulation_task_owner_question
from services.qna_types import RouteType
from services.qna_utils import _strip_json_fences
from utils.logger import LOG

_VALID: set[RouteType] = {"rag", "db_api", "hybrid", "unsupported"}

# 복합 질문(규정 본문 + 담당) 판별 — 단일 '휴가 규정 담당자'는 hybrid로 보내지 않는다.
_COMPOUND_CONN_RE = re.compile(
    r"(?:,\s*|그리고|및\s|\s및\s|/\s*|알려주고|말해주고|요약해주고|요약해서|\s과\s|\s와\s|"
    r"[가-힣]\s*과\s+담당|[가-힣]\s*와\s+담당)",
)
_DOC_ACTION_FOR_HYBRID_RE = re.compile(
    r"(?:절차|요약|기준\s*금액|기준금액|신청|지원\s*기준|내용|조항|확인\s*해)",
)


def _is_compound_regulation_and_assignee_question(question: str) -> bool:
    """문서 조회와 담당 조회가 둘로 나뉜 경우에만 hybrid (단일 담당 조회는 db_api 유지)."""
    qn = normalize_question(question)
    if len(decompose_subqueries(qn)) > 1:
        return True
    if _COMPOUND_CONN_RE.search(qn):
        return True
    if _DOC_ACTION_FOR_HYBRID_RE.search(qn) and re.search(
        r"(?:담당자|담당\s*부서|담당부서|누구)",
        qn,
    ):
        return True
    return False


def _parse_route_json(content: str) -> RouteType | None:
    try:
        data = json.loads(_strip_json_fences(content))
    except (json.JSONDecodeError, TypeError):
        return None
    rt = data.get("route_type")
    if rt in _VALID:
        return rt
    return None


def _invoke_route_llm(question: str, policy: QnaRuntimePolicy) -> RouteType | None:
    llm = _build_llm_client(policy, temperature=0, disable_streaming=True)
    if llm is None:
        return None
    prompt = build_llm_routing_prompt(question)
    msg = llm.invoke([HumanMessage(content=prompt)])
    raw = msg.content
    if isinstance(raw, list):
        return None
    return _parse_route_json(str(raw))


def apply_route_guardrail(question: str, llm_route: RouteType | None) -> RouteType:
    """Rule hints override obvious LLM mistakes; too-short questions stay unsupported."""
    q = question.lower().strip()
    if len(q) < 2:
        return "unsupported"

    hints = extract_route_hints(question)
    has_doc = bool(hints["has_doc_hint"])
    has_struct = bool(hints["has_structured_hint"])
    # 사용자가 조건을 직접 명시("내 직책이 XXX면", "직책이 XXX이면" 등)하면 DB 조회 불필요 → LLM 판단 존중
    _explicit_condition = bool(re.search(r"(?:내|제)\s*직책이|직책이\s*\S+이?면", question))

    # 규정 담당(mobigen) + 문서 힌트가 같아도, 단일 질의(예: 휴가 규정 담당자)는 db_api → 답변 LLM이 RAG만 보는 문제 방지
    if is_regulation_task_owner_question(question):
        if (
            has_doc
            and has_struct
            and not _explicit_condition
            and _is_compound_regulation_and_assignee_question(question)
        ):
            return "hybrid"
        return "db_api"

    if has_doc and has_struct and not _explicit_condition and llm_route not in ("hybrid", None):
        if _is_compound_regulation_and_assignee_question(question):
            return "hybrid"

    # 정형 힌트만 있는 질문(인명+이메일/담당 등)인데 LLM이 rag를 주면 DB 조회가 생략됨 → db_api로 보정
    if has_struct and not has_doc and llm_route not in ("db_api", "hybrid"):
        return "db_api"

    if llm_route in _VALID:
        return llm_route

    return classify_route(question)


def resolve_route_type(question: str, policy: QnaRuntimePolicy) -> RouteType:
    try:
        llm_route = _invoke_route_llm(question, policy)
    except Exception as e:
        LOG.warning(
            "qna_route_llm: route llm failed (q_preview=%r): %s",
            question[:80],
            e,
            exc_info=True,
        )
        llm_route = None
    return apply_route_guardrail(question, llm_route)
