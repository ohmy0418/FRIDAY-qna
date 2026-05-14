# QnA 전용 내부 LangGraph 서브그래프 오케스트레이터.
# 상위 graph(graph.py)에서는 ``qna_prep`` 다음에 이 컴파일 그래프를 노드 ``qna``로 붙인다.
# 흐름: 대화 맥락 → off-topic → 라우팅 → 검색 → LLM 답변·로깅을 ``_qna_node`` 한 노드에서 처리한다.
from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from services.qna_answer import (
    build_fallback_answer,
    format_sources,
    should_include_sources,
)
from services.graphio_stream_input import GRAPHIO_STREAM_TOKENS_ACTIVE_TOOL_KEY
from services.qna_config import load_runtime_policy
from services.qna_llm import _build_llm_client, generate_grounded_answer
from services.qna_logging import log_query, log_retrieval_eval
from services.qna_preprocess import (
    _EXPLICIT_NAMED_PERSON_PRO,
    apply_rag_query_focus_for_compound_questions,
    build_pre_retrieval,
    contextualize_follow_up_with_prior,
)
from services.qna_reference_files import (
    answer_supports_regulation_references,
    merge_reference_file_into_response,
)
from services.qna_retriever import retrieve, retrieval_from_db_chunks
from services.qna_route_llm import resolve_route_type
from services.qna_status import determine_status, failure_reason_to_error_code, resolve_llm_failure
from services.qna_topic_mapper import enrich_pre_retrieval_with_topic, set_matched_topic
from services.qna_types import FailureReason, RouteType, StatusType
from utils.logger import LOG

# ── Off-topic 감지 ────────────────────────────────────────────────────────
_OFF_TOPIC_ANSWER = (
    "안녕하세요! \n\n저는 사내 규정, 담당자와 담당 부서 관련 질문에 답변하는 챗봇입니다.\n\n"
    "이와 관련해서 질문을 해주세요."
)

# DB 조회가 명확화 안내 청크를 반환했을 때 LLM을 거치지 않고 content를 그대로 응답한다.
_CLARIFICATION_STRUCTURED_KINDS = frozenset({
    "person_ambiguous",
    "person_many_name_candidates",
    "clarify_departments",
})
_OFF_TOPIC_CLASSIFY_PROMPT = """\
당신은 사내 업무 챗봇의 질문 분류기입니다.
아래 질문이 회사 업무(사내 규정·담당자·부서·인사정보·출장·휴가·장비·법인카드 등)와 관련된 질문이면 'YES', \
업무와 전혀 무관한 질문(날씨·인사말·기분·음식·스포츠·개인 감정·잡담·일반 상식 등)이면 'NO'만 답하세요.
반드시 'YES' 또는 'NO' 딱 한 단어만 출력하세요. 설명, 이유, 다른 말은 절대 쓰지 마세요.

질문: "{question}"
답:"""


async def _is_off_topic_llm(question: str, policy: Any) -> bool:
    """LLM으로 업무 관련 여부를 분류. 업무 무관이면 True를 반환."""
    llm = _build_llm_client(policy, temperature=0, max_tokens=10, disable_streaming=True)
    if llm is None:
        return False
    prompt = _OFF_TOPIC_CLASSIFY_PROMPT.format(question=question.strip())
    try:
        msg = await llm.ainvoke([HumanMessage(content=prompt)])
        answer = (msg.content if isinstance(msg.content, str) else str(msg.content)).strip().lower()
        LOG.debug("qna_graph: off-topic raw answer=%r for question=%r", answer, question[:40])
        # YES/yes/네/예 계열 → 업무 관련 → off-topic 아님
        _ON_TOPIC = ("yes", "네", "예")
        if any(answer.startswith(p) for p in _ON_TOPIC):
            return False
        # NO/no/아니 계열 → 업무 무관 → off-topic
        _OFF_TOPIC = ("no", "아니")
        if any(answer.startswith(p) for p in _OFF_TOPIC):
            return True
        # 판별 불가(빈 응답, 이상한 출력 등) → 안전하게 off-topic 아님으로 처리
        LOG.warning("qna_graph: off-topic answer unrecognized=%r, treating as on-topic", answer)
        return False
    except Exception as e:
        LOG.warning("qna_graph: off-topic classifier failed: %s", e, exc_info=True)
        return False


class QnAState(TypedDict):
    # messages에 add_messages를 쓰지 않는다(LastValue). 서브그래프 종료 시 output에는
    # 이번 턴 ``_qna_node``가 반환한 메시지 목록만 남아 상위 astream_events와 맞물린다.
    messages: list[BaseMessage]
    last_response: dict
    user_id: str
    clarify_emp_nm: str | None
    clarify_dept_hint: str | None


def _contextualize_question(question: str, history: list[BaseMessage]) -> str:
    """후속 질문이면 이전 사용자 발화 주제를 붙여 독립 질의로 만든다."""
    last_user = next(
        (str(m.content).strip() for m in reversed(history) if isinstance(m, HumanMessage) and str(m.content).strip()),
        None,
    )
    merged = contextualize_follow_up_with_prior(question, last_user)

    # 직전 발화 기준으로 병합은 됐지만 이름이 없는 경우 → 히스토리를 거슬러 이름 있는 발화 찾기
    # (3턴 이상에서 Turn 2가 "핸드폰 번호도 알려줘"처럼 이름이 없을 때 Turn 1 이름 복원)
    if merged.strip() != question.strip() and not _EXPLICIT_NAMED_PERSON_PRO.search(merged):
        for msg in reversed(history):
            if not isinstance(msg, HumanMessage):
                continue
            content = str(msg.content).strip()
            if not content or content == last_user:
                continue
            if _EXPLICIT_NAMED_PERSON_PRO.search(content):
                candidate = contextualize_follow_up_with_prior(question, content)
                if candidate.strip() != question.strip():
                    return candidate

    return merged


def _conversation_audit_fields(history: list[BaseMessage], pre_retrieval: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_history_len": len(history),
        "conversation_expanded": bool(pre_retrieval.get("conversation_expanded")),
    }


def _include_regulation_citations(status: StatusType, answer: str) -> bool:
    """성공 응답이어도 답변이 '문서에 없음'이면 출처·참고 PDF를 붙이지 않는다."""
    return should_include_sources(status) and answer_supports_regulation_references(answer)


def _retry_route_plan(initial_route: RouteType) -> list[RouteType]:
    if initial_route == "db_api":
        return ["rag"]
    if initial_route == "hybrid":
        return ["rag", "db_api"]
    if initial_route == "rag":
        return ["db_api"]
    return []


def _build_response_dict(
    request_id, route_type, selected_route_type, answer, sources,
    status, sections, failure_reason, failure_error_code,
    pre_retrieval, policy, effective_max_retries, retry_routes, attempted_routes,
) -> dict:
    return {
        "request_id": request_id,
        "route_type": route_type,
        "retrieval_route_type": selected_route_type,
        "answer": answer,
        "sources": sources,
        "status": status,
        "sections": sections,
        "time_basis": "request_time",
        "failure_reason": failure_reason,
        "failure_error_code": failure_error_code,
        "original_query": pre_retrieval["original_query"],
        "conversation_expanded": bool(pre_retrieval.get("conversation_expanded")),
        "rewrite_applied": pre_retrieval["rewrite_applied"],
        "rewritten_query": pre_retrieval["rewritten_query"],
        "execution_policy": {
            "llm_provider_mode": policy.llm_provider_mode,
            "llm_timeout_seconds": policy.llm_timeout_seconds,
            "llm_max_retries": effective_max_retries,
            "timeout_action": policy.timeout_action,
            "route_retry_enabled": policy.route_retry_enabled,
            "retry_route_plan": retry_routes,
            "attempted_routes": attempted_routes,
        },
    }


_MAX_QNA_HISTORY_TURNS = 3

# main ``generator.stream_generator``는 ``stream_tokens=True``일 때 최종 AIMessage를 SSE로 보내지 않는다.
# ``active_tool``에 ``stream_tokens`` 반영은 ``graphio_stream_input``가 ``parse_input``에 패치한다.
# ``loading_message`` 커스텀 이벤트로 본문을 올려 동일 generator에서 노출한다.

def _stream_tokens_sse_mode(config: RunnableConfig | None) -> bool:
    if not config:
        return False
    at = config.get("configurable", {}).get("active_tool")
    if not isinstance(at, dict):
        return False
    return bool(at.get(GRAPHIO_STREAM_TOKENS_ACTIVE_TOOL_KEY))


async def _dispatch_answer_for_main_sse(config: RunnableConfig | None, messages: list[Any]) -> None:
    """``stream_tokens=True`` + main generator일 때 사용자에게 보이도록 커스텀 이벤트 전송.

    ``generator``는 ``on_chain_end``의 일반 ``AIMessage``를 스트림 모드에서 SSE로 보내지 않는다.
    테스트 UI 등은 ``LangchainChatMessage``+``highlights``가 아니라 ``exception``→``token`` 경로만
    채팅 영역에 그리는 경우가 있어, 본문은 ``AIMessage``+``type: exception``으로 올린다.
    (``highlights``만 쓰면 SSE에는 보이나 화면이 비는 현상이 난다.)
    """
    if not _stream_tokens_sse_mode(config):
        return
    text = ""
    for m in messages:
        if isinstance(m, AIMessage):
            raw = m.content
            if isinstance(raw, str) and raw.strip():
                text = raw.strip()
                break
    if not text:
        return
    try:
        await adispatch_custom_event(
            "loading_message",
            AIMessage(content=text, additional_kwargs={"type": "exception"}),
            config=config,
        )
    except Exception as e:
        LOG.warning("qna_graph: main SSE dispatch failed: %s", e, exc_info=True)


async def _return_qna(config: RunnableConfig | None, payload: dict[str, Any]) -> dict[str, Any]:
    await _dispatch_answer_for_main_sse(config, payload.get("messages") or [])
    return payload


async def _qna_node(state: QnAState, config: RunnableConfig) -> dict[str, Any]:
    started = perf_counter()
    request_id = str(uuid4())

    user_id = (state.get("user_id") or "").strip()
    if not user_id:
        return await _return_qna(
            config,
            {
                "messages": [AIMessage(content="인증되지 않은 사용자입니다.")],
                "last_response": {},
            },
        )

    # AgentState.messages에는 loading 등 비대화 메시지가 포함되므로 Human·AI 메시지만 추린다.
    qa_msgs = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))]
    if not qa_msgs or not isinstance(qa_msgs[-1], HumanMessage):
        return await _return_qna(
            config,
            {
                "messages": [AIMessage(content="질문을 찾을 수 없습니다.")],
                "last_response": {},
            },
        )

    current_msg = qa_msgs[-1]
    question = str(current_msg.content)
    history = qa_msgs[:-1][-(_MAX_QNA_HISTORY_TURNS * 2):]

    clarify_emp_nm = state.get("clarify_emp_nm")
    clarify_dept_hint = state.get("clarify_dept_hint")

    effective_question = _contextualize_question(question, history)
    pre_retrieval = build_pre_retrieval(effective_question)
    pre_retrieval["original_query"] = question
    pre_retrieval["conversation_expanded"] = bool(history) and (
        effective_question.strip() != question.strip()
    )
    pre_retrieval["retrieval_surface_query"] = (
        effective_question if pre_retrieval["conversation_expanded"] else None
    )
    normalized = str(pre_retrieval["rewritten_query"])
    policy = load_runtime_policy()

    # ── Clarify (structured DB 직접 조회) 분기 ──────────────────────────────
    if (clarify_emp_nm or "").strip() and (clarify_dept_hint or "").strip():
        from services.qna_structured_db import fetch_person_by_name_and_dept

        emp = clarify_emp_nm.strip()
        dept = clarify_dept_hint.strip()
        db_chunks = fetch_person_by_name_and_dept(emp, dept)
        retrieval = retrieval_from_db_chunks(
            normalized, "db_api", db_chunks, pre_retrieval["metadata_filter"]
        )
        answer, sections = generate_grounded_answer(normalized, retrieval.chunks, "db_api", history)
        route_type: RouteType = "db_api"
        selected_route_type: RouteType = "db_api"
        status, failure_reason = determine_status(
            "db_api", retrieval.has_context, retrieval.failure_reason
        )
        ch0 = retrieval.chunks[0] if retrieval.chunks else None
        if ch0 and ch0.metadata.get("structured_kind") == "person_not_found":
            status, failure_reason = "fallback", "structured_person_not_found"
        cite_ok = _include_regulation_citations(status, answer)
        sources = format_sources(retrieval.chunks) if cite_ok else []
        elapsed_ms = int((perf_counter() - started) * 1000)
        failure_error_code = failure_reason_to_error_code(failure_reason)
        response = _build_response_dict(
            request_id, route_type, selected_route_type, answer, sources,
            status, sections, failure_reason, failure_error_code,
            pre_retrieval, policy, 0, [], ["db_api"],
        )
        response["structured_db_search_term"] = f"{emp} / {dept}"
        response["structured_clarify"] = {"emp_nm": emp, "dept_hint": dept}
        log_query(
            {
                "request_id": request_id,
                "user_id": user_id,
                "question": question,
                "original_query": pre_retrieval["original_query"],
                "rewritten_query": pre_retrieval["rewritten_query"],
                "rewrite_applied": pre_retrieval["rewrite_applied"],
                "route_type": route_type,
                "retrieval_route_type": selected_route_type,
                "status": status,
                "latency_ms": elapsed_ms,
                "failure_reason": failure_reason,
                "failure_error_code": failure_error_code,
                "llm_provider_mode": policy.llm_provider_mode,
                "llm_timeout_seconds": policy.llm_timeout_seconds,
                "llm_max_retries": 0,
                "timeout_action": policy.timeout_action,
                "retry_route_plan": [],
                "attempted_routes": ["db_api"],
                **_conversation_audit_fields(history, pre_retrieval),
            }
        )
        log_retrieval_eval(
            {
                "request_id": request_id,
                "normalized_query": normalized,
                "keyword_query": retrieval.keyword_query,
                "semantic_query": retrieval.semantic_query,
                "original_query": pre_retrieval["original_query"],
                "rewritten_query": pre_retrieval["rewritten_query"],
                "rewrite_applied": pre_retrieval["rewrite_applied"],
                "subqueries": pre_retrieval["subqueries"],
                "metadata_filter": retrieval.metadata_filter,
                "document_candidates": retrieval.document_candidates,
                "chunk_candidates": retrieval.chunk_candidates,
                "final_context_chunk_ids": retrieval.final_context_chunk_ids,
                "rerank_applied": retrieval.rerank_applied,
                "dedup_applied": retrieval.dedup_applied,
                "guardrail_relaxations": retrieval.guardrail_relaxations,
                **_conversation_audit_fields(history, pre_retrieval),
            }
        )
        out_messages = merge_reference_file_into_response(
            response,
            retrieval.chunks,
            include_references=cite_ok,
        )
        return await _return_qna(
            config,
            {
                "messages": out_messages,
                "last_response": response,
            },
        )

    # ── Off-topic 감지 ────────────────────────────────────────────────────────
    if await _is_off_topic_llm(effective_question, policy):
        return await _return_qna(
            config,
            {
                "messages": [AIMessage(content=_OFF_TOPIC_ANSWER)],
                "last_response": {
                    "request_id": request_id,
                    "route_type": "unsupported",
                    "retrieval_route_type": "unsupported",
                    "answer": _OFF_TOPIC_ANSWER,
                    "sources": [],
                    "status": "fallback",
                    "sections": {},
                    "failure_reason": "off_topic",
                    "failure_error_code": failure_reason_to_error_code("off_topic"),
                    "original_query": question,
                    "conversation_expanded": False,
                    "rewrite_applied": False,
                    "rewritten_query": question,
                    "execution_policy": {},
                },
            },
        )

    # ── 일반 RAG / DB / hybrid 라우팅 ────────────────────────────────────────
    route_type: RouteType = await asyncio.to_thread(resolve_route_type, normalized, policy)

    if route_type in ("rag", "hybrid"):
        pre_retrieval = enrich_pre_retrieval_with_topic(pre_retrieval)
        pre_retrieval = apply_rag_query_focus_for_compound_questions(pre_retrieval, route_type)
    else:
        pre_retrieval = set_matched_topic(pre_retrieval)

    retry_routes = _retry_route_plan(route_type) if policy.route_retry_enabled else []
    effective_max_retries = len(retry_routes)
    attempt_routes = [route_type] + retry_routes
    attempted_routes: list[RouteType] = []

    selected_route_type: RouteType = route_type
    retrieval = retrieve(
        normalized, route_type,
        metadata_filter=pre_retrieval["metadata_filter"],
        keyword_query=str(pre_retrieval["keyword_query"]),
        semantic_query=str(pre_retrieval["semantic_query"]),
    )
    status, failure_reason = determine_status(route_type, retrieval.has_context, retrieval.failure_reason)
    llm_failure_reason: FailureReason | None = None
    answer = ""
    sections: dict = {}
    for idx, candidate_route in enumerate(attempt_routes):
        if idx == 0:
            candidate_retrieval = retrieval
            candidate_status, candidate_failure_reason = status, failure_reason
        else:
            candidate_retrieval = retrieve(
                normalized, candidate_route,
                metadata_filter=pre_retrieval["metadata_filter"],
                keyword_query=str(pre_retrieval["keyword_query"]),
                semantic_query=str(pre_retrieval["semantic_query"]),
            )
            candidate_status, candidate_failure_reason = determine_status(
                candidate_route, candidate_retrieval.has_context, candidate_retrieval.failure_reason,
            )

        attempted_routes.append(candidate_route)
        selected_route_type = candidate_route
        retrieval = candidate_retrieval
        status, failure_reason = candidate_status, candidate_failure_reason

        if candidate_status != "success":
            continue

        # 동명이인·부서명 불명확 등 안내 청크는 content 자체가 완성된 응답 — LLM 불필요
        _guidance_ch = candidate_retrieval.chunks[0] if candidate_retrieval.chunks else None
        if _guidance_ch and _guidance_ch.metadata.get("structured_kind") in _CLARIFICATION_STRUCTURED_KINDS:
            answer = _guidance_ch.content
            sections = {}
            status, failure_reason = "fallback", "structured_clarification"
            break

        try:
            answer, sections = generate_grounded_answer(normalized, candidate_retrieval.chunks, candidate_route, history)
            llm_failure_reason = None

            ch0 = candidate_retrieval.chunks[0] if candidate_retrieval.chunks else None
            if ch0 and ch0.metadata.get("structured_kind") == "person_not_found":
                status, failure_reason = "fallback", "structured_person_not_found"
            break
        except TimeoutError:
            llm_failure_reason = "llm_timeout"
            if idx < len(attempt_routes) - 1:
                continue
            status, failure_reason = resolve_llm_failure(policy.timeout_action, "llm_timeout")
            answer = build_fallback_answer(failure_reason)
            sections = {}
        except Exception as e:
            LOG.warning(
                "qna_graph: grounded answer failed (request_id=%s, route=%s, attempt=%d/%d): %s",
                request_id,
                candidate_route,
                idx + 1,
                len(attempt_routes),
                e,
                exc_info=True,
            )
            llm_failure_reason = "llm_error"
            if idx < len(attempt_routes) - 1:
                continue
            status, failure_reason = resolve_llm_failure(policy.timeout_action, "llm_error")
            answer = build_fallback_answer(failure_reason)
            sections = {}

    if not answer:
        if llm_failure_reason:
            status, failure_reason = resolve_llm_failure(policy.timeout_action, llm_failure_reason)
            answer = build_fallback_answer(failure_reason)
        else:
            answer = build_fallback_answer(failure_reason)
        sections = {}

    cite_ok = _include_regulation_citations(status, answer)
    sources = format_sources(retrieval.chunks) if cite_ok else []
    elapsed_ms = int((perf_counter() - started) * 1000)
    failure_error_code = failure_reason_to_error_code(failure_reason)

    response = _build_response_dict(
        request_id, route_type, selected_route_type, answer, sources,
        status, sections, failure_reason, failure_error_code,
        pre_retrieval, policy, effective_max_retries, retry_routes, attempted_routes,
    )
    out_messages = merge_reference_file_into_response(
        response,
        retrieval.chunks,
        include_references=cite_ok,
    )
    if "db_api" in attempted_routes:
        from services.qna_structured_db import (
            extract_department_headcount_phrase,
            extract_department_roster_phrase,
            extract_lookup_term,
        )
        roster = extract_department_roster_phrase(normalized)
        head_dept = extract_department_headcount_phrase(normalized)
        response["structured_db_search_term"] = head_dept or roster or extract_lookup_term(normalized)

    log_query(
        {
            "request_id": request_id,
            "user_id": user_id,
            "question": question,
            "original_query": pre_retrieval["original_query"],
            "rewritten_query": pre_retrieval["rewritten_query"],
            "rewrite_applied": pre_retrieval["rewrite_applied"],
            "route_type": route_type,
            "retrieval_route_type": selected_route_type,
            "status": status,
            "latency_ms": elapsed_ms,
            "failure_reason": failure_reason,
            "failure_error_code": failure_error_code,
            "llm_provider_mode": policy.llm_provider_mode,
            "llm_timeout_seconds": policy.llm_timeout_seconds,
            "llm_max_retries": effective_max_retries,
            "timeout_action": policy.timeout_action,
            "retry_route_plan": retry_routes,
            "attempted_routes": attempted_routes,
            **_conversation_audit_fields(history, pre_retrieval),
        }
    )
    log_retrieval_eval(
        {
            "request_id": request_id,
            "normalized_query": normalized,
            "keyword_query": retrieval.keyword_query,
            "semantic_query": retrieval.semantic_query,
            "original_query": pre_retrieval["original_query"],
            "rewritten_query": pre_retrieval["rewritten_query"],
            "rewrite_applied": pre_retrieval["rewrite_applied"],
            "subqueries": pre_retrieval["subqueries"],
            "metadata_filter": retrieval.metadata_filter,
            "document_candidates": retrieval.document_candidates,
            "chunk_candidates": retrieval.chunk_candidates,
            "final_context_chunk_ids": retrieval.final_context_chunk_ids,
            "rerank_applied": retrieval.rerank_applied,
            "dedup_applied": retrieval.dedup_applied,
            "guardrail_relaxations": retrieval.guardrail_relaxations,
            **_conversation_audit_fields(history, pre_retrieval),
        }
    )

    return await _return_qna(
        config,
        {
            "messages": out_messages,
            "last_response": response,
        },
    )


def build_qna_graph():
    graph = StateGraph(QnAState)
    graph.add_node("qna_answer", _qna_node)
    graph.set_entry_point("qna_answer")
    graph.add_edge("qna_answer", END)
    return graph.compile()


qna_graph = build_qna_graph()
