# LLM 클라이언트 팩토리와 답변 생성 로직을 제공한다.
# 청크 타입(사원 row / 문서)에 따라 프롬프트를 선택해 LLM을 호출하며, generate_grounded_answer()가 외부 진입점이다.
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from services.qna_answer import (
    _USER_ROW_PREFIX,
    _USER_SOURCE,
    build_success_answer,
    format_person_as_context,
)
from services.qna_config import QnaRuntimePolicy, load_runtime_policy
from services.qna_few_shot import answer_llm_few_shot_block
from services.qna_types import RetrievalChunk, RouteType
from utils.logger import LOG


def _build_llm_client(policy: QnaRuntimePolicy, **kwargs) -> ChatOpenAI | None:
    """공통 LLM 클라이언트 팩토리. 인증 정보가 없으면 None 반환.

    라우팅·오프토픽·정형 파서 등 비답변 호출은 ``disable_streaming=True``를 넘긴다.
    최종 답변은 ``invoke``로 수집한다(``stream``은 서브그래프에서도 부모 SSE 토큰으로
    전파되어 ``highlights`` 디스패치와 본문이 이중으로 보일 수 있음).
    """
    key = (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    if not key and not policy.llm_api_address:
        return None
    params: dict = {
        "model": policy.llm_model,
        "openai_api_key": key or "dummy",
        **kwargs,
    }
    if policy.llm_api_address:
        params["base_url"] = policy.llm_api_address
    return ChatOpenAI(**params)


def _stream_content_delta(content: Any) -> str:
    """단일 스트림 청크의 ``content``를 이어붙일 수 있는 문자열로 정규화한다."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _invoke_messages_to_text(llm: ChatOpenAI, messages: list) -> str:
    """답변 LLM을 ``invoke``로 호출해 문자열만 받는다.

    부모 그래프 ``astream_events``에 토큰이 새지 않도록 ``callbacks=[]``로 분리한다.
    (그렇지 않으면 ``invoke``여도 상위 ``on_chat_model_stream`` + ``_dispatch`` 본문 이벤트가
    겹쳐 동일 문장이 두 번 보일 수 있다.)
    """
    msg = llm.invoke(messages, config=RunnableConfig(callbacks=[]))
    return _stream_content_delta(getattr(msg, "content", None)).strip()


def _few_shot_answer_enabled() -> bool:
    v = (os.getenv("QNA_FEW_SHOT_ANSWER") or "1").strip().lower()
    return v not in ("0", "false", "off", "no")


def get_answer_prompt_few_shots() -> str:
    """Few-shot text for grounded answer LLM (style + lookup patterns)."""
    if not _few_shot_answer_enabled():
        return ""
    return answer_llm_few_shot_block()


def _build_rag_prompt(question: str, chunks: list[RetrievalChunk], history: list | None = None) -> str:
    context = "\n\n".join(
        f"[{c.title}]\n{c.content}" for c in chunks[:5]
    )
    few_shots = get_answer_prompt_few_shots()
    few_shot_block = f"참고 예시:\n{few_shots}\n\n" if few_shots else ""
    history_block = ""
    if history:
        lines = []
        for m in history[-6:]:
            if isinstance(m, HumanMessage):
                lines.append(f"사용자: {m.content}")
            elif isinstance(m, AIMessage) and m.content:
                lines.append(f"어시스턴트: {m.content}")
        if lines:
            history_block = "이전 대화:\n" + "\n".join(lines) + "\n\n"
    return (
        "당신은 사내 규정·정책 안내 어시스턴트입니다.\n"
        "아래 문서 내용만을 근거로 질문에 간결하고 정확하게 답변하세요.\n"
        "문서에 없는 내용은 추측하지 마세요.\n"
        "이전 대화는 맥락 파악 용도로만 참고하고, 이전 대화의 내용을 현재 답변에 반복하거나 포함하지 마세요.\n"
        "사용자가 직책·거리·금액·일수·연료 종류 등 구체적인 조건을 제시한 경우, "
        "문서의 해당 기준을 적용하여 계산하고 산식을 **굵게** 명시하세요.\n\n"
        f"{few_shot_block}"
        f"문서 내용:\n{context}\n\n"
        f"{history_block}"
        f"질문: {question}\n\n"
        "답변:"
    )


def _build_person_answer_prompt(
    question: str,
    person_rows: list[RetrievalChunk],
    rag_chunks: list[RetrievalChunk] | None = None,
    history: list | None = None,
) -> str:
    person_blocks: list[str] = []
    for i, chunk in enumerate(person_rows[:5], 1):
        body = format_person_as_context(chunk)
        if body:
            header = f"[사원 정보 {i}]" if len(person_rows) > 1 else "[사원 정보]"
            person_blocks.append(f"{header}\n{body}")

    person_context = "\n\n".join(person_blocks) if person_blocks else "(조회된 사원 정보 없음)"

    rag_block = ""
    if rag_chunks:
        rag_text = "\n\n".join(f"[{c.title}]\n{c.content}" for c in rag_chunks[:3])
        rag_block = f"\n\n[관련 규정·문서]\n{rag_text}"

    history_block = ""
    if history:
        lines = []
        for m in history[-6:]:
            if isinstance(m, HumanMessage):
                lines.append(f"사용자: {m.content}")
            elif isinstance(m, AIMessage) and m.content:
                lines.append(f"어시스턴트: {m.content}")
        if lines:
            history_block = "이전 대화:\n" + "\n".join(lines) + "\n\n"

    rag_rule = (
        "10. [관련 규정·문서]가 있는 경우, 사원 정보와 함께 관련 규정을 간결하게 보완하세요.\n"
        if rag_chunks else ""
    )
    return (
        "당신은 사내 인사 정보를 안내하는 어시스턴트입니다.\n"
        "아래 [사원 정보]는 사내 DB에서 조회한 실제 데이터입니다.\n\n"
        "반드시 지켜야 할 규칙:\n"
        "1. [사원 정보]에 명시된 항목과 값만 사용하세요. "
        "제공되지 않은 항목은 절대 추측하거나 임의로 생성하지 마세요.\n"
        "2. 항목 값이 비어 있거나 제공되지 않은 경우, "
        "'등록되어 있지 않습니다'라고 명확히 안내하세요.\n"
        "3. 질문에서 요청한 항목을 중심으로 답하되, "
        "요청하지 않은 항목을 임의로 추가하지 마세요.\n"
        "4. 이름·이메일·연락처·사번·부서 등 모든 항목 값을 "
        "임의로 변경하거나 재조합하지 마세요. 제공된 값을 그대로 사용하세요.\n"
        "5. 사원 정보가 여럿이면 각각 구분하여 안내하세요.\n"
        "6. 질문이 부서·이메일·휴대폰(연락처)·사번 중 **하나만** 묻는 경우, 불릿을 쓰지 말고 한 문장으로만 답합니다. "
        "값은 `**굵게**` 처리합니다(이메일은 `[주소](mailto:주소) 입니다.` 형식 허용).\n"
        "   - 부서: `{이름} 프로의 부서는 **{부서}** 입니다.`\n"
        "   - 휴대폰·연락처: 질문에 팀·부서명이 이름 앞에 있으면 그대로 붙입니다. "
        "예: `NI개발1팀 강은성 프로의 휴대폰 번호는 **010-…** 입니다.` "
        "조직명이 질문에 없으면 `{이름} 프로의 휴대폰 번호는 **…** 입니다.`\n"
        "   - 이메일: `{이름} 프로의 이메일은 [주소](mailto:주소) 입니다.`\n"
        "   - 사번: `{이름} 프로의 사번은 **{사번}** 입니다.`\n"
        "7. 프로필 전반을 묻거나 답에 서로 다른 항목이 둘 이상 포함되면, "
        "불릿 앞에 `{사원 이름} 프로 정보는 아래와 같습니다.` 한 문장과 빈 줄을 둡니다.\n"
        "8. 위 7번에 해당할 때만 마크다운 불릿(-)을 씁니다. 레이블은 굵게 하지 말고 `이름:`, `이메일:` 형태로 적습니다.\n"
        "   - 이름과 이메일은 **항상 서로 다른 불릿**으로 나눕니다: `- 이름: OOO`, `- 이메일: 주소` "
        "(프로필 목록에서는 이메일을 링크로 감싸지 말고 주소 문자열만 적습니다).\n"
        "   - 부서·연락처·사번은 값이 있는 항목만 "
        "`- 부서:`, `- 연락처:`, `- 사번:` 불릿으로 한 줄씩 이어 적습니다.\n"
        "9. 질문에서 둘 이상의 서로 다른 항목을 묻는 경우, 7~8번 규칙(안내 문장+불릿)을 따릅니다.\n"
        f"{rag_rule}\n"
        f"{person_context}"
        f"{rag_block}\n\n"
        f"{history_block}"
        f"질문: {question}\n\n"
        "답변:"
    )


def _call_answer_llm(
    question: str,
    chunks: list[RetrievalChunk],
    policy: QnaRuntimePolicy,
    history: list | None = None,
) -> tuple[str, dict]:
    if not chunks:
        return build_success_answer(question, chunks, "rag")
    llm = _build_llm_client(policy, temperature=0.1)
    if llm is None:
        return build_success_answer(question, chunks, "rag")

    prompt = _build_rag_prompt(question, chunks, history)
    messages = [HumanMessage(content=prompt)]
    try:
        answer = _invoke_messages_to_text(llm, messages)
    except Exception as e:
        LOG.warning(
            "qna_llm: rag answer llm failed (model=%s, q_preview=%r): %s",
            policy.llm_model,
            question[:80],
            e,
            exc_info=True,
        )
        raise
    if not answer:
        msg = llm.invoke(messages, config=RunnableConfig(callbacks=[]))
        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
        answer = answer.strip()

    return answer, {}


def _call_person_answer_llm(
    question: str,
    person_rows: list[RetrievalChunk],
    policy: QnaRuntimePolicy,
    rag_chunks: list[RetrievalChunk] | None = None,
    history: list | None = None,
) -> tuple[str, dict]:
    llm = _build_llm_client(policy, temperature=0)
    if llm is None:
        return build_success_answer(question, person_rows, "db_api")

    prompt = _build_person_answer_prompt(question, person_rows, rag_chunks, history)
    messages = [HumanMessage(content=prompt)]
    try:
        answer = _invoke_messages_to_text(llm, messages)
        if not answer:
            msg = llm.invoke(messages, config=RunnableConfig(callbacks=[]))
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            answer = answer.strip()
        return answer, {"person_selective": True}
    except Exception as e:
        LOG.warning(
            "qna_llm: person answer llm failed (model=%s, q_preview=%r): %s",
            policy.llm_model,
            question[:80],
            e,
            exc_info=True,
        )
        return build_success_answer(question, person_rows, "db_api")


def _has_person_rows(chunks: list[RetrievalChunk]) -> bool:
    for c in chunks:
        if c.chunk_id.startswith(_USER_ROW_PREFIX) and c.metadata.get("source") == _USER_SOURCE:
            if (c.metadata.get("emp_nm") or "").strip():
                return True
    return False


def _split_person_and_rag(
    chunks: list[RetrievalChunk],
) -> tuple[list[RetrievalChunk], list[RetrievalChunk]]:
    person: list[RetrievalChunk] = []
    rag: list[RetrievalChunk] = []
    for c in chunks:
        is_person = (
            c.chunk_id.startswith(_USER_ROW_PREFIX)
            and c.metadata.get("source") == _USER_SOURCE
            and (c.metadata.get("emp_nm") or "").strip()
        )
        if is_person:
            person.append(c)
        else:
            rag.append(c)
    return person, rag


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievalChunk],
    route_type: RouteType,
    history: list | None = None,
) -> tuple[str, dict]:
    policy = load_runtime_policy()
    if policy.force_llm_failure == "timeout":
        raise TimeoutError("Forced LLM timeout for policy validation")
    if policy.force_llm_failure == "error":
        raise RuntimeError("Forced LLM error for policy validation")

    if route_type == "db_api":
        person_rows, _ = _split_person_and_rag(chunks)
        if person_rows:
            return _call_person_answer_llm(question, person_rows, policy, history=history)
        return build_success_answer(question, chunks, route_type)

    # hybrid이지만 실제 person rows가 없으면 사용자가 조건을 직접 명시한 경우이므로 LLM으로 처리
    if route_type == "hybrid" and _has_person_rows(chunks):
        person_rows, rag_chunks = _split_person_and_rag(chunks)
        return _call_person_answer_llm(
            question, person_rows, policy, rag_chunks=rag_chunks or None, history=history
        )

    return _call_answer_llm(question, chunks, policy, history)
