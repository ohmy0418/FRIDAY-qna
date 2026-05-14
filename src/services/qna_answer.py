# 답변 포맷팅 유틸 모듈.
# 출처 목록 생성(format_sources), 사원 정보 텍스트 변환(format_person_as_context),
# LLM 불가 시 fallback 답변(build_success_answer·build_fallback_answer) 생성을 담당한다.
from __future__ import annotations

import re

from services.qna_types import RetrievalChunk, StatusType

_USER_ROW_PREFIX = "db-user-"
_USER_SOURCE = "office_guide.user"


def format_person_as_context(chunk: RetrievalChunk) -> str:
    """사원 row 청크의 metadata를 LLM 컨텍스트용 텍스트로 변환."""
    meta = chunk.metadata or {}
    lines: list[str] = []
    if meta.get("emp_nm"):     lines.append(f"이름: {meta['emp_nm']}")
    if meta.get("emp_email"):  lines.append(f"이메일: {meta['emp_email']}")
    if meta.get("dept_nm"):    lines.append(f"부서: {meta['dept_nm']}")
    if meta.get("emp_cphone"): lines.append(f"연락처: {meta['emp_cphone']}")
    if meta.get("emp_cd"):     lines.append(f"사번: {meta['emp_cd']}")
    return "\n".join(lines)


def _person_question_focuses(question: str) -> set[str]:
    """질문에서 요청하는 단일 인사 항목(부서·이메일·휴대폰·사번) 추정. 0개면 프로필 전체로 본다."""
    q = question.strip()
    if not q:
        return set()
    s: set[str] = set()
    if re.search(r"부서|소속|소속팀|담당\s*부서", q):
        s.add("dept")
    if re.search(r"이메일|e[- ]?mail|메일\s*주소|\b메일\b", q, re.I):
        s.add("email")
    if re.search(r"휴대폰|연락처|전화번호|전화\s*번호|핸드폰|폰\s*번호", q):
        s.add("phone")
    if re.search(r"사번", q):
        s.add("emp_cd")
    if re.search(r"어디\s*야|어디예요|어느\s*팀", q) and not s:
        s.add("dept")
    return s


def _leading_org_before_emp_name(question: str, emp_nm: str) -> str:
    """질문 문자열에서 사원명 앞의 조직·팀명 등 접두(공백 제거 전 원문 기준)."""
    en = (emp_nm or "").strip()
    if not en:
        return ""
    m = re.search(re.escape(en) + r"(?:\s*(?:프로|님|씨|사원|대리|과장|차장|부장))?", question)
    if not m or m.start() == 0:
        return ""
    return question[: m.start()].strip()


def _person_single_field_missing_line(field: str, emp_nm: str) -> str:
    label = {"dept": "부서", "phone": "휴대폰 번호", "email": "이메일", "emp_cd": "사번"}[field]
    return f"{emp_nm} 프로의 {label}은(는) 등록되어 있지 않습니다."


def format_person_user_markdown(meta: dict, question: str | None = None) -> str:
    """사원 메타데이터를 사용자-facing 답변으로 변환 (LLM fallback 등).

    질문이 특정 항목 하나만 묻는 경우 한 문장으로 답하고, 그 외에는 불릿+안내 문장을 쓴다.
    """
    emp_nm = (meta.get("emp_nm") or "").strip()
    emp_email = (meta.get("emp_email") or "").strip()
    dept = (meta.get("dept_nm") or "").strip()
    phone = (meta.get("emp_cphone") or "").strip()
    emp_cd = (meta.get("emp_cd") or "").strip()

    q = (question or "").strip()
    focuses = _person_question_focuses(q) if q else set()

    if len(focuses) == 1 and emp_nm:
        only = next(iter(focuses))
        if only == "dept":
            return f"{emp_nm} 프로의 부서는 **{dept}** 입니다." if dept else _person_single_field_missing_line("dept", emp_nm)
        if only == "phone":
            if not phone:
                return _person_single_field_missing_line("phone", emp_nm)
            prefix = _leading_org_before_emp_name(q, emp_nm)
            if prefix:
                return f"{prefix} {emp_nm} 프로의 휴대폰 번호는 **{phone}** 입니다."
            return f"{emp_nm} 프로의 휴대폰 번호는 **{phone}** 입니다."
        if only == "email":
            if not emp_email:
                return _person_single_field_missing_line("email", emp_nm)
            return f"{emp_nm} 프로의 이메일은 [{emp_email}](mailto:{emp_email}) 입니다."
        if only == "emp_cd":
            return f"{emp_nm} 프로의 사번은 **{emp_cd}** 입니다." if emp_cd else _person_single_field_missing_line("emp_cd", emp_nm)

    bullets: list[str] = []
    if emp_nm:
        bullets.append(f"- 이름: {emp_nm}")
    if emp_email:
        bullets.append(f"- 이메일: {emp_email}")
    if dept:
        bullets.append(f"- 부서: {dept}")
    if phone:
        bullets.append(f"- 연락처: {phone}")
    if emp_cd:
        bullets.append(f"- 사번: {emp_cd}")
    body = "\n".join(bullets)
    if emp_nm and body:
        return f"{emp_nm} 프로 정보는 아래와 같습니다.\n\n{body}"
    return body


def format_sources(chunks: list[RetrievalChunk]) -> list[dict]:
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for c in chunks:
        key = (c.document_id, c.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "document_name": c.title,
                "page": c.page,
                "section": f"p{c.page}",
                "url": c.metadata.get("source_url") or "",
                "document_id": c.document_id,
                "snippet": c.content[:100],
                "chunk_id": c.chunk_id,
            }
        )
    return sources


def build_success_answer(_question: str, chunks: list[RetrievalChunk], route_type: str) -> tuple[str, dict]:
    """LLM 사용 불가 시 fallback 답변. 정상 경로는 qna_llm.generate_grounded_answer 참조."""
    if not chunks:
        return "관련 문맥을 찾지 못했습니다.", {}

    if route_type in ("db_api", "hybrid"):
        blocks: list[str] = []
        for c in chunks[:10]:
            if (
                c.chunk_id.startswith(_USER_ROW_PREFIX)
                and c.metadata.get("source") == _USER_SOURCE
                and (c.metadata.get("emp_nm") or "").strip()
            ):
                blocks.append(format_person_user_markdown(c.metadata, _question))
            else:
                blocks.append(c.content)
        answer = "\n\n---\n\n".join(blocks) if blocks else "관련 문맥을 찾지 못했습니다."
        return answer, {}

    answer = f"{chunks[0].title} 기준으로 안내드립니다: {chunks[0].content}"
    return answer, {}


_FALLBACK_ANSWERS: dict[str, str] = {
    "no_result": (
        "현재 검색 결과만으로는 정확한 답변을 확인하기 어렵습니다.\n\n"
        "정확한 안내가 필요하시면 피플팀에 문의해 주세요."
    ),
    "routing_failed": (
        "질문을 정확히 이해하지 못했습니다.\n\n"
        "업무 내용을 조금 더 구체적으로 적어 주시거나, 정확한 안내가 필요하시면 피플팀에 문의해 주세요."
    ),
    "permission_denied": (
        "해당 내용은 안내해 드릴 수 있는 범위에 포함되지 않거나, 조회 권한이 없습니다.\n\n"
        "자세한 사항은 피플팀에 문의해 주세요."
    ),
    "llm_timeout": (
        "답변을 준비하는 데 예상보다 시간이 걸리고 있습니다.\n\n"
        "잠시 후 다시 시도해 주시고, 문제가 계속되면 피플팀에 문의해 주세요."
    ),
    "llm_error": (
        "답변을 만드는 과정에서 일시적인 오류가 발생했습니다.\n\n"
        "잠시 후 다시 시도해 주시고, 반복될 경우 피플팀에 문의해 주세요."
    ),
    "unknown": (
        "처리 중 문제가 발생했습니다.\n\n"
        "잠시 후 다시 시도해 주시고, 동일 현상이 계속되면 피플팀에 문의해 주세요."
    ),
    "structured_person_not_found": (
        "요청하신 사원 정보를 찾지 못했습니다.\n\n"
        "이름·부서 등을 확인하신 뒤 다시 질문해 주시거나, 피플팀에 문의해 주세요."
    ),
    "structured_clarification": (
        "답변드리기 위해 조금 더 구체적인 정보가 필요합니다.\n\n"
        "질문을 보완해 주시면 정확히 안내해 드릴 수 있습니다."
    ),
    "off_topic": (
        "안녕하세요! \n\n저는 사내 규정, 담당자와 담당 부서 관련 질문에 답변하는 챗봇입니다.\n\n"
        "이와 관련해서 질문을 해주세요."
    ),
}


def build_fallback_answer(failure_reason: str | None) -> str:
    key = failure_reason or "unknown"
    return _FALLBACK_ANSWERS.get(key, _FALLBACK_ANSWERS["unknown"])


def should_include_sources(status: StatusType) -> bool:
    return status == "success"
