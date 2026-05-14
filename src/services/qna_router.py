# 규칙 기반(키워드 힌트) 라우팅 로직을 제공한다.
# LLM 라우팅 실패 시 fallback으로 classify_route()가 호출되며,
# LLM 라우팅 프롬프트 생성(build_llm_routing_prompt)도 여기서 담당한다.
from __future__ import annotations

import os

from services.qna_few_shot import routing_few_shot_block
from services.qna_structured_db import is_regulation_task_owner_question
from services.qna_types import RouteType

DOC_HINT_TOKENS = [
    "절차",
    "규정",
    "가이드",
    "문서",
    "조항",
    "사내",
    "개인 차량",
    "매뉴얼",
    "여비",
    "기숙사",
    "동호회",
    "동아리",
    "보안",
    "업무용 장비",
    "esg",
    "윤리경영",
    "환경관리",
    "사회관리",
    "목표관리",
    "조직개편",
    "공지문",
    "법인카드",
    "출장비",
    "정산",
    "지침",
    "개정안",
    "비교표",
]
STRUCTURED_HINT_TOKENS = [
    "관리",
    "담당",
    "부서",
    "사원",
    "몇 명",
    "몇명",
    "명단",
    "팀",
    "팀장",
    "그룹장",
    "실장",
    "담당자",
    "연락처",
    "이메일",
    "담당 부서",
    "현재",
    "핸드폰",
    "핸드폰 번호",
    "휴대폰",
    "휴대폰 번호",
    "전화번호",
    "사번",
    "직책",
    "생일",
]


def extract_route_hints(question: str) -> dict[str, object]:
    q = question.lower()
    matched_doc_tokens = [token for token in DOC_HINT_TOKENS if token in q]
    matched_structured_tokens = [token for token in STRUCTURED_HINT_TOKENS if token in q]
    return {
        "has_doc_hint": bool(matched_doc_tokens),
        "has_structured_hint": bool(matched_structured_tokens),
        "matched_doc_tokens": matched_doc_tokens,
        "matched_structured_tokens": matched_structured_tokens,
    }


def _few_shot_routing_enabled() -> bool:
    v = (os.getenv("QNA_FEW_SHOT_ROUTING") or "1").strip().lower()
    return v not in ("0", "false", "off", "no")


def build_llm_routing_prompt(question: str) -> str:
    hints = extract_route_hints(question)
    few_shot = routing_few_shot_block() if _few_shot_routing_enabled() else ""
    few_shot_section = (
        f"\n\n아래는 출력 형식(route_type JSON)을 보여주는 few-shot 예시입니다:\n{few_shot}\n"
        if few_shot.strip()
        else ""
    )
    return f"""
당신은 QnA 라우팅 분류기입니다.
사용자 질문을 아래 route_type 중 하나로 분류하세요.

- rag: 비정형 문서/규정/절차/가이드 기반 조회가 필요한 질문
- db_api: 담당자/부서/연락처/이메일 등 정형 DB 조회가 필요한 질문
- hybrid: 비정형 + 정형 조회가 모두 필요한 질문
- unsupported: 질문이 너무 짧거나 의미가 불분명해 분류가 어려운 질문

중요 규칙:
1) 반드시 JSON 1개만 출력합니다. (마크다운 금지)
2) 형식은 {{"route_type":"rag|db_api|hybrid|unsupported","reason":"짧은 근거"}} 로 고정합니다.
3) 질문의 핵심 의도가 문서 규정/절차 확인이면 rag를 우선합니다.
4) 질문의 핵심 의도가 담당자/부서/연락처/이메일 조회면 db_api를 우선합니다.
5) 두 의도가 동시에 존재하면 hybrid를 선택합니다.

규칙 기반 힌트(참고용):
- 비정형 힌트 단어: {DOC_HINT_TOKENS}
- 정형 힌트 단어: {STRUCTURED_HINT_TOKENS}
- 질문에서 감지된 비정형 힌트: {hints["matched_doc_tokens"]}
- 질문에서 감지된 정형 힌트: {hints["matched_structured_tokens"]}
{few_shot_section}
사용자 질문:
\"\"\"{question}\"\"\"
""".strip()


def classify_route(question: str) -> RouteType:
    if is_regulation_task_owner_question(question):
        return "db_api"
    hints = extract_route_hints(question)
    has_doc_hint = bool(hints["has_doc_hint"])
    has_structured_hint = bool(hints["has_structured_hint"])

    if has_doc_hint and has_structured_hint:
        return "hybrid"
    if has_doc_hint:
        return "rag"
    if has_structured_hint:
        return "db_api"
    if len(question.strip()) < 2:
        return "unsupported"
    return "rag"
