# 질문 전처리 전담 모듈.
# 정규화·리라이팅·서브질의 분해·FTS 키워드 추출·의미 질의 생성·대화 컨텍스트 병합을 수행하고
# 모든 결과를 build_pre_retrieval()로 묶어 반환한다.
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from services.qna_structured_db import TASK_KNOWLEDGE_DOMAIN_HINTS
from services.qna_types import MAX_SUBQUERIES, RouteType

_DISCOURSE_LEAD = re.compile(r"^(?:(?:그럼|그래|근데)\s+|그\s+)")
# 사원 디렉터리 질의(이름+직함)는 이전 규정/담당 턴과 붙이지 않는다. (\b는 한글 '프로의' 경계에서 맞지 않음)
_EXPLICIT_NAMED_PERSON_PRO = re.compile(
    r"[가-힣]{2,5}\s*(?:프로|님|씨|팀장|대리|과장|차장|부장|주임|계장|이사|상무|전무|대표)"
    r"(?=\s|의|은|는|이|가|을|를|와|과|,|\.|…|!|\?|？|$)",
    re.UNICODE,
)

SYNONYM_MAP = {
    "법카": "법인카드",
    "반차": "반일연차",
    "여비": "출장비",
}

# 인원·명단 질문은 이전 사원 정보와 무관하므로 독립 질문으로 처리
_HEADCOUNT_ROSTER_HINTS: frozenset[str] = frozenset({
    "몇 명", "몇명", "명수", "명단", "리스트", "목록", "인원수",
})
# 업무 외 도메인 키워드 — 이전 업무 컨텍스트와 병합하지 않는다
_OFFSITE_DOMAIN_HINTS: frozenset[str] = frozenset({
    "날씨", "주식", "스포츠", "축구", "야구", "농구", "영화", "음악",
    "요리", "게임", "뉴스", "맛집", "여행", "쇼핑",
})
# 이전 발화에서 주제어를 잘라낼 경계 키워드 (문서 유형 단어 이전까지만 topic으로 사용)
_DOC_TYPE_BOUNDARY_TOKENS: tuple[str, ...] = (
    "규정", "지침", "가이드", "문서", "매뉴얼", "내규", "사규",
)

FILLER_WORDS = ["아", "좀", "혹시", "근데", "아니", "그", "너"]
AMBIGUOUS_HINTS = ["이거", "그거", "이건", "저건", "어떻게", "뭐야", "뭔지"]
TIME_SCOPE_HINTS = {
    "최신": "latest",
    "현재": "current",
    "올해": "this_year",
}
DOC_TYPE_HINTS = {
    "규정": "policy",
    "가이드": "guide",
    "공지": "notice",
    "신청서": "form",
}

# FTS에서 의미 없는 조사·어미·의문 종결 패턴 (종결 표현 → 중간 어미 → 조사 순으로 제거)
_KEYWORD_STOP_PATTERNS = [
    # 질문 종결/요청 표현
    r"[이가]?\s*(뭐야|뭔지|어떻게|어떤|무엇|어디|왜|언제|어디서)\??$",
    r"(?:나요|인가요|인지요|인데요|하나요|할까요|하죠|입니까|습니까|되나요|되는지|되는가요|됩니까)\??$",
    r"(?:알고\s*싶[어습]|알려줘|알려주세요|설명해줘|설명해주세요)\??$",
    r"\?+$",
    # 중간 연결 표현
    r"(?:지원한|지급한|제공한|운영한|관련한|포함한|받은|된|하는|되는)(?:\s|$)",
    r"관련\s*(?:해서|하여|된|있는|한)\s*",
    r"에\s*대(?:해\s*(?:서|서는|요)?|하(?:여|여서|여서는)?)\s*",
    # 혼합 질문 중간 연결(문장 앞·끝 $ 패턴으로는 제거되지 않음)
    r"(?:알려주고|알려주시고|말해주고|말해주시고)\s*,?\s*",
    r"(?:요약해주고|요약해서|요약해\s*주고)\s*,?\s*",
    r"(?:알려줘|알려주세요)\s*,\s*",
    # 단어 뒤 조사 제거: "규정에서" → "규정", "규정의" → "규정", "동호회의" → "동호회" 등
    # lookbehind로 앞 한글 음절은 보존하고 조사만 제거
    r"(?<=[가-힣])(에서|에게서|에게|으로부터|으로|로부터|로|의|와|과|이나|나)(?=\s|$)",
    # 종결 후 남은 조사 제거 (마지막에 적용)
    r"[은는이가을를의도에서]\s*$",
]

# 의미 검색 질의 확장: 핵심 개념별 관련 표현 추가
# - 동의어 수준의 표현만 추가 (광범위한 활동 표현은 임베딩 희석 위험)
# - DB document_chunks 실 텍스트에서 추출한 표현 우선
_SEMANTIC_EXPANSIONS: dict[str, list[str]] = {
    # 일반 HR 키워드
    "출장비": ["출장 정산", "출장 경비", "비용 처리"],
    "법인카드": ["카드 사용", "법카 한도", "카드 승인"],
    "연차": ["연차휴가", "휴가 신청", "연차 사용"],
    "반일연차": ["반차", "오전 반차", "오후 반차"],
    "기숙사": ["기숙사 신청", "기숙사 배정", "숙소"],
    "담당자": ["담당 부서", "연락처", "담당"],
    "학자금": ["자녀 학자금", "학자금 지원", "교육비 지원"],
    # 동호회/동아리 — 동의어만 유지 (활동/사내 표현은 포상·등록 등 다른 개념과 겹쳐 희석)
    "동호회": ["동아리"],
    "동아리": ["동호회"],
    # 동호회 규정 조항별 핵심 표현 (DB 실 텍스트 기반)
    "포상": ["포상금", "포상금액", "시상", "우수 동아리", "심사기준", "연말 포상"],
    "등록 취소": ["등록취소", "지원금 취소", "통폐합", "해체", "해산"],
    "등록취소": ["등록 취소", "지원금 취소", "통폐합", "해체", "해산"],
    "등록": ["동아리 등록", "신규 등록", "등록 요건", "등록 절차"],
    "활동비": ["재정 지원", "활동 지원금", "동아리 활동비", "활동 경비"],
    "지원": ["활동비", "재정 지원", "활동 지원금"],
    "가입": ["회원 가입", "동아리 가입", "탈퇴"],
    "규정": ["사규", "내규", "규정집", "조항", "지침"],
}

# 복합 질문에서 RAG와 분리할 '담당/부서 조회' 성격의 서브질의 (규정 본문 힌트가 없을 때만)
_STRUCTURED_LOOKUP_SUBQUERY_RE = re.compile(
    r"(?:담당자|담당\s*부서|담당부서|담당\s*팀|어느\s*부서|"
    r"누구(?:이|가|야|예요|지|죠)?|책임\s*자|문의\s*처)",
    re.IGNORECASE,
)
_DOC_BODY_SUBQUERY_HINT_RE = re.compile(
    r"(?:절차|기준\s*금액|기준금액|[가-힣]{2,}금액|구입|지급|지원|신청|요건|한도|활동비|"
    r"교체|수당|내용|조항|해지|취소|등록|가입|탈퇴|연차|휴가|출장|비용|"
    r"포상|보험|급여|계약|구체적|지원\s*기준)",
)

_TOKEN_TRAILING_JOSA = re.compile(
    r"(?:을|를|은|는|이|가|도|만|에서|으로|와|과|이나|나|까지|부터|보다)$"
)


def normalize_question(question: str) -> str:
    normalized = " ".join(question.strip().split())
    for filler in FILLER_WORDS:
        normalized = re.sub(rf"(^|\s){re.escape(filler)}(\s|$)", " ", normalized)
    normalized = " ".join(normalized.split())
    for src, dst in SYNONYM_MAP.items():
        normalized = normalized.replace(src, dst)
    return normalized


def should_rewrite(question: str) -> bool:
    normalized = " ".join(question.strip().split())
    if not normalized:
        return False
    if any(hint in normalized for hint in AMBIGUOUS_HINTS):
        return True
    if "?" in normalized:
        return True
    # 구조화된 질문(명시적 키워드 + 조사형 종결)은 그대로 사용한다.
    if any(marker in normalized for marker in ["절차", "규정", "담당", "부서", "연락처", "이메일"]):
        return False
    return len(normalized.split()) <= 2


def rewrite_query(question: str) -> str:
    normalized = normalize_question(question)
    # 너무 짧거나 단편적인 질의를 "~에 대해 알려주세요" 형태로 확장한다.
    tokens = normalized.split()
    if len(tokens) <= 2 and tokens:
        return f"{normalized}에 대해 알려주세요"
    return normalized


def _clean_keyword_token(raw: str) -> str:
    """FTS 토큰에서 구두점·말미 조사를 제거한다 (예: '절차를' → '절차')."""
    t = raw.strip().strip("\"'“”‘’,.!?…、，")
    if len(t) < 2:
        return ""
    for _ in range(5):
        if len(t) < 3:
            break
        m = _TOKEN_TRAILING_JOSA.search(t)
        if not m:
            break
        head = t[: m.start()]
        if len(head) < 2:
            break
        t = head
    return t if len(t) >= 2 else ""


def _is_structured_lookup_only_subquery(fragment: str) -> bool:
    """mobigen 담당자/담당부서 조회 조각만 있고 규정 본문 검색 힌트가 없으면 True."""
    s = fragment.strip()
    if len(s) < 4:
        return False
    if not _STRUCTURED_LOOKUP_SUBQUERY_RE.search(s):
        return False
    if _DOC_BODY_SUBQUERY_HINT_RE.search(s):
        return False
    return True


def generate_keyword_query(normalized_q: str) -> str:
    """FTS용 핵심 키워드 질의 생성.

    조사·어미·의문 종결 표현을 제거하고 명사/핵심어만 남긴다.
    """
    q = normalized_q.strip()
    for pattern in _KEYWORD_STOP_PATTERNS:
        q = re.sub(pattern, "", q).strip()
    # 공백 기준 토큰 → 구두점·말미 조사 정리 후 2글자 미만 제거
    tokens = [_clean_keyword_token(t) for t in q.split() if t.strip()]
    tokens = [t for t in tokens if len(t) >= 2]
    return " ".join(tokens) if tokens else normalized_q


def generate_semantic_query(normalized_q: str, subqueries: list[str]) -> str:
    """의미 기반 검색용 질의 생성.

    subqueries가 있으면 이를 합쳐 의미 범위를 확장하고,
    핵심 개념에 관련 표현을 덧붙여 임베딩 커버리지를 높인다.
    """
    base = normalized_q
    if len(subqueries) > 1:
        base = " ".join(subqueries)

    expansions: list[str] = []
    for keyword, related in _SEMANTIC_EXPANSIONS.items():
        if keyword in base:
            expansions.extend(related)

    if expansions:
        extra = " ".join(dict.fromkeys(expansions))  # 순서 보존 중복 제거
        return f"{base} {extra}".strip()
    return base


def decompose_subqueries(question: str, max_subqueries: int = MAX_SUBQUERIES) -> list[str]:
    separators = [" 그리고 ", " 및 ", ",", "/"]
    parts = [question]
    for sep in separators:
        next_parts: list[str] = []
        for item in parts:
            next_parts.extend(item.split(sep))
        parts = next_parts
    cleaned = []
    for part in parts:
        item = normalize_question(part)
        if item and item not in cleaned:
            cleaned.append(item)
    if not cleaned:
        return []
    return cleaned[:max_subqueries]


def apply_rag_query_focus_for_compound_questions(
    pre_retrieval: dict[str, Any],
    route_type: RouteType,
) -> dict[str, Any]:
    """rag/hybrid에서 서브질의가 둘 이상일 때, 담당자·담당부서 조회 조각을 제외한 문서 검색용 질의로 재구성한다.

    정형 DB 조회(`fetch_user_lookup_chunks`)는 그래프에서 `normalized` 전체를 쓰므로 여기서는 변경하지 않는다.
    """
    if route_type not in ("rag", "hybrid"):
        return pre_retrieval
    raw_subs = pre_retrieval.get("subqueries") or []
    if not isinstance(raw_subs, list) or len(raw_subs) <= 1:
        return pre_retrieval

    doc_subs = [s.strip() for s in raw_subs if isinstance(s, str) and s.strip() and not _is_structured_lookup_only_subquery(s)]
    if not doc_subs:
        return pre_retrieval
    if len(doc_subs) == len(raw_subs):
        return pre_retrieval

    rag_surface = " ".join(doc_subs)
    out = dict(pre_retrieval)
    out["keyword_query"] = generate_keyword_query(rag_surface)
    out["semantic_query"] = generate_semantic_query(rag_surface, doc_subs)

    topic = (out.get("matched_topic") or "").strip()
    if topic:
        kq = str(out.get("keyword_query") or "")
        if topic not in kq:
            out["keyword_query"] = f"{kq} {topic}".strip()
        sq = str(out.get("semantic_query") or "")
        if topic not in sq:
            out["semantic_query"] = f"{sq} {topic}".strip()
    return out


def _last_human_text_from_lc_history(history: list[Any]) -> str | None:
    for m in reversed(history):
        if isinstance(m, HumanMessage) and (m.content or "").strip():
            return str(m.content).strip()
    return None


def _last_human_text_from_dict_history(history: list[dict[str, str]]) -> str | None:
    for item in reversed(history or []):
        if (item.get("role") or "").strip() != "user":
            continue
        content = (item.get("content") or "").strip()
        if content:
            return content
    return None


def _follow_up_introduces_new_domain_hint(follow: str, last_user: str) -> bool:
    for h in sorted(TASK_KNOWLEDGE_DOMAIN_HINTS, key=len, reverse=True):
        if h in follow and h not in last_user:
            return True
    return False


def _topic_prefix_from_prior_user_turn(last_user: str) -> str:
    s = last_user.strip()
    if "담당" in s:
        return s.split("담당", 1)[0].strip()
    # 이전 질문에 이름+호칭이 있으면 그것만 추출 ("이미선 프로" 등)
    # — 전체 문장을 붙이면 이전 의도 키워드(이메일 등)가 merged query에 오염됨
    m = _EXPLICIT_NAMED_PERSON_PRO.search(s)
    if m:
        return m.group(0).strip()
    # "동아리 규정 요약해서 알려줘" → "동아리"처럼 문서 유형 키워드 앞 주제어만 추출
    # 전체 문장을 그대로 붙이면 "요약해서 알려줘" 같은 노이즈가 DB 검색 패턴을 오염시킴
    for boundary in _DOC_TYPE_BOUNDARY_TOKENS:
        if boundary in s:
            prefix = s.split(boundary, 1)[0].strip()
            if prefix:
                return prefix
    return s


def _strip_follow_up_discourse_lead(text: str) -> str:
    t = text.strip()
    t2 = _DISCOURSE_LEAD.sub("", t, count=1)
    return t2.strip()


def contextualize_follow_up_with_prior(question: str, last_human_text: str | None) -> str:
    """직전 사용자 발화와 후속 질문을 하나의 독립 질의로 만든다 (그래프·전처리 공통)."""
    q = (question or "").strip()
    if not last_human_text:
        return q
    # 업무 외 도메인(날씨·스포츠 등) 질문은 이전 업무 컨텍스트와 병합하지 않는다
    if any(hint in q for hint in _OFFSITE_DOMAIN_HINTS):
        return q
    if _EXPLICIT_NAMED_PERSON_PRO.search(q):
        return q
    if _follow_up_introduces_new_domain_hint(q, last_human_text):
        return q
    if len(q) > 35 and not _DISCOURSE_LEAD.match(q):
        return q
    if any(kw in q for kw in _HEADCOUNT_ROSTER_HINTS):
        return q
    q_core = _strip_follow_up_discourse_lead(q)
    topic = _topic_prefix_from_prior_user_turn(last_human_text)
    if not topic:
        return f"{last_human_text} {q}".strip()
    merged = f"{topic} {q_core}".strip()
    return merged


def expand_follow_up_with_conversation(
    question: str,
    history: list[dict[str, str]] | None,
) -> str:
    """API·테스트용: dict 히스토리로 후속 질문을 이전 턴 주제와 합친 문자열을 반환한다."""
    q = (question or "").strip()
    if not history:
        return q
    last_user = _last_human_text_from_dict_history(history)
    return contextualize_follow_up_with_prior(q, last_user)


def expand_question_with_conversation(
    question: str,
    history: list[dict[str, str]],
) -> tuple[str, bool]:
    """대화 히스토리(dict role/content)로 후속 질문 병합 여부를 판별한다."""
    merged = expand_follow_up_with_conversation(question, history)
    return merged, merged.strip() != (question or "").strip()


def generate_metadata_filter(question: str) -> dict[str, object]:
    q = normalize_question(question)
    metadata: dict[str, object] = {}
    for token, mapped in TIME_SCOPE_HINTS.items():
        if token in q:
            metadata["time_scope"] = mapped
            break
    for token, mapped in DOC_TYPE_HINTS.items():
        if token in q:
            metadata["doc_type"] = mapped
            break

    if "부서" in q:
        # "피플팀 부서" 등. "부서명" 안의 "부서"는 제외(의\s*부서 로 잘못 캡처되는 것 방지)
        match = re.search(r"([가-힣A-Za-z0-9]+)\s*부서(?!명)", q)
        if match:
            dep = match.group(1)
            if dep.endswith("의") and len(dep) <= 5:
                pass
            elif dep in ("프로", "님", "씨"):
                pass
            else:
                metadata["department"] = dep
    if "담당자" in q:
        match = re.search(r"([가-힣A-Za-z]+)\s*담당자", q)
        if match:
            metadata["name"] = match.group(1)

    number_patterns = {
        "days": r"(\d+)\s*일",
        "distance_km": r"(\d+)\s*km",
        "amount": r"(\d+)\s*원",
    }
    for prefix, pattern in number_patterns.items():
        match = re.search(pattern, q)
        if match:
            value = int(match.group(1))
            metadata[f"{prefix}_min"] = value
            metadata[f"{prefix}_max"] = value
    return metadata


def build_pre_retrieval(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    if conversation_history:
        surface, conv = expand_question_with_conversation(question, conversation_history)
        original = question
    else:
        surface = question.strip()
        conv = False
        original = question

    rewrite_applied = should_rewrite(surface)
    rewritten = rewrite_query(surface) if rewrite_applied else normalize_question(surface)
    subqueries = decompose_subqueries(rewritten)
    metadata_filter = generate_metadata_filter(rewritten)
    keyword_query = generate_keyword_query(rewritten)
    semantic_query = generate_semantic_query(rewritten, subqueries)
    return {
        "original_query": original,
        "rewritten_query": rewritten,
        "rewrite_applied": rewrite_applied,
        "subqueries": subqueries,
        "metadata_filter": metadata_filter,
        "keyword_query": keyword_query,
        "semantic_query": semantic_query,
        "conversation_expanded": conv,
        "retrieval_surface_query": surface if conv else None,
    }
