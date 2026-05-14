# 정형 DB 조회 전담 모듈 (db_api / hybrid 라우트).
# 사원 이름·부서·이메일 조회, 부서 인원수·명단, mobigen_task_owner 규정 담당자 조회를 모두 담당한다.
# fetch_user_lookup_chunks()가 외부 진입점이며 LLM 파싱 → 규칙 분기 순으로 의도를 파악한다.
"""Office guide `user` + `friday.mobigen_task_owner` lookup for QnA db_api / hybrid routes."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID

from langchain_core.messages import HumanMessage
from db.connect_rdb import AsyncPostgresDB
from services.qna_config import load_runtime_policy
from services.qna_llm import _build_llm_client
from services.qna_types import RetrievalChunk
from services.qna_utils import _strip_json_fences
from utils.logger import LOG

_DB_SCHEMA = (os.getenv("C_DATABASE_SCHEMA") or "").strip() or None
_schema_prefix = f"{_DB_SCHEMA}." if _DB_SCHEMA else ""
_USER_TABLE = f'"{_DB_SCHEMA}"."user"' if _DB_SCHEMA else '"user"'
_EMP_COLS = "emp_nm, emp_cd, emp_id, emp_cphone, emp_email, dept_cd"
_MOBIGEN_TASK_TABLE = "friday.mobigen_task_owner"
_DOCUMENT_TASK_MAPPING_TABLE = "friday.documents_task_mapping"
_RAG_DOCS_TABLE = "friday.documents"
_RAG_CHUNKS_TABLE = "friday.document_chunks"
_DEPT_TABLE = f"{_schema_prefix}dept"
# mobigen 담당자: assignee = user.emp_cd → emp_nm. 담당 부서: user.dept_cd = dept.id → dept.dept_nm (AS department_name).
# mobigen_task_owner.department_name 컬럼은 사용하지 않는다.
# dept_nm·검색어 공백 무시 비교용 (일반 공백 + U+3000)
def _sql_dept_compact(expr: str) -> str:
    return f"replace(replace(btrim(COALESCE({expr}, '')), ' ', ''), chr(12288), '')"
_PEOPLE_TEAM_REFER = "피플팀에 문의 바랍니다"

# mobigen_task_owner + 담당자(user) + 담당 부서명(dept) 공통 FROM … JOIN
_MOBIGEN_FROM_JOIN = f"""
            FROM {_MOBIGEN_TASK_TABLE} mto
            LEFT JOIN {_USER_TABLE} u ON u.emp_cd = mto.assignee
            LEFT JOIN {_DEPT_TABLE} d ON d.id = u.dept_cd
"""


def _sort_dept_names(names: list[str]) -> tuple[str, ...]:
    """긴 이름 우선 매칭용으로 길이 내림차순 정렬."""
    return tuple(sorted(names, key=len, reverse=True))


def _particle_wa_gwa(word: str) -> str:
    """마지막 글자 받침 여부에 따라 조사 '과' 또는 '와'. 한글·숫자·라틴 등 비한글은 '와'."""
    if not word:
        return "와"
    last = word[-1]
    if "\uac00" <= last <= "\ud7a3":
        code = ord(last) - ord("\uac00")
        has_batchim = code % 28 != 0
        return "과" if has_batchim else "와"
    return "와"


def _particle_eun_neun(word: str) -> str:
    """마지막 글자 받침 여부에 따라 조사 '은' 또는 '는'. 비한글 마지막 글자는 '는'."""
    if not word:
        return "은"
    last = word[-1]
    if "\uac00" <= last <= "\ud7a3":
        code = ord(last) - ord("\uac00")
        has_batchim = code % 28 != 0
        return "은" if has_batchim else "는"
    return "는"


async def _fetch_dept_names_from_db_async() -> list[str]:
    db = AsyncPostgresDB()
    await db.connect()
    try:
        sql = f"""
            SELECT DISTINCT dept_nm
            FROM {_DEPT_TABLE}
            WHERE btrim(COALESCE(dept_nm, '')) <> ''
            ORDER BY dept_nm
        """
        recs = await db.fetch(sql)
        return [str(r["dept_nm"]).strip() for r in recs if r.get("dept_nm")]
    finally:
        await db.close()


# DB를 쓸 수 없거나 빈 결과일 때 정규식·부서-사원 추출 단위 테스트가 동작하도록 최소 목록을 둔다.
_FALLBACK_CANONICAL_DEPARTMENTS: tuple[str, ...] = (
    "피플팀",
    "메시징사업그룹",
    "지능데이터 서비스팀",
)


def _load_canonical_department_names() -> tuple[str, ...]:
    """서버 시작 시 dept 테이블에서 부서명을 로드.

    asyncio.run()을 직접 호출하지 않고 ThreadPoolExecutor 내에서 실행해
    FastAPI/uvicorn 등 이미 이벤트 루프가 돌고 있는 환경에서의 RuntimeError를 방지한다.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            names = pool.submit(asyncio.run, _fetch_dept_names_from_db_async()).result(timeout=10)
        if names:
            LOG.info("qna_structured_db: dept names loaded from DB (%d entries)", len(names))
            return _sort_dept_names(names)
        LOG.warning("qna_structured_db: dept table returned no rows")
    except Exception as e:
        LOG.warning("qna_structured_db: failed to load dept names from DB (%s)", e)
    return _sort_dept_names(list(_FALLBACK_CANONICAL_DEPARTMENTS))


_CANONICAL_DEPARTMENTS_TUPLE: tuple[str, ...] = _load_canonical_department_names()
_CANONICAL_DEPT_NAME_SET: frozenset[str] = frozenset(_CANONICAL_DEPARTMENTS_TUPLE)
# 공백 제거 버전 → 원본 이름 매핑 ('AX플랫폼팀' 같이 공백 없는 canonical vs 'AX 플랫폼팀' 같은 사용자 입력 불일치 해소)
_CANONICAL_COMPACT_TO_FULL: dict[str, str] = {
    name.replace(" ", ""): name for name in _CANONICAL_DEPARTMENTS_TUPLE
}

# Case 5 — 사용자가 줄인 compact 표기(예: 지능데이터팀)를 canonical로 고정(운영 시 확장·DB 우선)
_DEPT_COMPACT_ALIAS_TO_CANONICAL: dict[str, str] = {
    "지능데이터팀": "지능데이터 서비스팀",
}


def _dept_compact_key(s: str) -> str:
    return (s or "").strip().replace(" ", "").replace("\u3000", "")


def _resolve_dept_to_canonical(dept: str) -> str:
    """추출된 부서명을 canonical 이름으로 정규화.

    'AX 플랫폼팀'(사용자 입력 공백 포함) → 'AX플랫폼팀'(canonical) 처럼
    공백 차이로 ILIKE 미매칭이 생기는 경우를 해소한다.
    Case 5: 공백만 다른 동의어·짧은 표기는 compact 매핑·별칭·유사도로 보강한다.
    """
    d = (dept or "").strip()
    if not d:
        return d
    if d in _CANONICAL_DEPT_NAME_SET:
        return d
    compact = _dept_compact_key(d)
    hit = _CANONICAL_COMPACT_TO_FULL.get(compact)
    if hit:
        return hit
    alias = _DEPT_COMPACT_ALIAS_TO_CANONICAL.get(compact)
    if alias and alias in _CANONICAL_DEPT_NAME_SET:
        return alias
    if len(compact) < 4:
        return d
    best: str | None = None
    best_r = 0.0
    for c_full in _CANONICAL_DEPARTMENTS_TUPLE:
        cc = _dept_compact_key(c_full)
        if not cc:
            continue
        r = difflib.SequenceMatcher(a=compact, b=cc).quick_ratio()
        if r > best_r:
            best_r = r
            best = c_full
    if best and best_r >= 0.88:
        return best
    return d


# 'OO그룹인 김철수' 등 부서명 뒤 이어지는 서술격 '인' 제거 시 참고 (토큰이 공식 부서와 같을 때는 파일 집합으로 판별)
_ORG_TAIL_FOR_IN_DROP = re.compile(
    r"(?:팀|부|본부|그룹|실|센터|연구소|처|원|국|위|단|TF|팀_D|연구팀|개발팀|영업부|사업본부|"
    r"사업부문|연구그룹|연구실|솔루션|플랫폼|마케팅|Ops|AI|DX|AX|NI|DN|IoT|PS|섹터|본부)$",
    re.UNICODE,
)


def _canonical_dept_flex_pattern(dept: str) -> re.Pattern[str]:
    """부서명에 띄어쓰기가 끼어 들어가도 매칭되도록 정규식 생성."""
    d = dept.strip()
    if not d:
        return re.compile(r"$^")
    if re.search(r"\s", d):
        segs = [s for s in re.split(r"\s+", d) if s]
        body = r"\s+".join(re.escape(s) for s in segs)
    else:
        body = r"\s*".join(re.escape(c) for c in d)
    return re.compile(body, re.UNICODE | re.IGNORECASE)


# 모듈 로드 시 한 번만 컴파일 — try_extract_known_dept_person_pair 루프마다 재컴파일 방지
_CANONICAL_DEPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (dept, _canonical_dept_flex_pattern(dept)) for dept in _CANONICAL_DEPARTMENTS_TUPLE
)

_JOSA_AFTER_DEPT = re.compile(r"^(?:인|의|은|는)\s*")

# 직급·호칭 힌트 — 부서명+이름 패턴 트리거 조건으로만 사용 (프로·님 포함)
_PERSON_TITLE_HINTS: frozenset[str] = frozenset({
    "프로", "님", "사원", "대리", "과장", "차장", "부장", "주임", "계장", "팀장", "이사", "상무", "전무", "대표",
})
# 이름 뒤 파싱용: 한글 조사(프로의·사원은 …) 앞에는 \\b가 없으므로 긍정 전방 탐색만 사용한다.
_PERSON_TITLE_PATTERN = re.compile(
    r"(?:프로|님|사원|대리|과장|차장|부장|주임|계장|팀장|이사|상무|전무|대표|책임)"
    r"(?=\s|$|[\.,!?…]|[\uac00-\ud7a3])",
    re.UNICODE,
)


def _question_mentions_person_title(raw: str) -> bool:
    """프로·사원 등은 부분 문자열로도 질문에 들어가므로 힌트 집합으로 본다.
    '책임'만 단독일 때는 '책임자' 등과 구분한다."""
    if any(t in raw for t in _PERSON_TITLE_HINTS):
        return True
    return bool(re.search(r"(?<![가-힣])책임(?![가-힣])", raw))


_TITLES_FOR_GLUE_INSERT: tuple[str, ...] = tuple(
    sorted(_PERSON_TITLE_HINTS | {"책임"}, key=len, reverse=True)
)
_GLUED_NAME_TITLE = re.compile(
    rf'([가-힣]{{2,5}})({"|".join(re.escape(t) for t in _TITLES_FOR_GLUE_INSERT)})(?=\s|$|[\.,!?…]|[\uac00-\ud7a3])',
    re.UNICODE,
)


def _insert_space_before_glued_person_titles(s: str) -> str:
    """Case 4 — 이름과 직함이 붙어 있으면(이미선프로) 직함 앞에 공백 삽입."""

    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)} {m.group(2)}"

    return _GLUED_NAME_TITLE.sub(repl, s)


def _compact_query_for_hints(s: str) -> str:
    """띄어쓰기 삽입 정규화(임시 사원증 등) 뒤에도 업무 힌트 문자열이 매칭되도록 공백을 제거한다."""
    return "".join(((s or "").strip()).split())


def normalize_structured_db_query(question: str) -> str:
    """§8.2 정형 DB 조회 직전 1차 정규화: 직함 붙여쓰기·과도한 공백 정리.

    few-shot만으로는 DB 파라미터가 정리되지 않으므로, 코드에서 반드시 적용한다.
    """
    t = (question or "").strip()
    t = _insert_space_before_glued_person_titles(t)
    # 업무명 구분(판교사옥 > 임시 사원증 …)이 한글 토큰에 붙지 않도록 구분자만 공백화
    t = re.sub(r"[>＞]+", " ", t)
    t = " ".join(t.split())
    return t


def try_extract_known_dept_person_pair(raw: str) -> tuple[str, str] | None:
    """공식 부서명 + 'OO 프로/사원/대리…' 패턴이면 (이름, 부서힌트) 반환. clarify 없이 사원 조회에 사용."""
    raw = normalize_structured_db_query(raw)
    if not _CANONICAL_DEPT_PATTERNS or not _question_mentions_person_title(raw):
        return None
    # "사원 명단/목록/몇 명" 처럼 직함이 아닌 집합 표현으로 쓰인 경우 — roster 분기가 처리해야 한다
    if _ROSTER_TRIGGER.search(raw):
        return None
    for dept, pattern in _CANONICAL_DEPT_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        tail = raw[m.end():].lstrip()
        # "(주)모비젠 팀 김태수 프로 ..."처럼 부서명 뒤 보조 토큰 '팀'이 한 번 더 들어와도 이름 파싱 유지.
        tail = re.sub(r"^(?:팀)\s+", "", tail, count=1)
        # '인|의|은|는' 만 제거. 단일 '이|가'는 이름 첫 글('이미선')과 충돌해 앞글자를 잘라먹는다.
        tail = _JOSA_AFTER_DEPT.sub("", tail)
        m2 = re.match(r"([가-힣]{2,4})\s*(?:" + _PERSON_TITLE_PATTERN.pattern + r")?", tail)
        if not m2:
            continue
        emp_nm = m2.group(1)
        if len(emp_nm) < 2:
            continue
        return emp_nm, dept
    return None


def _strip_topic_in_after_org_token(pick: str) -> str:
    """'메시징사업그룹인'처럼 부서명+서술격 '인'이 붙어 한글 토큰이 된 경우."""
    t = _strip_trailing_josa(pick.strip())
    if len(t) >= 4 and t.endswith("인"):
        stem = t[:-1]
        if stem in _CANONICAL_DEPT_NAME_SET or bool(_ORG_TAIL_FOR_IN_DROP.search(stem)):
            return stem
    return t


# 사내 규정 업무(담당자/담당 부서) 질문 — mobigen_task_owner 매칭 힌트 (예시·운영 데이터 기준)
_TASK_KNOWLEDGE_HINTS = frozenset(
    {
        "법인카드",
        "휴가",
        "근무",
        "원격",
        "파견",
        "심야",
        "휴일근로",
        "휴일",
        "휴직",
        "복직",
        "4대보험",
        "계산서",
        "보증",
        "보증보험",
        "증명",
        "직인",
        "예산",
        "경비",
        "국책",
        "구매",
        "복리",
        "복리후생",
        "우리사주",
        "영업",
        "영업지원",
        "건강검진",
        "사무용품",
        "주재",
        "매입",
        "매출",
        "출장비",
        "동호회",
        "동아리",
        "정부지원금",
        "사옥",
        "그룹웨어",
        "임시사원증",
        "자기계발비",
        "장비",
    }
)
# 동의어 쌍 — DB task_name이 한쪽 표기로만 등록된 경우 반대 표기도 함께 검색
_TASK_KNOWLEDGE_SYNONYMS: dict[str, str] = {
    "동아리": "동호회",
    "동호회": "동아리",
}
TASK_KNOWLEDGE_DOMAIN_HINTS = _TASK_KNOWLEDGE_HINTS
_TOPIC_STOP_FOR_MOBIGEN = frozenset(
    {
        "규정",
        "사내",
        "관련",
        "업무",
        "궁금",
        "문의",
        "알려줘",
        "알려",
        "있는데",
        "있어",
        "해서",
        "어떻게",
        "무엇",
        "뭐야",
        "뭔가",
        "담당자",
        "담당",
        "부서",
        "관리",
        "어느",
        "누구",
        "프로",
        "입니다",
        "이에요",
        "예요",
        "해요",
        "주세요",
        "바랍니다",
    }
)
_TASK_OWNER_INTENT = re.compile(
    r"(?:담당자|담당\s*부서|어느\s*부서|관리해|관리하는|누구(?:이|가|야|예요|지|죠)?|책임\s*자)",
    re.IGNORECASE,
)
# 업무명(task_name) 1차 · 세부(task_detail) 2차 — "기타의 정부지원금 담당자" 등
_MOBIGEN_POSSESSIVE_TASK = re.compile(
    r"(?P<tn>[\uac00-\ud7a3A-Za-z0-9]+(?:\s+[\uac00-\ud7a3A-Za-z0-9]+)*?)\s*의\s*"
    r"(?P<td>[\uac00-\ud7a3A-Za-z0-9]+(?:\s+[\uac00-\ud7a3A-Za-z0-9]+)*?)"
    # \b는 한글 음절 사이에서 기대와 다르게 동작할 수 있어 제외한다.
    r"(?=\s*(?:담당자|담당|누구|알려|부서|프로)|[?？]|$)",
    re.UNICODE,
)

_NOISE = (
    "담당자",
    "담당",
    "이메일",
    "이메일주소",
    "주소",
    "알려줘",
    "알려",
    "휴대폰",
    "핸드폰",
    "번호",
    "부서명",
    "부서",
    "연락처",
    "사번",
    "직책",
    "그룹장",
    "팀장",
    "실장",
    "아이디",
    "id",
    "ID",
    "확인",
    "조회",
    "누구",
    "뭐야",
    "무엇",
    "사원",
    "대리",
    "과장",
    "차장",
    "부장",
    "주임",
    "계장",
    "이사",
    "상무",
    "전무",
)

_CHUNK_SOURCE = "office_guide.user"
_MOBIGEN_SOURCE = "office_guide.friday.mobigen_task_owner"
_TASK_RELATED_DOC_SOURCE = "office_guide.friday.documents_task_mapping"
_ROSTER_KIND = "department_roster"
KIND_CLARIFY_DEPARTMENTS = "clarify_departments"
KIND_DEPT_HEADCOUNT = "department_headcount"
# 사원 디렉터리 row 청크(테스트·답변 선택용 메타)
KIND_PERSON_ROW = "person_row"
KIND_TASK_OWNER = "task_owner"
KIND_TASK_OWNER_CLARIFY = "task_owner_clarify"
KIND_TASK_OWNER_NOT_FOUND = "task_owner_not_found"
KIND_TASK_RELATED_DOCUMENT = "task_related_document"
KIND_PERSON_NOT_FOUND = "person_not_found"
KIND_PERSON_AMBIGUOUS = "person_ambiguous"
KIND_PERSON_MANY_NAMES = "person_many_name_candidates"

# 부서 소속 인원·명단 질문 (예: AX 사업팀 사원이 몇 명…)
_ROSTER_TRIGGER = re.compile(r"(?:몇\s*명|몇명|명수|전체|모두|명단|리스트|목록)", re.IGNORECASE)
_DEPT_BEFORE_SAWON = re.compile(r"(?P<dept>[\w가-힣\s.\-]{2,60}?)\s*사원", re.UNICODE)

# 질문에 이메일이 있으면 한글 조각(가진, 사람의 …)보다 이 값을 먼저 DB 검색어로 쓴다.
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)

# 이름 뒤에 붙는 조사(…의 부서명)가 검색어에 붙으면 ILIKE가 어긋난다.
_JOSA_TRAIL = re.compile(
    r"(?:의|은|는|이|가|을|를|와|과|도|만|에게|한테|께|로|에서|으로|부터|까지)$"
)
# 질문형 어미(담당이야 → 담당, 어디야 → 어디) — _JOSA_TRAIL만으로는 '이야'가 남음
_INFORMAL_QUERY_TAIL = re.compile(
    r"(?:이야|예요|이지|이죠|이라서|라서|야|요|지|죠|니까)$",
    re.UNICODE,
)


def _strip_trailing_josa(s: str) -> str:
    t = s.strip()
    while len(t) >= 2:
        n = _JOSA_TRAIL.sub("", t)
        if n == t:
            break
        t = n
    return t


def _looks_like_org_unit_label(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if re.search(r"[A-Za-z]", s):
        return True
    markers = ("팀", "부", "본부", "센터", "그룹", "실", "단", "처", "원", "위", "국")
    return any(m in s for m in markers)


def _sanitize_dept_phrase(s: str) -> str:
    cleaned = "".join(
        c for c in s.strip() if c.isalnum() or c.isspace() or "\uac00" <= c <= "\ud7a3" or c in ".-_&"
    )
    return " ".join(cleaned.split())[:80]


_VAGUE_TEAM = re.compile(
    r"(?P<prefix>[\w가-힣]+(?:\s+[\w가-힣]+){0,2})\s*팀\b",
    re.UNICODE,
)


def extract_vague_team_keyword(question: str) -> str | None:
    """'AX 팀 정보'처럼 팀명이 모호할 때 부서명 DISTINCT 검색용 키워드."""
    raw = normalize_structured_db_query(question)
    if extract_department_roster_phrase(raw):
        return None
    if _EMAIL_RE.search(raw):
        return None
    if _ROSTER_TRIGGER.search(raw):
        return None
    # 인명+직함 질의는 사원 조회가 우선이며, vague team clarify로 보내면 오탐이 된다.
    if _question_mentions_person_title(raw):
        return None
    if "팀" not in raw:
        return None
    m = _VAGUE_TEAM.search(raw)
    if not m:
        return None
    prefix = _sanitize_dept_phrase(m.group("prefix"))
    if len(prefix) < 1:
        return None
    if not _looks_like_org_unit_label(prefix + "팀"):
        return None
    # '피플팀'처럼 접두+팀이 붙어 있으면 실제 팀명인데, 정규식이 prefix=피플로만 잡아 모호 팀으로 오인한다.
    # 'AX 플랫폼팀' → compact 'AX플랫폼팀'도 canonical 체크에 포함한다.
    assembled = _sanitize_dept_phrase(prefix + "팀")
    if assembled in _CANONICAL_DEPT_NAME_SET or assembled.replace(" ", "") in _CANONICAL_COMPACT_TO_FULL:
        return None
    return prefix[:40]


# '피플팀 구성원 정보' / '에이전트 팀 구성원'(띄어쓰기) 등 — 팀 단위 명단 질문
# 이전 패턴은 '에이전트' + ' 팀'을 (?:…)* 로 소비한 뒤 또 리터럴 '팀'을 요구해
# '에이전트 팀 구성원' 형태가 매칭되지 않았다.
_DEPT_TEAM_MEMBER = re.compile(
    r"(?P<dept>.{1,80}?팀)\s*(?:구성원|팀원|멤버|소속)",
    re.UNICODE,
)
# 'OO팀 정보 알려줘' — 구성원/명단 키워드 없이도 팀 구성 조회로 처리
_DEPT_TEAM_INFO = re.compile(
    r"(?P<dept>.{1,80}?팀)\s*정보",
    re.UNICODE,
)


def _extract_dept_team_member_roster_dept(raw: str) -> str | None:
    for rx in (_DEPT_TEAM_MEMBER, _DEPT_TEAM_INFO):
        m = rx.search(raw)
        if not m:
            continue
        dept = _sanitize_dept_phrase(m.group("dept"))
        if len(dept) < 2 or not _looks_like_org_unit_label(dept):
            continue
        resolved = _resolve_dept_to_canonical(dept)
        # _DEPT_TEAM_INFO('OO팀 정보') 매칭은 canonical 이름이 확인된 경우에만 roster로 처리.
        # canonical 미확인 시 vague team 분기(extract_vague_team_keyword)가 처리한다.
        if rx is _DEPT_TEAM_INFO and resolved not in _CANONICAL_DEPT_NAME_SET:
            continue
        return resolved
    return None


def extract_department_roster_phrase(question: str) -> str | None:
    """'AX 사업팀 사원이 몇 명…' 또는 'OO팀 구성원 …'처럼 부서 단위 명단 질문이면 dept_nm 검색어."""
    raw = normalize_structured_db_query(question)
    mem = _extract_dept_team_member_roster_dept(raw)
    if mem:
        return mem
    if "사원" not in raw or not _ROSTER_TRIGGER.search(raw):
        return None
    m = _DEPT_BEFORE_SAWON.search(raw)
    if not m:
        return None
    dept = _sanitize_dept_phrase(m.group("dept"))
    if len(dept) < 2 or not _looks_like_org_unit_label(dept):
        return None
    return _resolve_dept_to_canonical(dept)


# '피플팀은 몇 명이야?'처럼 사원·명단 없이 부서 인원만 묻는 경우
_DEPT_HEADCOUNT = re.compile(
    r"(?P<dept>[\w가-힣]+(?:\s+[\w가-힣]+){0,3})\s*(?:은|는|이|가)?\s*"
    r"(?:몇\s*명|몇명|명수)(?:이야|야|인가요|이지|예요|요|[\?？]|\s|$)",
    re.UNICODE,
)
_DEPT_HEADCOUNT_SKIP_DEPTS = frozenset(
    {"전체", "모든", "각", "해당", "이번", "지금", "총", "합계", "모두"}
)
_DEPT_HEADCOUNT_TRAIL_TOPIC = re.compile(r"(?:은|는|이|가)$")


def _strip_trailing_topic_particle_from_dept(s: str) -> str:
    """'피플팀은'처럼 조사가 dept 토큰에 붙어 캡처된 경우 제거."""
    t = s.strip()
    while len(t) >= 2:
        n = _DEPT_HEADCOUNT_TRAIL_TOPIC.sub("", t)
        if n == t:
            break
        t = n
    return t


# 'AX플랫폼팀 구성원은 몇 명?' → 정규식이 dept를 'AX플랫폼팀 구성원'으로 잡는데, dept_nm은 보통 '…팀'만 있다.
# '피플팀 사원 몇 명?' → dept 캡처가 '피플팀 사원'이 되므로 '사원'도 동일하게 제거한다.
_HEADCOUNT_ROLE_SUFFIX = re.compile(
    r"\s+(?:구성원|팀원|직원|멤버|인원|소속|사원)\s*$",
    re.UNICODE,
)


def _strip_headcount_collective_suffix(dept: str) -> str:
    """부서명 뒤에 붙은 집합 표현(구성원·팀원·사원 등)을 제거해 dept_nm ILIKE와 맞춘다."""
    t = dept.strip()
    while len(t) >= 2:
        n = _HEADCOUNT_ROLE_SUFFIX.sub("", t)
        if n == t:
            break
        t = n.strip()
    return t


def extract_department_headcount_phrase(question: str) -> str | None:
    """'피플팀은 몇 명이야?'처럼 부서명 + 인원 수 질문이면 dept_nm 검색어.

    'OO팀 구성원은 몇 명?'처럼 로스터형 표현과 겹칠 때는 인원 수(headcount)를 우선한다.
    """
    raw = normalize_structured_db_query(question)
    if not _ROSTER_TRIGGER.search(raw):
        return None
    last: str | None = None
    for m in _DEPT_HEADCOUNT.finditer(raw):
        dept = _strip_trailing_topic_particle_from_dept(_sanitize_dept_phrase(m.group("dept")))
        dept = _strip_headcount_collective_suffix(dept)
        if len(dept) < 2:
            continue
        if dept in _DEPT_HEADCOUNT_SKIP_DEPTS:
            continue
        if not _looks_like_org_unit_label(dept):
            continue
        last = _resolve_dept_to_canonical(dept)
    if last is not None:
        return last
    return None


def _strip_mobigen_search_token(t: str) -> str:
    """연속 한글 토큰에서 조사·구어체 어미를 걷어내 mobigen ILIKE 검색어로 쓴다."""
    t = t.strip()
    if len(t) < 2:
        return t
    t = _strip_trailing_josa(t)
    while len(t) >= 2:
        n = _INFORMAL_QUERY_TAIL.sub("", t)
        if n == t:
            break
        t = n
    return _strip_trailing_josa(t)


def _extract_task_owner_search_patterns(question: str) -> list[str]:
    """mobigen_task_owner ILIKE용 토큰(긴 것 우선, 최대 3개).

    '휴가 규정은 어느 부서가 담당이야?'처럼 조사가 붙은 긴 토큰(규정은·부서가·담당이야)만
    상위 3개에 들어가면 DB ILIKE가 실패한다. 정규화 후 업무 힌트(휴가 등)를 앞에 둔다.
    """
    raw = normalize_structured_db_query((question or "").strip())
    compact = _compact_query_for_hints(raw)
    tokens = re.findall(r"[\uac00-\ud7a3]{2,12}", raw)
    seen: set[str] = set()
    picked: list[str] = []
    for t in sorted(set(tokens), key=len, reverse=True):
        norm = _strip_mobigen_search_token(t)
        if len(norm) < 2:
            continue
        if norm in _TOPIC_STOP_FOR_MOBIGEN or norm in _NOISE:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        picked.append(norm)
        if len(picked) >= 3:
            break

    hints = [
        h
        for h in sorted(_TASK_KNOWLEDGE_HINTS, key=len, reverse=True)
        if h in raw or h in compact
    ]
    merged: list[str] = []
    seen_m: set[str] = set()
    for h in hints:
        if h not in seen_m:
            merged.append(h)
            seen_m.add(h)
    for p in picked:
        if p not in seen_m:
            merged.append(p)
            seen_m.add(p)
        if len(merged) >= 3:
            break
    return merged[:3]


def _task_owner_answer_preferences(raw: str) -> tuple[bool, bool]:
    """(담당자 안내 여부, 담당 부서 안내 여부)."""
    r = raw
    wants_assignee = bool(re.search(r"(?:담당자|누구|프로|책임\s*자)", r))
    wants_dept = bool(re.search(r"(?:부서|팀|어느)", r))
    if wants_assignee and wants_dept:
        return True, True
    if wants_assignee:
        return True, False
    if wants_dept:
        return False, True
    return True, True


def is_regulation_task_owner_question(question: str) -> bool:
    """사내 규정 업무 담당자/담당 부서(mobigen_task_owner) 조회로 볼 수 있는 질문이면 True. 라우트 db_api 우선용."""
    raw = normalize_structured_db_query((question or "").strip())
    compact = _compact_query_for_hints(raw)
    if not _TASK_OWNER_INTENT.search(raw):
        return False
    # 공지·게시물 위치 문의 + 담당 부서 — 규정 업무 담당 DB가 아님
    if ("공지" in raw or "게시" in raw) and ("어디" in raw or "찾" in raw or "위치" in raw):
        if "규정" not in raw and "사내" not in raw:
            return False
    if extract_department_roster_phrase(raw) or extract_department_headcount_phrase(raw):
        return False
    if "규정" in raw or "사내" in raw:
        return True
    if any(h in raw or h in compact for h in _TASK_KNOWLEDGE_HINTS):
        return True
    # 힌트셋에 없는 업무 키워드 대응 — 4글자 이상 토큰이 있으면 mobigen 조회 시도.
    # 한국어 인명은 보통 2~3글자이므로 4글자 이상이면 업무 도메인 용어일 가능성이 높다.
    patterns = _extract_task_owner_search_patterns(raw)
    return any(len(p) >= 4 for p in patterns)


def _should_skip_generic_person_ilike_lookup(raw: str) -> bool:
    """장비·규정·기준 금액 등 문서형 질의가 직무 토큰(예: '디자인')으로 ILIKE 사원 조회 폴백에 걸리지 않게 한다."""
    if not (raw or "").strip():
        return False
    rl = raw.lower()
    has_equip = any(k in raw for k in ("장비", "업무용", "노트북")) or "pc" in rl
    has_policy = any(
        k in raw for k in ("규정", "지침", "기준", "금액", "지급", "구매")
    )
    if not (has_equip and has_policy):
        return False
    if re.search(r"(?:담당자|책임\s*자|담당\s*부서)", raw):
        return False
    if "누구" in raw or "누가" in raw:
        return False
    return True


def structured_db_enabled() -> bool:
    return os.getenv("QNA_STRUCTURED_DB", "1").strip().lower() in ("1", "true", "yes", "on")


def _structured_parse_mode() -> str:
    """정형 질의 파싱 모드.

    - rule_based: regex/규칙만 사용
    - hybrid: LLM 우선 시도, 실패/저신뢰면 rule_based fallback
    - llm: LLM 우선 강제(그래도 실패 시 rule_based fallback)
    """
    mode = (os.getenv("QNA_STRUCTURED_PARSE_MODE") or "hybrid").strip().lower()
    if mode in {"rule_based", "hybrid", "llm"}:
        return mode
    return "hybrid"


def _llm_parse_timeout(timeout_sec: float) -> int:
    raw = os.getenv("QNA_STRUCTURED_PARSE_TIMEOUT", "4").strip()
    try:
        t = int(raw)
    except ValueError:
        t = 4
    return max(1, min(int(timeout_sec), t))


def _llm_parse_conf_threshold() -> float:
    raw = os.getenv("QNA_STRUCTURED_PARSE_CONF_THRESHOLD", "0.70").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 0.70
    return min(1.0, max(0.0, v))


def _structured_db_timeout(default: float = 8.0) -> float:
    raw = os.getenv("QNA_STRUCTURED_DB_TIMEOUT", str(default)).strip()
    try:
        timeout = float(raw)
    except ValueError:
        LOG.warning("qna_structured_db: invalid QNA_STRUCTURED_DB_TIMEOUT=%r, fallback=%s", raw, default)
        timeout = default
    return max(1.0, timeout)


_STRUCTURED_INTENTS = frozenset(
    {
        "person_lookup",
        "dept_headcount",
        "dept_roster",
        "task_owner",
        "unknown",
    }
)


def _parse_query_plan_with_llm(question: str, timeout_sec: float) -> dict[str, Any] | None:
    if _structured_parse_mode() == "rule_based":
        return None

    policy = load_runtime_policy()
    llm = _build_llm_client(
        policy, temperature=0, timeout=_llm_parse_timeout(timeout_sec), disable_streaming=True
    )
    if llm is None:
        return None

    prompt = f"""
당신은 사내 정형 DB 질의 파서입니다.
아래 질문을 분석해 JSON 1개만 출력하세요. (마크다운 금지)

출력 스키마:
{{
  "intent": "person_lookup|dept_headcount|dept_roster|task_owner|unknown",
  "person_name": "<이름 또는 빈 문자열>",
  "dept_name": "<부서명 또는 빈 문자열>",
  "lookup_term": "<인명/이메일/사번/부서 키워드 또는 빈 문자열>",
  "task_name": "<업무명 또는 빈 문자열>",
  "task_detail": "<업무 세부 또는 빈 문자열>",
  "ask_assignee": true,
  "ask_department": true,
  "confidence": 0.0
}}

분류 기준:
- person_lookup: 사원 이메일/휴대폰/연락처/사번/부서 조회
- dept_headcount: 특정 부서 인원 수 조회
- dept_roster: 특정 부서 구성원/명단/팀원 조회
- task_owner: 규정/업무 항목의 담당자/담당부서 조회
- unknown: 애매하거나 판단 불가

규칙:
- confidence는 0~1 실수
- 질문에 없는 값을 지어내지 말 것
- 확실하지 않으면 intent=unknown
- ask_assignee/ask_department는 질문 의도를 반영
- '담당자'가 있어도 특정 사람을 찾는 질문이면 person_lookup, 업무·규정 담당을 묻는 질문이면 task_owner

예시:
질문: "법인카드 담당자가 누구야?"
출력: {{"intent": "task_owner", "person_name": "", "dept_name": "", "lookup_term": "", "task_name": "법인카드", "task_detail": "", "ask_assignee": true, "ask_department": false, "confidence": 0.95}}

질문: "이미선 프로 담당자 연락처 알려줘"
출력: {{"intent": "person_lookup", "person_name": "이미선", "dept_name": "", "lookup_term": "이미선", "task_name": "", "task_detail": "", "ask_assignee": false, "ask_department": false, "confidence": 0.95}}

질문: "기타의 정부지원금 담당자 알려줘"
출력: {{"intent": "task_owner", "person_name": "", "dept_name": "", "lookup_term": "", "task_name": "기타", "task_detail": "정부지원금", "ask_assignee": true, "ask_department": false, "confidence": 0.88}}

질문:
\"\"\"{question}\"\"\"
""".strip()

    try:
        msg = llm.invoke([HumanMessage(content=prompt)])
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        data = json.loads(_strip_json_fences(raw))
    except Exception as e:
        LOG.debug("qna_structured_db: llm parse failed (%s)", e)
        return None

    intent = str(data.get("intent") or "").strip()
    if intent not in _STRUCTURED_INTENTS:
        return None
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf))

    return {
        "intent": intent,
        "person_name": str(data.get("person_name") or "").strip(),
        "dept_name": str(data.get("dept_name") or "").strip(),
        "lookup_term": str(data.get("lookup_term") or "").strip(),
        "task_name": str(data.get("task_name") or "").strip(),
        "task_detail": str(data.get("task_detail") or "").strip(),
        "ask_assignee": bool(data.get("ask_assignee", True)),
        "ask_department": bool(data.get("ask_department", True)),
        "confidence": conf,
    }


def _headcount_chunks_for_dept(dept_name: str, timeout_sec: float) -> list[RetrievalChunk]:
    n = _blocking_run_coro(_count_department_members_async(dept_name), timeout_sec)
    if n is None:
        return []
    if n > 0:
        dept_names = _blocking_run_coro(_fetch_distinct_dept_nm_async(dept_name), timeout_sec)
        if dept_names is None:
            return []
        if len(dept_names) >= 2:
            pg = _particle_wa_gwa(dept_name)
            shown = ", ".join(dept_names[:8])
            if len(dept_names) > 8:
                shown += f" 외 {len(dept_names) - 8}개"
            body = (
                f"{dept_name}{pg} 일치하는 부서명이 {shown} 등 여러 단위로 나뉘어 있습니다. "
                f"현재 검색 조건에 해당하는 등록 인원은 총 {n}명입니다. "
                f"한 부서만 알고 싶으면 정확한 부서명을 입력해 주세요."
            )
            return [_structured_guidance_chunk(KIND_DEPT_HEADCOUNT, "부서 인원", body)]
        en = _particle_eun_neun(dept_name)
        body = f"{dept_name}{en} {n}명으로 구성되어 있습니다."
        return [_structured_guidance_chunk(KIND_DEPT_HEADCOUNT, "부서 인원", body)]
    body = (
        f"{dept_name}에 해당하는 부서명으로는 사원 정보를 찾지 못했습니다. "
        f"다른 부서명으로 다시 질문해 주세요."
    )
    return [_structured_guidance_chunk(KIND_DEPT_HEADCOUNT, "부서 인원", body)]


def _roster_chunks_for_dept(dept_name: str, timeout_sec: float) -> list[RetrievalChunk]:
    rows = _blocking_run_coro(_fetch_department_members_async(dept_name), timeout_sec)
    if rows is None:
        return []
    if rows:
        return [_roster_summary_chunk(rows, dept_name)]
    LOG.info("qna_structured_db: roster empty (dept=%r)", dept_name)
    return []


def _task_owner_chunks_from_slots(
    raw: str,
    timeout_sec: float,
    task_name: str,
    task_detail: str,
) -> list[RetrievalChunk]:
    tn = _sanitize_like_term(task_name)
    td = _sanitize_like_term(task_detail)
    if len(tn) >= 2 and len(td) >= 2:
        scoped = _blocking_run_coro(
            _fetch_mobigen_task_scoped_by_name_detail_async(tn, td),
            timeout_sec,
        )
        if scoped:
            return _mobigen_chunks_from_rows(raw, scoped, timeout_sec)
    # LLM이 task_name을 추출했으면 is_regulation_task_owner_question regex 게이트를 우회해
    # 직접 ILIKE 검색한다. regex가 해당 질문을 업무 담당 질문으로 인식하지 못해도 LLM 판단을 살린다.
    if len(tn) >= 2:
        patterns = [tn] + ([td] if len(td) >= 2 else [])
        rows = _blocking_run_coro(_fetch_mobigen_task_rows_async(patterns[:3]), timeout_sec)
        if rows:
            return _mobigen_chunks_from_rows(raw, rows, timeout_sec)
    return fetch_mobigen_task_owner_chunks(raw, timeout_sec)


def _chunks_from_llm_plan(raw: str, timeout_sec: float) -> list[RetrievalChunk] | None:
    mode = _structured_parse_mode()
    if mode == "rule_based":
        return None

    plan = _parse_query_plan_with_llm(raw, timeout_sec)
    if not plan:
        return None
    conf = float(plan.get("confidence") or 0.0)
    if mode == "hybrid" and conf < _llm_parse_conf_threshold():
        return None

    intent = str(plan.get("intent") or "")
    person_name = _sanitize_like_term(str(plan.get("person_name") or ""))
    dept_name = _resolve_dept_to_canonical(_sanitize_dept_phrase(str(plan.get("dept_name") or "")))
    lookup_term = _sanitize_like_term(str(plan.get("lookup_term") or ""))
    task_name = str(plan.get("task_name") or "")
    task_detail = str(plan.get("task_detail") or "")

    LOG.debug(
        "qna_structured_db: llm plan intent=%r conf=%.2f person=%r dept=%r lookup=%r task=%r/%r",
        intent, conf, person_name, dept_name, lookup_term, task_name, task_detail
    )

    if intent == "person_lookup":
        if _should_skip_generic_person_ilike_lookup(raw):
            return None
        if len(person_name) >= 2 and len(dept_name) >= 2:
            out = fetch_person_by_name_and_dept(person_name, dept_name)
            return out or None
        term = lookup_term or extract_last_named_employee_lookup_term(raw)
        term = _sanitize_like_term(term)
        if len(term) < 2:
            return None
        rows = _blocking_run_coro(_fetch_rows_async(term), timeout_sec)
        if rows is None:
            return None
        return _person_lookup_chunks(term, rows, catalog_raw=raw)

    if intent == "dept_headcount" and len(dept_name) >= 2:
        out = _headcount_chunks_for_dept(dept_name, timeout_sec)
        return out or None

    if intent == "dept_roster" and len(dept_name) >= 2:
        out = _roster_chunks_for_dept(dept_name, timeout_sec)
        return out or None

    if intent == "task_owner":
        out = _task_owner_chunks_from_slots(raw, timeout_sec, task_name, task_detail)
        return out or None

    return None


def _generic_user_lookup_limit_clause() -> str:
    """`user` 테이블 넓은 ILIKE 조회의 LIMIT 절. 미설정·0 이하면 LIMIT 없음(전부 조회).

    과거 기본 LIMIT 20은 동명이인·겸직 등으로 같은 이름 행이 많을 때 목록이 잘리는 문제가 있었다.
    폭주 방지가 필요하면 QNA_STRUCTURED_DB_USER_LOOKUP_LIMIT 에 양의 정수만 설정한다.
    """
    raw = os.getenv("QNA_STRUCTURED_DB_USER_LOOKUP_LIMIT", "").strip()
    if not raw:
        return ""
    try:
        n = int(raw)
    except ValueError:
        return ""
    if n <= 0:
        return ""
    return f"\n            LIMIT {n}"


def _sanitize_like_term(s: str) -> str:
    return "".join(c for c in s.strip() if c.isalnum() or "\uac00" <= c <= "\ud7a3" or c in "._@-")[:128]


def extract_lookup_term(normalized_query: str) -> str:
    raw = normalize_structured_db_query(normalized_query)
    email_m = _EMAIL_RE.search(raw)
    if email_m:
        return _sanitize_like_term(email_m.group(0).lower())

    q = raw.lower()
    for w in sorted(_NOISE, key=len, reverse=True):
        q = q.replace(w.lower(), " ")
    q = " ".join(q.split())
    hangul = re.findall(r"[\uac00-\ud7a3]{2,40}", q)
    if hangul:
        # 부서명(canonical)은 마지막 순위 — 같은 길이일 때 인명이 선택되도록
        non_dept = [h for h in hangul if h not in _CANONICAL_DEPT_NAME_SET]
        pick = max(non_dept, key=len) if non_dept else max(hangul, key=len)
        pick = _strip_topic_in_after_org_token(pick)
        return _sanitize_like_term(_strip_trailing_josa(pick))
    token = re.search(r"[A-Za-z0-9._-]{3,}", q)
    if token:
        return token.group(0)
    cleaned = _sanitize_like_term(_strip_trailing_josa(q))
    return cleaned


_NAMED_EMPLOYEE_TITLE = re.compile(
    r"([\uac00-\ud7a3]{2,4})\s*(?:프로|님|씨|사원|대리|과장|차장|부장)",
    re.UNICODE,
)


def extract_last_named_employee_lookup_term(text: str) -> str:
    """한 문장에 여러 'OO 프로'가 있으면 마지막 인명을 user ILIKE 검색어로 쓴다."""
    raw = normalize_structured_db_query(text)
    hits = list(_NAMED_EMPLOYEE_TITLE.finditer(raw))
    if not hits:
        return extract_lookup_term(raw)
    return _sanitize_like_term(hits[-1].group(1))


def _row_to_chunk(row: dict[str, Any], idx: int) -> RetrievalChunk:
    content = (
        f"이름: {row.get('emp_nm') or ''}\n"
        f"사번: {row.get('emp_cd') or ''}\n"
        f"ID: {row.get('emp_id') or ''}\n"
        f"휴대폰: {row.get('emp_cphone') or ''}\n"
        f"이메일: {row.get('emp_email') or ''}\n"
        f"부서: {row.get('dept_nm') or ''}"
    )
    emp_id = str(row.get("emp_id") or "").strip()
    title = f"{row.get('emp_nm') or ''} ({row.get('dept_nm') or ''})".strip()
    return RetrievalChunk(
        chunk_id=f"db-user-{idx}",
        document_id=f"db:user:{emp_id or idx}",
        title=title or "사용자",
        content=content,
        page=1,
        score=1.0 - idx * 0.02,
        metadata={"source": _CHUNK_SOURCE, **{k: row.get(k) for k in ("emp_nm", "emp_cd", "emp_id", "emp_cphone", "emp_email", "dept_nm")}},
    )


def _roster_summary_chunk(rows: list[dict[str, Any]], dept_label: str) -> RetrievalChunk:
    n = len(rows)
    names = [(r.get("emp_nm") or "").strip() for r in rows if (r.get("emp_nm") or "").strip()]
    label = (dept_label or "").strip()
    intro = f"{label} 구성원 정보는 아래와 같습니다." if label.endswith("팀") else f"{label} 팀 구성원 정보는 아래와 같습니다."
    shown = sorted(names)[:120]
    bullet_lines = "\n".join(f"- {nm}" for nm in shown)
    if len(names) > 120:
        bullet_lines += f"\n- 외 {len(names) - 120}명"
    content = (
        f"{intro}\n\n"
        f"**총 인원: {n}명**\n\n"
        f"{bullet_lines}"
    )
    if len(content) > 12000:
        content = content[:11900] + "\n…(명단 일부 생략)"
    return RetrievalChunk(
        chunk_id="db-dept-roster-0",
        document_id=f"db:dept:{dept_label}",
        title=f"{dept_label} 소속 사원",
        content=content,
        page=1,
        score=1.0,
        metadata={
            "source": _CHUNK_SOURCE,
            "kind": _ROSTER_KIND,
            "dept_search": dept_label,
            "roster_count": n,
        },
    )


def _structured_guidance_chunk(kind: str, title: str, body: str, *, source: str | None = None) -> RetrievalChunk:
    src = source or _CHUNK_SOURCE
    return RetrievalChunk(
        chunk_id=f"db-guidance-{kind}",
        document_id=f"db:guidance:{kind}",
        title=title,
        content=body,
        page=1,
        score=1.0,
        metadata={"source": src, "structured_kind": kind},
    )


async def _fetch_distinct_dept_nm_async(key: str) -> list[str]:
    if not key:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        compact_pat = f"%{_dept_compact_key(key)}%"
        sql = f"""
            SELECT DISTINCT d.dept_nm
            FROM {_USER_TABLE} u
            LEFT JOIN {_DEPT_TABLE} d ON d.id = u.dept_cd
            WHERE {_sql_dept_compact("d.dept_nm")} ILIKE $1
              AND btrim(COALESCE(d.dept_nm, '')) <> ''
            ORDER BY 1
        """
        recs = await db.fetch(sql, compact_pat)
        return [str(r["dept_nm"]).strip() for r in recs if r.get("dept_nm")]
    finally:
        await db.close()


async def _fetch_person_name_dept_async(emp_nm: str, dept_hint: str) -> list[dict[str, Any]]:
    db = AsyncPostgresDB()
    await db.connect()
    try:
        sql = f"""
            SELECT u.emp_nm, u.emp_cd, u.emp_id, u.emp_cphone, u.emp_email, d.dept_nm
            FROM {_USER_TABLE} u
            LEFT JOIN {_DEPT_TABLE} d ON d.id = u.dept_cd
            WHERE lower(btrim(u.emp_nm)) = lower(btrim($1))
              AND {_sql_dept_compact("d.dept_nm")} ILIKE $2
            ORDER BY u.emp_nm NULLS LAST
            LIMIT 10
        """
        dept_pat = f"%{_dept_compact_key(dept_hint)}%"
        rows = await db.fetch(sql, emp_nm.strip(), dept_pat)
        return [dict(r) for r in rows]
    finally:
        await db.close()


def _task_owner_not_found_guidance(_raw: str) -> list[RetrievalChunk]:
    body = (
        "현재 검색 결과만으로는 정확한 답변을 확인하기 어렵습니다.\n\n"
        "정확한 안내가 필요하시면 피플팀에 문의해 주세요."
    )
    return [_structured_guidance_chunk(KIND_TASK_OWNER_NOT_FOUND, "규정 업무 담당", body, source=_MOBIGEN_SOURCE)]


def _person_lookup_chunks(
    term: str,
    rows: list[dict[str, Any]],
    *,
    catalog_raw: str | None = None,
) -> list[RetrievalChunk]:
    if not rows:
        if catalog_raw and is_regulation_task_owner_question(catalog_raw):
            return _task_owner_not_found_guidance(catalog_raw)
        body = (
            f"'{term}'에 해당하는 사원 정보를 찾지 못했습니다. "
            f"확인 후 다시 질문해 주세요"
        )
        return [_structured_guidance_chunk(KIND_PERSON_NOT_FOUND, "사원 조회", body)]

    term_l = term.strip().lower()
    exact = [r for r in rows if (r.get("emp_nm") or "").strip().lower() == term_l]
    if len(exact) >= 2:
        depts = sorted({(r.get("dept_nm") or "").strip() for r in exact if (r.get("dept_nm") or "").strip()})
        nm = (exact[0].get("emp_nm") or "").strip()
        if depts:
            dept_line = ", ".join(depts)
            body = (
                f"{nm} 프로의 담당 부서 정보는 아래와 같습니다.\n\n"
                f"- **담당부서 정보:** {dept_line}\n\n"
                f"담당자와 담당부서를 함께 질문을 입력해 주세요."
            )
        else:
            body = (
                f"'{nm}'(이)라는 이름의 프로가 여러 명 확인되었습니다. "
                f"사번 또는 부서명을 함께 입력해 주세요."
            )
        return [_structured_guidance_chunk(KIND_PERSON_AMBIGUOUS, "동명이인", body)]

    if len(exact) == 1:
        return [_row_to_chunk(exact[0], 0)]

    if len(rows) == 1:
        return [_row_to_chunk(rows[0], 0)]

    names = sorted({(r.get("emp_nm") or "").strip() for r in rows if (r.get("emp_nm") or "").strip()})
    preview = ", ".join(names[:8])
    if len(names) > 8:
        preview += f" 외 {len(names) - 8}명"
    body = (
        f"'{term}'(으)로 여러 사원이 검색되었습니다: {preview}. "
        f"이름·부서·사번 등을 더 구체적으로 입력해 주세요."
    )
    return [_structured_guidance_chunk(KIND_PERSON_MANY_NAMES, "사원 조회", body)]


def _vague_team_chunks(keyword: str, dept_names: list[str]) -> list[RetrievalChunk]:
    pg = _particle_wa_gwa(keyword)
    if len(dept_names) >= 2:
        bullet_lines = "\n".join(f"- {nm}" for nm in dept_names)
        body = (
            f"'{keyword}'{pg} 유사한 부서명은 아래와 같습니다.\n\n"
            f"{bullet_lines}\n\n"
            f"정확한 부서명을 입력해 주세요."
        )
        return [_structured_guidance_chunk(KIND_CLARIFY_DEPARTMENTS, "부서명 확인", body)]
    if len(dept_names) == 1:
        body = (
            f"'{keyword}'{pg} 가장 가까운 부서는 '{dept_names[0]}' 입니다. "
            f"이 이름으로 다시 질문하거나 더 구체적인 팀명을 적어 주세요."
        )
        return [_structured_guidance_chunk(KIND_CLARIFY_DEPARTMENTS, "부서명 확인", body)]
    body = (
        f"'{keyword}'{pg} 일치하는 부서명을 찾지 못했습니다. "
        f"다른 검색어로 질문해 주세요."
    )
    return [_structured_guidance_chunk(KIND_CLARIFY_DEPARTMENTS, "부서명 확인", body)]


def fetch_person_by_name_and_dept(emp_nm: str, dept_hint: str) -> list[RetrievalChunk]:
    if not structured_db_enabled():
        return []
    timeout_sec = _structured_db_timeout()
    rows = _blocking_run_coro(_fetch_person_name_dept_async(emp_nm, dept_hint), timeout_sec)
    if rows is None:
        return []
    if not rows:
        body = (
            f"{emp_nm} 프로를 아래 부서 정보로는 찾지 못했습니다.\n\n"
            f"- **이름:** {emp_nm}\n"
            f"- **부서:** {dept_hint}\n\n"
            f"팀명을 확인한 뒤 이름과 부서를 함께 질문해 주세요."
        )
        return [_structured_guidance_chunk(KIND_PERSON_NOT_FOUND, "사원 조회", body)]
    return [_row_to_chunk(rows[0], 0)]


async def _fetch_rows_async(like_term: str) -> list[dict[str, Any]]:
    if not like_term:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        pattern = f"%{like_term}%"
        sql = f"""
            SELECT u.emp_nm, u.emp_cd, u.emp_id, u.emp_cphone, u.emp_email, d.dept_nm
            FROM {_USER_TABLE} u
            LEFT JOIN {_DEPT_TABLE} d ON d.id = u.dept_cd
            WHERE u.emp_nm ILIKE $1 OR u.emp_id ILIKE $1 OR d.dept_nm ILIKE $1
               OR u.emp_email ILIKE $1 OR u.emp_cphone ILIKE $1
            ORDER BY u.emp_nm NULLS LAST{_generic_user_lookup_limit_clause()}
        """
        rows = await db.fetch(sql, pattern)
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def _fetch_department_members_async(dept_sub: str) -> list[dict[str, Any]]:
    if not dept_sub:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        pattern = f"%{_dept_compact_key(dept_sub)}%"
        sql = f"""
            SELECT u.emp_nm, u.emp_cd, u.emp_id, u.emp_cphone, u.emp_email, d.dept_nm
            FROM {_USER_TABLE} u
            LEFT JOIN {_DEPT_TABLE} d ON d.id = u.dept_cd
            WHERE {_sql_dept_compact("d.dept_nm")} ILIKE $1
            ORDER BY u.emp_nm NULLS LAST
            LIMIT 500
        """
        rows = await db.fetch(sql, pattern)
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def _count_department_members_async(dept_sub: str) -> int:
    if not dept_sub:
        return 0
    db = AsyncPostgresDB()
    await db.connect()
    try:
        pattern = f"%{_dept_compact_key(dept_sub)}%"
        sql = f"""
            SELECT COUNT(*)::bigint AS c
            FROM {_USER_TABLE} u
            LEFT JOIN {_DEPT_TABLE} d ON d.id = u.dept_cd
            WHERE {_sql_dept_compact("d.dept_nm")} ILIKE $1
        """
        row = await db.fetchrow(sql, pattern)
        return int(row["c"]) if row and row.get("c") is not None else 0
    finally:
        await db.close()


def _mobigen_row_key(r: dict[str, Any]) -> tuple[str, str, str, str, str]:
    def _ms(v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    return (
        _ms(r.get("task_id") or r.get("id")),
        _ms(r.get("task_name")),
        _ms(r.get("task_detail")),
        _ms(r.get("assignee")),
        _ms(r.get("department_name")),
    )


def _dedupe_mobigen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        k = _mobigen_row_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _task_label_in_question(label: str, raw: str) -> bool:
    """등록부 task_name / task_detail 문자열이 질문에 포함되는지(공백·전각 공백 무시)."""
    label = (label or "").strip()
    if len(label) < 2:
        return False
    if label in raw:
        return True
    compact_raw = raw.replace(" ", "").replace("\u3000", "")
    compact_lbl = label.replace(" ", "").replace("\u3000", "")
    return len(compact_lbl) >= 2 and compact_lbl in compact_raw


def _extract_mobigen_possessive_task_pair(raw: str) -> tuple[str, str] | None:
    """'업무명의 세부항목' 형태에서 task_name, task_detail 후보를 추출한다."""
    m = _MOBIGEN_POSSESSIVE_TASK.search(raw.strip())
    if not m:
        return None
    tn = (m.group("tn") or "").strip()
    td = (m.group("td") or "").strip()
    if len(tn) < 2 or len(td) < 2:
        return None
    return tn, td


def _narrow_mobigen_rows_task_name_then_detail(raw: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """task_detail이 질문에 있고 task_name도 질문에 있는 행만 남긴다. 없으면 None(기존 흐름 유지)."""
    narrowed: list[dict[str, Any]] = []
    for r in rows:
        tn = (r.get("task_name") or "").strip()
        td = (r.get("task_detail") or "").strip()
        if len(td) < 2 or not _task_label_in_question(td, raw):
            continue
        if len(tn) < 2 or not _task_label_in_question(tn, raw):
            continue
        narrowed.append(r)
    if not narrowed:
        return None
    return _dedupe_mobigen_rows(narrowed)


async def _fetch_mobigen_task_scoped_by_name_detail_async(task_name_sub: str, task_detail_sub: str) -> list[dict[str, Any]]:
    tn = _sanitize_like_term(task_name_sub)
    td = _sanitize_like_term(task_detail_sub)
    if len(tn) < 2 or len(td) < 2:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        sql = f"""
            SELECT mto.id::text AS task_id, mto.task_name, mto.task_detail, mto.assignee,
                   d.dept_nm AS department_name,
                   u.emp_nm AS assignee_nm
            {_MOBIGEN_FROM_JOIN}
            WHERE mto.task_name ILIKE $1 AND mto.task_detail ILIKE $2
            ORDER BY mto.task_name NULLS LAST, mto.task_detail NULLS LAST
            LIMIT 20
        """
        recs = await db.fetch(sql, f"%{tn}%", f"%{td}%")
        return [dict(r) for r in recs]
    finally:
        await db.close()


async def _fetch_mobigen_by_topic_id_async(topic_id: str) -> list[dict[str, Any]]:
    if not topic_id:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        sql = f"""
            SELECT mto.id::text AS task_id, mto.task_name, mto.task_detail, mto.assignee,
                   d.dept_nm AS department_name,
                   u.emp_nm AS assignee_nm
            {_MOBIGEN_FROM_JOIN}
            WHERE mto.topic_id = $1
            ORDER BY mto.task_name NULLS LAST, mto.task_detail NULLS LAST
        """
        recs = await db.fetch(sql, topic_id)
        return [dict(r) for r in recs]
    finally:
        await db.close()


async def _fetch_mobigen_task_rows_async(patterns: list[str]) -> list[dict[str, Any]]:
    if not patterns:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        parts: list[str] = []
        args: list[str] = []
        for i, p in enumerate(patterns):
            parts.append(f'(mto.task_name ILIKE ${i + 1} OR mto.task_detail ILIKE ${i + 1})')
            args.append(f"%{p.strip()}%")
        where_sql = " OR ".join(parts)
        sql = f"""
            SELECT mto.id::text AS task_id, mto.task_name, mto.task_detail, mto.assignee,
                   d.dept_nm AS department_name,
                   u.emp_nm AS assignee_nm
            {_MOBIGEN_FROM_JOIN}
            WHERE {where_sql}
            ORDER BY mto.task_name NULLS LAST, mto.task_detail NULLS LAST
            LIMIT 50
        """
        recs = await db.fetch(sql, *args)
        return [dict(r) for r in recs]
    finally:
        await db.close()


async def _fetch_distinct_task_details_for_task_name_async(task_name_sub: str) -> list[str]:
    if not task_name_sub:
        return []
    db = AsyncPostgresDB()
    await db.connect()
    try:
        pat = f"%{task_name_sub.strip()}%"
        sql = f"""
            SELECT DISTINCT task_detail
            FROM {_MOBIGEN_TASK_TABLE}
            WHERE task_name ILIKE $1 AND btrim(COALESCE(task_detail, '')) <> ''
            ORDER BY 1
            LIMIT 40
        """
        recs = await db.fetch(sql, pat)
        return [str(r["task_detail"]).strip() for r in recs if r.get("task_detail")]
    finally:
        await db.close()


def _mobigen_assignee_code_missing(row: dict[str, Any]) -> bool:
    v = row.get("assignee")
    if v is None:
        return True
    return not str(v).strip()


def _mobigen_task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("id") or "").strip()


def _mobigen_task_label(row: dict[str, Any]) -> str:
    tn = (row.get("task_name") or "").strip()
    td = (row.get("task_detail") or "").strip()
    if tn and td:
        return f"{tn}/{td}"
    return tn or td


async def _fetch_related_document_chunks_for_task_ids_async(
    task_ids: list[str],
    per_document_limit: int = 2,
    total_limit: int = 6,
) -> list[dict[str, Any]]:
    task_uuid_values: list[UUID] = []
    seen: set[UUID] = set()
    for task_id in task_ids:
        try:
            task_uuid = UUID(str(task_id))
        except (TypeError, ValueError):
            continue
        if task_uuid not in seen:
            task_uuid_values.append(task_uuid)
            seen.add(task_uuid)
    if not task_uuid_values:
        return []

    per_document_limit = max(1, min(per_document_limit, 3))
    total_limit = max(1, min(total_limit, 12))

    db = AsyncPostgresDB()
    await db.connect()
    try:
        sql = f"""
            WITH requested_tasks AS (
                SELECT task_id, ord
                FROM unnest($1::uuid[]) WITH ORDINALITY AS t(task_id, ord)
            )
            SELECT
                rt.task_id::text AS task_id,
                dtm.document_id::text AS document_id,
                d.title,
                d.document_type,
                d.effective_date,
                d.is_latest,
                d.source_file_path,
                d.source_url,
                dc.chunk_id::text AS chunk_id,
                COALESCE(dc.heading, d.title) AS heading,
                dc.content,
                COALESCE(dc.page_start, 1) AS page,
                dc.chunk_order
            FROM requested_tasks rt
            JOIN {_DOCUMENT_TASK_MAPPING_TABLE} dtm ON dtm.task_id = rt.task_id
            JOIN {_RAG_DOCS_TABLE} d ON d.document_id = dtm.document_id
            JOIN LATERAL (
                SELECT chunk_id, heading, content, page_start, chunk_order
                FROM {_RAG_CHUNKS_TABLE} dc
                WHERE dc.document_id = d.document_id
                  AND dc.document_status = 'completed'
                  AND dc.content IS NOT NULL
                ORDER BY dc.chunk_order ASC
                LIMIT $2
            ) dc ON true
            WHERE d.document_status = 'completed'
            ORDER BY rt.ord ASC, dtm.created_at ASC, d.title ASC, dc.chunk_order ASC
            LIMIT $3
        """
        recs = await db.fetch(sql, task_uuid_values, per_document_limit, total_limit)
        return [dict(r) for r in recs]
    finally:
        await db.close()


def _related_document_chunks_from_rows(
    rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> list[RetrievalChunk]:
    task_meta = {_mobigen_task_id(row): row for row in task_rows if _mobigen_task_id(row)}
    chunks: list[RetrievalChunk] = []
    seen_chunk_ids: set[str] = set()
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "").strip()
        document_id = str(row.get("document_id") or "").strip()
        content = str(row.get("content") or "").strip()
        if not chunk_id or not document_id or not content or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        reference_search_text = content

        task_id = str(row.get("task_id") or "").strip()
        task_row = task_meta.get(task_id, {})
        task_label = _mobigen_task_label(task_row)
        if task_label:
            content = f"관련 업무: {task_label}\n{content}"

        try:
            page = int(row.get("page") or 1)
        except (TypeError, ValueError):
            page = 1

        title = str(row.get("title") or row.get("heading") or "관련 문서").strip()
        chunks.append(
            RetrievalChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                content=content,
                page=page,
                score=max(0.70, 0.94 - len(chunks) * 0.02),
                metadata={
                    "source": _TASK_RELATED_DOC_SOURCE,
                    "structured_kind": KIND_TASK_RELATED_DOCUMENT,
                    "task_id": task_id,
                    "task_name": (task_row.get("task_name") or "").strip(),
                    "task_detail": (task_row.get("task_detail") or "").strip(),
                    "document_type": row.get("document_type"),
                    "effective_date": (
                        str(row["effective_date"]) if row.get("effective_date") else None
                    ),
                    "is_latest": row.get("is_latest"),
                    "source_file_path": row.get("source_file_path"),
                    "source_url": row.get("source_url"),
                    "reference_search_text": reference_search_text,
                    "related_via": "documents_task_mapping",
                },
            )
        )
    return chunks


def _fetch_task_related_document_chunks(
    task_rows: list[dict[str, Any]],
    timeout_sec: float | None,
) -> list[RetrievalChunk]:
    if timeout_sec is None:
        return []
    task_ids = [_mobigen_task_id(row) for row in task_rows]
    task_ids = [task_id for task_id in task_ids if task_id]
    if not task_ids:
        return []
    rows = _blocking_run_coro(
        _fetch_related_document_chunks_for_task_ids_async(task_ids),
        timeout_sec,
    )
    if not rows:
        return []
    return _related_document_chunks_from_rows(rows, task_rows)


def _mobigen_answer_body(
    row: dict[str, Any],
    wants_assignee: bool,
    wants_dept: bool,
) -> str:
    assignee_nm = (row.get("assignee_nm") or "").strip()
    assignee_raw = str(row.get("assignee") or "").strip()
    assignee_label = assignee_nm or (assignee_raw if not assignee_raw.isdigit() else "")
    dept = (row.get("department_name") or "").strip()
    parts: list[str] = []
    if wants_dept:
        if dept:
            parts.append(f"담당 부서는 {dept}입니다.")
        else:
            parts.append(f"담당 부서 정보는 등록되어 있지 않아 {_PEOPLE_TEAM_REFER}.")
    if wants_assignee:
        if _mobigen_assignee_code_missing(row):
            parts.append(f"담당자가 존재하지 않습니다. {_PEOPLE_TEAM_REFER}.")
        elif assignee_label:
            parts.append(f"담당자는 {assignee_label} 프로입니다.")
        else:
            parts.append(f"담당자 정보는 등록되어 있지 않아 {_PEOPLE_TEAM_REFER}.")
    if not parts:
        parts.append(_PEOPLE_TEAM_REFER)
    return " ".join(parts).strip()


def _mobigen_chunks_from_rows(
    raw: str,
    rows: list[dict[str, Any]],
    timeout_sec: float | None = None,
) -> list[RetrievalChunk]:
    rows = _dedupe_mobigen_rows(rows)
    wants_a, wants_d = _task_owner_answer_preferences(raw)
    if not rows:
        body = (
            "질문과 일치하는 사내 규정 업무(담당) 정보를 찾지 못했습니다. "
            "업무명·세부 항목을 더 구체적으로 입력해 주세요."
        )
        return [_structured_guidance_chunk(KIND_TASK_OWNER_NOT_FOUND, "규정 업무 담당", body, source=_MOBIGEN_SOURCE)]

    and_narrowed = _narrow_mobigen_rows_task_name_then_detail(raw, rows)
    if and_narrowed is not None:
        rows = and_narrowed
    else:
        narrowed = [
            r
            for r in rows
            if (r.get("task_detail") and str(r["task_detail"]).strip() in raw)
            or (
                r.get("task_name")
                and len(str(r["task_name"]).strip()) >= 2
                and str(r["task_name"]).strip() in raw
            )
        ]
        if len(narrowed) == 1:
            rows = narrowed
        elif len(narrowed) > 1:
            rows = _dedupe_mobigen_rows(narrowed)

    if len(rows) == 1:
        body = _mobigen_answer_body(rows[0], wants_a, wants_d)
        owner_chunk = _structured_guidance_chunk(
            KIND_TASK_OWNER, "규정 업무 담당", body, source=_MOBIGEN_SOURCE
        )
        related_docs = _fetch_task_related_document_chunks(rows, timeout_sec)
        return [owner_chunk, *related_docs]

    task_names = sorted({(r.get("task_name") or "").strip() for r in rows if (r.get("task_name") or "").strip()})
    if len(task_names) == 1:
        tn = task_names[0]
        details = sorted(
            {(r.get("task_detail") or "").strip() for r in rows if (r.get("task_detail") or "").strip()}
        )
        user_named_detail = any(len(d) >= 2 and d in raw for d in details)
        if len(details) >= 2 and not user_named_detail:
            shown = ", ".join(details[:20])
            if len(details) > 20:
                shown += f" 외 {len(details) - 20}개"
            body = (
                f"'{tn}'에 대한 사내 규정 업무는 {shown} 등으로 나뉘어 있습니다. "
                f"명확한 질문을 해 주세요."
            )
            return [_structured_guidance_chunk(KIND_TASK_OWNER_CLARIFY, "규정 업무 담당", body, source=_MOBIGEN_SOURCE)]

    pairs: list[str] = []
    for r in rows[:12]:
        tn = (r.get("task_name") or "").strip()
        td = (r.get("task_detail") or "").strip()
        if tn and td:
            pairs.append(f"{tn}/{td}")
        elif tn:
            pairs.append(tn)
        elif td:
            pairs.append(td)
    prev = ", ".join(pairs)
    if len(rows) > 12:
        prev += f" 외 {len(rows) - 12}건"
    body = (
        f"여러 업무가 검색되었습니다: {prev}. "
        f"업무명·세부 항목을 더 구체적으로 입력해 주세요."
    )
    return [_structured_guidance_chunk(KIND_TASK_OWNER_CLARIFY, "규정 업무 담당", body, source=_MOBIGEN_SOURCE)]


def fetch_mobigen_task_owner_chunks(normalized_query: str, timeout_sec: float) -> list[RetrievalChunk]:
    raw = normalize_structured_db_query(normalized_query)
    if not is_regulation_task_owner_question(raw):
        return []
    possessive = _extract_mobigen_possessive_task_pair(raw)
    if possessive:
        tn0, td0 = possessive
        scoped = _blocking_run_coro(
            _fetch_mobigen_task_scoped_by_name_detail_async(tn0, td0),
            timeout_sec,
        )
        if scoped:
            return _mobigen_chunks_from_rows(raw, scoped, timeout_sec)

    # topic_mapping id → mobigen_task_owner.topic_id 직접 조회 (우선 경로)
    try:
        from services.qna_topic_mapper import resolve_topic, topic_to_search_tokens, topic_to_task_parts
        topic_row = resolve_topic(raw, timeout_sec=min(timeout_sec, 2.0))
        if topic_row:
            topic_id = str(topic_row.get("id") or "").strip()
            if topic_id:
                topic_id_rows = _blocking_run_coro(
                    _fetch_mobigen_by_topic_id_async(topic_id), timeout_sec
                )
                if topic_id_rows:
                    LOG.debug(
                        "qna_structured_db: topic_id direct hit (id=%r, rows=%d)",
                        topic_id, len(topic_id_rows),
                    )
                    return _mobigen_chunks_from_rows(raw, topic_id_rows, timeout_sec)
                LOG.debug("qna_structured_db: topic_id=%r matched no rows, falling back to scoped ILIKE", topic_id)

            # topic_id 조회 실패 → topic 텍스트 파싱으로 AND ILIKE (task_name + task_detail)
            topic_text = (topic_row.get("topic") or "").strip()
            if topic_text:
                task_name, task_detail = topic_to_task_parts(topic_text)
                if task_detail:
                    scoped = _blocking_run_coro(
                        _fetch_mobigen_task_scoped_by_name_detail_async(task_name, task_detail),
                        timeout_sec,
                    )
                    if scoped:
                        LOG.debug(
                            "qna_structured_db: scoped ILIKE hit (task_name=%r, task_detail=%r, rows=%d)",
                            task_name, task_detail, len(scoped),
                        )
                        return _mobigen_chunks_from_rows(raw, scoped, timeout_sec)
                    LOG.debug(
                        "qna_structured_db: scoped ILIKE no rows (task_name=%r, task_detail=%r), falling back to broad ILIKE",
                        task_name, task_detail,
                    )

            topic_patterns = topic_to_search_tokens(topic_text)
        else:
            topic_patterns = []
    except Exception as exc:
        LOG.debug("qna_structured_db: topic_mapper failed (%s)", exc)
        topic_patterns = []

    # fallback: ILIKE 패턴 검색
    patterns = _extract_task_owner_search_patterns(raw)
    if not patterns and ("근무" in raw or "휴가" in raw or "법인카드" in raw):
        for w in ("법인카드", "휴가", "근무", "복리후생", "영업지원", "예산", "경비", "우리사주", "국책", "구매"):
            if w in raw:
                patterns = [w]
                break

    if topic_patterns:
        # 질문에서 나온 patterns(휴가·법인카드 등)를 topic_mapping 토큰보다 앞에 둔다.
        # topic_patterns만 앞세우면 merged[:3]에 질문 키워드가 밀려 mobigen ILIKE가 빈 결과가 되고,
        # 이후 user 사원 조회로 떨어져 '휴가' 같은 오탐 답이 나올 수 있다.
        merged: list[str] = []
        seen: set[str] = set()
        for p in patterns + topic_patterns:
            if p not in seen:
                merged.append(p)
                seen.add(p)
        patterns = merged[:3]

    # 동의어 확장: "동아리" → "동호회" 등 DB 표기와 불일치 방지
    _syn_extra: list[str] = []
    _syn_seen: set[str] = set(patterns)
    for _p in patterns:
        _syn = _TASK_KNOWLEDGE_SYNONYMS.get(_p)
        if _syn and _syn not in _syn_seen:
            _syn_extra.append(_syn)
            _syn_seen.add(_syn)
    if _syn_extra:
        patterns = (patterns + _syn_extra)[:4]

    if not patterns:
        return []
    rows = _blocking_run_coro(_fetch_mobigen_task_rows_async(patterns), timeout_sec)
    if rows is None:
        return []

    task_names_hit = sorted({(r.get("task_name") or "").strip() for r in rows if (r.get("task_name") or "").strip()})
    if len(task_names_hit) == 1 and patterns:
        primary = patterns[0]
        if primary in task_names_hit[0] or task_names_hit[0] in primary:
            details = _blocking_run_coro(
                _fetch_distinct_task_details_for_task_name_async(task_names_hit[0]),
                timeout_sec,
            )
            if details is None:
                # 세부 목록 조회 실패 시 빈 결과로 두고 아래 _mobigen_chunks_from_rows 로 진행한다.
                # return [] 하면 일반 사원 ILIKE 로 떨어져 오탐 메시지가 나온다.
                details = []
            user_named_detail = any(len(d) >= 2 and d in raw for d in details)
            if len(details) >= 2 and not user_named_detail:
                shown = ", ".join(details[:20])
                if len(details) > 20:
                    shown += f" 외 {len(details) - 20}개"
                body = (
                    f"'{task_names_hit[0]}'에 대한 사내 규정은 {shown} 등이 있습니다. "
                    f"명확한 질문을 해 주세요."
                )
                return [_structured_guidance_chunk(KIND_TASK_OWNER_CLARIFY, "규정 업무 담당", body, source=_MOBIGEN_SOURCE)]

    return _mobigen_chunks_from_rows(raw, rows, timeout_sec)


def _blocking_run_coro(coro, timeout_sec: float) -> Any:
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=timeout_sec)
    except Exception as e:
        LOG.warning("qna_structured_db: lookup failed: %s", e)
        return None


def fetch_user_lookup_chunks(normalized_query: str) -> list[RetrievalChunk]:
    if not structured_db_enabled():
        return []
    raw = normalize_structured_db_query(normalized_query)
    timeout_sec = _structured_db_timeout()

    llm_chunks = _chunks_from_llm_plan(raw, timeout_sec)
    if llm_chunks is not None:
        return llm_chunks

    vague_key = extract_vague_team_keyword(raw)
    if vague_key:
        dept_names = _blocking_run_coro(_fetch_distinct_dept_nm_async(vague_key), timeout_sec)
        if dept_names is None:
            return []
        return _vague_team_chunks(vague_key, dept_names)

    dept_person = try_extract_known_dept_person_pair(raw)
    if dept_person:
        emp_nm, dept_hint = dept_person
        return fetch_person_by_name_and_dept(emp_nm, dept_hint)

    head_dept = extract_department_headcount_phrase(raw)
    if head_dept:
        return _headcount_chunks_for_dept(head_dept, timeout_sec)

    dept_roster = extract_department_roster_phrase(raw)
    if dept_roster:
        return _roster_chunks_for_dept(dept_roster, timeout_sec)

    mob = fetch_mobigen_task_owner_chunks(raw, timeout_sec)
    if mob:
        return mob

    if is_regulation_task_owner_question(raw):
        return _task_owner_not_found_guidance(raw)

    if _should_skip_generic_person_ilike_lookup(raw):
        return []

    term = extract_lookup_term(raw)
    term = _sanitize_like_term(term)
    if len(term) < 2:
        return []

    rows = _blocking_run_coro(_fetch_rows_async(term), timeout_sec)
    if rows is None:
        return []

    if not rows:
        LOG.info("qna_structured_db: no rows matched (term=%r, table=%s)", term, _USER_TABLE)

    return _person_lookup_chunks(term, rows, catalog_raw=raw)
