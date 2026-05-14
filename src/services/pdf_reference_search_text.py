# PDF 텍스트 레이어에서 청크와 대응하는 구간을 잘라 reference_search_text 후보를 만든다.
# 청킹·적재(ETL) 배치에서 호출해 DB document_chunks.reference_search_text 등에 저장하면 된다.
from __future__ import annotations

import difflib
import io
from typing import BinaryIO

from utils.logger import LOG

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[misc, assignment]


def extract_text_layer_page(pdf: bytes | BinaryIO, *, page_one_based: int) -> str:
    """PDF 한 페이지(1-based)의 텍스트 레이어를 이어붙인 문자열로 반환한다."""
    if PdfReader is None:
        raise RuntimeError("pypdf 패키지가 필요합니다. requirements.txt의 pypdf를 설치하세요.")

    bio = io.BytesIO(pdf) if isinstance(pdf, (bytes, bytearray)) else pdf
    bio.seek(0)
    reader = PdfReader(bio)
    n = len(reader.pages)
    if n < 1:
        return ""
    p = max(1, min(int(page_one_based), n))
    try:
        return reader.pages[p - 1].extract_text() or ""
    except Exception as e:
        LOG.warning("pdf_reference_search_text: extract_text failed page=%s: %s", p, e)
        return ""


def _longest_common_substring(a: str, b: str, *, min_size: int) -> tuple[int, int] | None:
    """page_text[a0:a1] 구간이 chunk와 최소 min_size 글자로 정확히 일치."""
    if not a or not b:
        return None
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    if m.size < min_size:
        return None
    return (m.a, m.a + m.size)


def _span_via_whitespace_bridge(page_text: str, chunk_text: str, *, max_chars: int, min_run: int) -> str | None:
    """공백만 정규화한 공통 부분을 pc에서 찾은 뒤, page_text에서 첫·끝 토큰으로 원문 구간을 잡는다."""
    pc = " ".join(page_text.split())
    cc = " ".join(chunk_text.split()).strip()[:max_chars]
    if len(cc) < min_run:
        return None
    sm = difflib.SequenceMatcher(None, pc, cc, autojunk=False)
    m = sm.find_longest_match(0, len(pc), 0, len(cc))
    if m.size < min_run:
        return None
    bridge = pc[m.a : m.a + m.size].strip()
    words = bridge.split()
    if len(words) < 1:
        return None
    first, last = words[0], words[-1]
    i0 = page_text.find(first)
    if i0 == -1:
        return None
    tail = page_text[i0:]
    j = tail.find(last)
    if j == -1:
        return page_text[i0 : i0 + len(first)][:max_chars]
    j_abs = i0 + j + len(last)
    return page_text[i0:j_abs][:max_chars]


def reference_search_text_from_pdf(
    *,
    chunk_text: str,
    pdf_bytes: bytes,
    page_one_based: int,
    max_chars: int = 500,
) -> str:
    """청크 본문과 같은 PDF 페이지 텍스트 레이어에서 실제로 존재하는 스팬을 잘라 반환한다.

    1) chunk 접두가 page_text의 부분 문자열이면 그대로 사용
    2) 공백 정규화 브리지로 page 원문 구간 추출
    3) difflib 최장 공통부 (page 쪽 원문)
    실패 시 chunk 앞부분 폴백.
    """
    page_text = extract_text_layer_page(pdf_bytes, page_one_based=page_one_based)
    if not page_text.strip():
        return (chunk_text or "").strip()[:max_chars]

    ct = (chunk_text or "").strip()
    if not ct:
        return page_text.strip()[:max_chars]

    head = ct[:max_chars]
    if head in page_text:
        i = page_text.index(head)
        return page_text[i : i + len(head)]

    min_run = max(12, min(40, len(ct) // 4))
    bridged = _span_via_whitespace_bridge(page_text, ct, max_chars=max_chars, min_run=min_run)
    if bridged:
        return bridged[:max_chars]

    span = _longest_common_substring(page_text, ct[:max_chars], min_size=min_run)
    if span:
        a0, a1 = span
        return page_text[a0:a1][:max_chars]

    return head


def reference_search_text_for_chunk_row(
    *,
    chunk_content: str,
    pdf_bytes: bytes,
    page_start: int,
    max_chars: int = 500,
) -> str:
    """DB 적재용: document_chunks.content + page_start + PDF 바이트 → reference_search_text."""
    return reference_search_text_from_pdf(
        chunk_text=chunk_content,
        pdf_bytes=pdf_bytes,
        page_one_based=page_start,
        max_chars=max_chars,
    )
