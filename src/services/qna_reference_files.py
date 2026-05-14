# QnA 응답에 MinIO 원본 PDF 참고 링크(type=file)를 붙이기 위한 모듈.
# 스펙: ChatMessage(role=assistant, content=str([{title,url}]), additional_kwargs.type=file) — 문서 1건 1링크
# Graphio 참고 문서 표기: ``/{C_MINIO_CONTEXT_PATH}/{bucket}/{*.pdf}#phrase=true&page=…&search=…`` 만 사용한다.
# (presign ``?X-Amz-…`` 쿼리는 EventStream·클라이언트로 넘기지 않음 — 앱 ``/download/…`` 프록시가 권한 처리.)
# MinIO 객체 키를 못 쓰면 metadata.source_url만 type=file에 넣는다.
# 그 값은 DB friday.documents.source_url 이 검색 청크 조인 시 metadata에 실린 것과 동일하다.
#
# 튜닝은 환경변수가 아니라 QnaRuntimePolicy(qna_config.load_runtime_policy) 기본값만 수정한다.
# 버킷은 config.minio_bucket_name(C_MINIO_BUCKET_NAME), 비면 "friday".
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
import unicodedata
from urllib.parse import quote, urlparse, urlunparse

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage as LangchainChatMessage
from minio import Minio
from minio.error import S3Error

from core.config import config
from services.qna_config import QnaRuntimePolicy, load_runtime_policy
from services.qna_types import RetrievalChunk
from utils.logger import LOG
from utils.minio_utils import create_client


def _minio_bucket_for_references() -> str:
    cfg = (getattr(config, "minio_bucket_name", None) or "").strip()
    return cfg or "friday"


def _nfc_object_key(path: str) -> str:
    """MinIO 객체 키는 UTF-8 바이트가 정확히 일치해야 한다. DB에 자모·분해형이 섞이면 버킷(NFC 완성형)과 불일치할 수 있어 NFC로 맞춘다."""
    return unicodedata.normalize("NFC", path)


def _strip_bucket_prefix(object_path: str, bucket: str) -> str:
    """'{bucket}/object/key.pdf' → 'object/key.pdf'. 접두가 bucket과 다르면 전체 경로를 object key로 본다."""
    s = object_path.strip().lstrip("/")
    prefix = f"{bucket.strip()}/"
    if s.lower().startswith(prefix.lower()):
        return s[len(prefix) :].lstrip("/")
    return s


def _object_name_candidates(meta: dict[str, Any], bucket: str, strip_prefix: bool) -> list[str]:
    """DB source_file_path에서 MinIO 객체 키 후보(원문 스트립 · NFC · NFD)를 만든다."""
    raw = (meta.get("source_file_path") or "").strip()
    if not raw:
        return []
    if strip_prefix:
        stripped = _strip_bucket_prefix(raw, bucket)
    else:
        stripped = raw.lstrip("/")
    if not stripped:
        return []
    nfc = _nfc_object_key(stripped)
    nfd = unicodedata.normalize("NFD", nfc)
    out: list[str] = []
    for k in (stripped, nfc, nfd):
        if k not in out:
            out.append(k)
    return out


def _stat_first_existing_key(client: Minio, bucket: str, candidates: list[str]) -> str | None:
    for name in candidates:
        try:
            client.stat_object(bucket, name)
            return name
        except S3Error as e:
            code = getattr(e, "code", "") or ""
            if code in ("NoSuchKey", "NoSuchBucket"):
                continue
            LOG.warning(
                "qna_reference_files: stat_object failed bucket=%r object=%r: %s",
                bucket,
                name,
                e,
            )
            continue
        except Exception as e:
            LOG.warning(
                "qna_reference_files: stat_object unexpected failure bucket=%r object=%r: %s",
                bucket,
                name,
                e,
            )
            continue
    return None


def _fallback_public_style_path(bucket: str, object_name: str) -> str:
    """presign 없이 기존 `/download/{bucket}/{object}` 스타일 경로를 가리킨다."""
    ctx = _download_proxy_context_path()
    safe_obj = "/".join(quote(seg, safe="") for seg in object_name.split("/") if seg is not None)
    return f"/{ctx}/{bucket}/{safe_obj}"


def _download_proxy_context_path() -> str:
    """Graphio MinIO 다운로드 프록시 URL에 쓰이는 context path (`config.minio_context_path`)."""
    return (config.minio_context_path or "download").strip().strip("/")


def _strip_s3_style_presign_query(url: str) -> str:
    """S3/MinIO presign 쿼리(X-Amz-*)는 스펙상 type=file URL에 포함하지 않는다."""
    if "?" not in url:
        return url
    p = urlparse(url)
    q = (p.query or "").lower()
    if "x-amz-algorithm" not in q and "x-amz-signature" not in q:
        return url
    return urlunparse((p.scheme, p.netloc, p.path, "", "", p.fragment))


def _fragment_for_page_search(page: int, snippet: str, max_chars: int) -> str:
    """Graphio 참고 문서 표기 예시와 동일: `#phrase=true&page=…&search=…` (`search`는 percent-encoding).

    페이지·하이라이트는 Graphio 앱 PDF 뷰어(또는 동일 규약 뷰어)가 fragment를 해석한다.
    """
    sn = " ".join((snippet or "").split())[:max_chars]
    enc = quote(sn, safe="")
    return f"#phrase=true&page={int(page)}&search={enc}"


def _display_filename(object_name: str) -> str:
    return PurePosixPath(object_name).name or object_name


def _chunk_page(chunk: RetrievalChunk) -> int:
    try:
        return int(chunk.page or 1)
    except (TypeError, ValueError):
        return 1


def _is_pdf_object(object_name: str) -> bool:
    return PurePosixPath(object_name).suffix.lower() == ".pdf"


def _source_url_link(chunk: RetrievalChunk, page: int) -> dict[str, str] | None:
    su = str((chunk.metadata or {}).get("source_url") or "").strip()
    if not su:
        return None
    doc_title = (chunk.title or "").strip() or "사내 규정"
    return {"title": f"(Page {page}) {doc_title}", "url": _strip_s3_style_presign_query(su)}


def _reference_search_text(chunk: RetrievalChunk) -> str:
    meta = chunk.metadata or {}
    raw = meta.get("reference_search_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return (chunk.content or "").strip()


def _reference_document_key(c: RetrievalChunk) -> str:
    did = (c.document_id or "").strip()
    if did:
        return did
    return (c.chunk_id or "").strip() or "__unknown_chunk__"


def _group_chunks_by_document(chunks: list[RetrievalChunk]) -> dict[str, list[RetrievalChunk]]:
    by_doc: dict[str, list[RetrievalChunk]] = defaultdict(list)
    for c in chunks:
        by_doc[_reference_document_key(c)].append(c)
    return by_doc


@dataclass
class _ReferenceLinkSession:
    """collect 한 번에 MinIO 클라이언트·객체 키 resolve 결과를 재사용한다."""

    policy: QnaRuntimePolicy
    bucket: str
    max_snip: int
    _key_cache: dict[tuple[str, tuple[str, ...]], str | None] = field(default_factory=dict)
    _minio: Minio | None = field(default=None, init=False)

    def _minio_client(self) -> Minio:
        if self._minio is None:
            self._minio = create_client()
        return self._minio

    def _resolved_pdf_object_key(self, meta: dict[str, Any]) -> str | None:
        candidates = _object_name_candidates(
            meta, self.bucket, self.policy.reference_strip_bucket_prefix
        )
        if not candidates:
            return None
        cache_key = (self.bucket, tuple(candidates))
        if cache_key in self._key_cache:
            return self._key_cache[cache_key]

        if self.policy.reference_presign_disabled:
            resolved = candidates[0]
        else:
            try:
                resolved = _stat_first_existing_key(
                    self._minio_client(), self.bucket, candidates
                )
            except Exception as e:
                LOG.warning(
                    "qna_reference_files: MinIO resolve failed bucket=%r candidates=%r: %s",
                    self.bucket,
                    candidates,
                    e,
                )
                resolved = None
            if not resolved:
                LOG.warning(
                    "qna_reference_files: MinIO에 객체 없음 bucket=%r tried=%r — "
                    "source_file_path·버킷 키·source_url을 확인하세요.",
                    self.bucket,
                    candidates,
                )
        self._key_cache[cache_key] = resolved
        return resolved

    def link_for_chunk(self, c: RetrievalChunk) -> dict[str, str] | None:
        meta = c.metadata or {}
        page = _chunk_page(c)
        key = self._resolved_pdf_object_key(meta)
        if key and _is_pdf_object(key):
            base = _fallback_public_style_path(self.bucket, key)
            snip = _reference_search_text(c)
            url = f"{base}{_fragment_for_page_search(page, snip, self.max_snip)}"
            return {"title": f"(Page {page}) {_display_filename(key)}", "url": url}
        return _source_url_link(c, page)


def _normalize_answer_for_reference_gate(text: str) -> str:
    t = (text or "").replace("\u200b", "").strip()
    if not t:
        return ""
    return " ".join(t.split())


def _answer_suggests_no_regulation_hit(text: str) -> bool:
    """답변이 '제공 문서·규정에 해당 내용 없음'처럼 인용 대상이 없음을 말하면 True."""
    t = _normalize_answer_for_reference_gate(text)
    if not t:
        return True
    doc_ctx = (
        "제공된 문서",
        "제공한 문서",
        "주어진 문서",
        "제시된 문서",
        "참고 정보",
        "참고문서",
    )
    if any(dc in t for dc in doc_ctx):
        if any(
            neg in t
            for neg in (
                "포함되어 있지",
                "포함되지 않",
                "해당 내용이 없",
                "해당 사항이 없",
                "관한 내용은 없",
                "에 대한 내용은 없",
                "명시되어 있지 않",
                "규정하지 않",
            )
        ):
            return True
    if any(
        phrase in t
        for phrase in (
            "문서에서 확인할 수 없습니다",
            "문서에서 확인할 수 없음",
            "참고 문서에서 확인할 수 없",
            "규정 문서에서 찾을 수 없",
            "검색된 문서에서 찾을 수 없",
        )
    ):
        return True
    if "찾을 수 없습니다" in t and any(
        h in t for h in ("문서", "규정", "제공된", "검색 결과", "관련 규정")
    ):
        return True
    if "포함되어 있지 않" in t and any(
        h in t for h in ("질문하신", "질문에", "요청하신", "요청하신 내용", "귀하의 질문")
    ):
        return True
    if t.startswith("죄송") and any(
        p in t for p in ("알 수 없", "답변드릴 수 없", "안내드릴 수 없")
    ):
        return True
    return False


def answer_supports_regulation_references(answer: str) -> bool:
    """규정·참고 PDF 링크를 붙여도 될 만한 답변인지(빈 답·미포함 안내 문장이면 False)."""
    if not _normalize_answer_for_reference_gate(answer):
        return False
    return not _answer_suggests_no_regulation_hit(answer)


def _best_link_for_document(
    doc_chunks: list[RetrievalChunk], session: _ReferenceLinkSession
) -> tuple[float, dict[str, str]] | None:
    """문서 내 최고 score와, 그 문서에서 만들 수 있는 가장 점수 높은 청크의 링크. 링크 불가면 None."""
    top_score = max(c.score for c in doc_chunks)
    for c in sorted(doc_chunks, key=lambda ch: ch.score, reverse=True):
        link = session.link_for_chunk(c)
        if link is not None:
            return (top_score, link)
    return None


def collect_reference_file_links(chunks: list[RetrievalChunk]) -> list[dict[str, str]]:
    """검색 청크 중 `score` 대표값이 가장 큰 문서 1개에 대해 참고 링크 1개만 반환한다."""
    policy = load_runtime_policy()
    if not policy.reference_files_enabled or not chunks:
        return []

    max_snip = max(8, min(policy.reference_search_snippet_chars, 500))
    session = _ReferenceLinkSession(
        policy=policy,
        bucket=_minio_bucket_for_references(),
        max_snip=max_snip,
    )

    # (문서 대표 score, document_key, link) — 대표 score로 우승 문서를 고른다.
    rows: list[tuple[float, str, dict[str, str]]] = []
    for doc_key, doc_chunks in _group_chunks_by_document(chunks).items():
        picked = _best_link_for_document(doc_chunks, session)
        if picked is not None:
            score, link = picked
            rows.append((score, doc_key, link))

    if not rows:
        return []
    _, _, best = max(rows, key=lambda r: (r[0], r[1]))
    return [best]


def build_answer_messages_with_reference_files(
    answer: str,
    chunks: list[RetrievalChunk],
    *,
    include_references: bool = True,
) -> tuple[list[BaseMessage], dict[str, Any] | None]:
    """AIMessage + (있으면) type=file ChatMessage, 그리고 API용 직렬화 dict."""
    if not include_references or not answer_supports_regulation_references(answer):
        return [AIMessage(content=answer)], None

    links = collect_reference_file_links(chunks)
    if not links:
        return [AIMessage(content=answer)], None

    msgs: list[BaseMessage] = [AIMessage(content=answer)]

    content_str = str(links)
    file_lc = LangchainChatMessage(
        content=content_str,
        role="assistant",
        additional_kwargs={"type": "file"},
    )
    msgs.append(file_lc)
    serializable = {
        "role": "assistant",
        "content": content_str,
        "additional_kwargs": {"type": "file"},
    }
    return msgs, serializable


def merge_reference_file_into_response(
    response: dict[str, Any],
    chunks: list[RetrievalChunk],
    *,
    include_references: bool = True,
) -> list[BaseMessage]:
    """response dict에 reference_file_message를 넣고, LangGraph용 messages 리스트를 반환한다."""
    answer = str(response.get("answer") or "")
    msgs, ref = build_answer_messages_with_reference_files(
        answer, chunks, include_references=include_references
    )
    if msgs and isinstance(msgs[0], AIMessage):
        response["answer"] = str(msgs[0].content)
    if ref is not None:
        response["reference_file_message"] = ref
    return msgs
