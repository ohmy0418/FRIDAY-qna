# 서비스 전반에서 공유하는 공통 유틸 모듈.
# 이미 실행 중인 이벤트 루프에서 코루틴을 안전하게 실행하는 run_async(),
# OpenAI API로 텍스트 임베딩을 생성하는 _generate_embedding(),
# LLM JSON 출력에서 마크다운 코드 펜스를 제거하는 _strip_json_fences()를 제공한다.
from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from utils.logger import LOG


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def run_async(coro: Any, timeout_sec: float) -> Any:
    """코루틴을 별도 스레드의 새 이벤트 루프에서 실행한다.

    이미 실행 중인 루프가 있는 컨텍스트(FastAPI 등)에서 호출해도 안전하다.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout_sec)


def _generate_embedding(text: str, log_prefix: str = "qna") -> list[float] | None:
    try:
        from openai import OpenAI
        api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
        if not api_key:
            return None
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        resp = OpenAI(api_key=api_key).embeddings.create(model=model, input=text)
        return resp.data[0].embedding
    except Exception as exc:
        LOG.debug("%s: embedding failed (%s)", log_prefix, exc)
        return None
