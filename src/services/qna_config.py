# 환경변수를 읽어 QnaRuntimePolicy(LLM 모델·타임아웃·로그 저장소 등 런타임 정책)를 로드한다.
# 모든 서비스가 load_runtime_policy()를 호출해 실행 정책을 참조한다.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProviderMode = Literal["direct", "gateway"]
TimeoutAction = Literal["fallback", "error"]
ForcedLlmFailure = Literal["none", "timeout", "error"]
LogStore = Literal["memory", "jsonl"]

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in allowed else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


@dataclass(frozen=True)
class QnaRuntimePolicy:
    llm_provider_mode: ProviderMode
    llm_model: str
    llm_api_address: str | None
    llm_timeout_seconds: int
    llm_max_retries: int
    timeout_action: TimeoutAction
    force_llm_failure: ForcedLlmFailure
    log_store: LogStore
    log_dir: Path
    route_retry_enabled: bool
    # MinIO 참고 PDF(type=file) — 값은 코드 기본만 사용(환경변수 분산 없음). 필요 시 이 dataclass만 수정.
    reference_files_enabled: bool
    reference_strip_bucket_prefix: bool
    reference_search_snippet_chars: int
    reference_presign_disabled: bool
    reference_presign_fallback_on_error: bool


def load_runtime_policy() -> QnaRuntimePolicy:
    llm_model = (os.getenv("LLM_MODEL") or "google/gemma-3-27b-it").strip()
    llm_api_address = (os.getenv("LLM_API_ADDRESS") or "").strip() or None
    return QnaRuntimePolicy(
        llm_provider_mode=_env_choice("QNA_LLM_PROVIDER_MODE", "direct", {"direct", "gateway"}),
        llm_model=llm_model,
        llm_api_address=llm_api_address,
        llm_timeout_seconds=_env_int("QNA_LLM_TIMEOUT_SECONDS", 15, minimum=1),
        llm_max_retries=_env_int("QNA_LLM_MAX_RETRIES", 2, minimum=0),
        timeout_action=_env_choice("QNA_TIMEOUT_ACTION", "fallback", {"fallback", "error"}),
        force_llm_failure=_env_choice("QNA_FORCE_LLM_FAILURE", "none", {"none", "timeout", "error"}),
        log_store=_env_choice("QNA_LOG_STORE", "memory", {"memory", "jsonl"}),
        log_dir=Path(os.getenv("QNA_LOG_DIR", ".qna_logs")),
        route_retry_enabled=_env_bool("QNA_ROUTE_RETRY", True),
        reference_files_enabled=True,
        reference_strip_bucket_prefix=True,
        reference_search_snippet_chars=120,
        reference_presign_disabled=False,
        reference_presign_fallback_on_error=False,
    )
