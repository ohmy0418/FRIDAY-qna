"""
배포에서 루트 `.env`를 쓰지 못할 때: 비어 있는 환경 변수만 문자열로 채운다.

- 이미 설정된 OS·플랫폼 시크릿 값은 덮어쓰지 않는다.
- :func:`apply_deploy_env_defaults`는 **최초** ``import core.config`` (또는
  ``db.connect_rdb``) **이전**에 호출되어야 ``Config`` 객체에 반영된다.
- env.template(graphio 프레임워크 기본값)과 동일한 항목은 생략하고,
  이 프로젝트에서 다른 값을 쓰는 항목·프로젝트 전용 항목만 선언한다.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- LLM: 배포 시에만 채움 (저장소·공개 배포 주의) ---
DEPLOY_LLM_API_KEY = ""
DEPLOY_OPENAI_API_KEY = ""


def _set_if_blank(key: str, value: str) -> None:
    if not (os.environ.get(key) or "").strip():
        os.environ[key] = value


def apply_deploy_env_defaults() -> None:
    """비어 있는 키만 기본 문자열로 설정한다."""
    # 프로젝트 전용 (env.template에 없음)
    _set_if_blank("APP_ID", "dd39500f-6e21-41ab-ad20-8909d83ecc3e")
    _set_if_blank("LLM_MODEL_TYPE", "openAI")
    _set_if_blank("QNA_DEV_ALLOW_NO_AUTH", "1")
    _set_if_blank("QNA_DEV_FALLBACK_USER_ID", "local-test-user")
    _set_if_blank("C_DATABASE_SCHEMA", "friday")
    _set_if_blank("QNA_RAG_DB", "1")
    _set_if_blank("QNA_STRUCTURED_DB", "1")

    # env.template 기본값과 다른 항목
    _set_if_blank("EMBEDDING_MODEL", "text-embedding-3-small")
    _set_if_blank("LLM_MODEL", "google/gemma-4-31B-it")
    _set_if_blank("LLM_API_ADDRESS", "http://192.168.109.254:32609/v1")
    _set_if_blank("C_APP_PLATFORM_PORT", "31557")
    _set_if_blank("C_MINIO_CLIENT_HOST", "192.168.109.254")
    _set_if_blank("C_MINIO_CLIENT_PORT", "30901")
    _set_if_blank("C_MINIO_ACCESS_KEY", "root")
    _set_if_blank("C_MINIO_SECRET_KEY", "platform.manse")
    _set_if_blank("C_MINIO_BUCKET_NAME", "friday")
    _set_if_blank("C_DATABASE_PORT", "31032")
    _set_if_blank("C_DATABASE_DB", "office_guide")

    if (DEPLOY_LLM_API_KEY or "").strip() and not (os.environ.get("LLM_API_KEY") or "").strip():
        os.environ["LLM_API_KEY"] = DEPLOY_LLM_API_KEY.strip()
    if (DEPLOY_OPENAI_API_KEY or "").strip() and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = DEPLOY_OPENAI_API_KEY.strip()


def load_repo_dotenv_then_apply_defaults() -> None:
    """레포 루트 ``.env``를 읽은 뒤(가능하면) 비어 있는 키만 채운다.

    ``python-dotenv``가 없으면 ``apply_deploy_env_defaults``만 실행한다.
    레포 루트는 ``src/services/`` 기준 상위 두 단계로 가정한다.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        apply_deploy_env_defaults()
        return
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    apply_deploy_env_defaults()
