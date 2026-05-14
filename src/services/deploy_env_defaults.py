"""
배포에서 루트 `.env`를 쓰지 못할 때: 비어 있는 환경 변수만 문자열로 채운다.

- 이미 설정된 OS·플랫폼 시크릿 값은 덮어쓰지 않는다.
- :func:`apply_deploy_env_defaults`는 **최초** ``import core.config`` (또는
  ``db.connect_rdb``) **이전**에 호출되어야 ``Config`` 객체에 반영된다.
- 아래 기본값 중 DB·MinIO 등은 FRIDAY 배포 샘플을 따른 것이므로, 환경에 맞게
  이 파일의 상수 또는 플랫폼 환경 변수로 조정할 것.
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
    _set_if_blank("TEST_UI", "true")
    _set_if_blank("APP_HOST", "0.0.0.0")
    _set_if_blank("APP_PORT", "8888")
    _set_if_blank("APP_HEARTBEAT_SEC", "30")
    _set_if_blank("LOG_LEVEL", "DEBUG")
    _set_if_blank("LOG_DIRECTORY", "app_logs")
    _set_if_blank("APP_ID", "dd39500f-6e21-41ab-ad20-8909d83ecc3e")
    _set_if_blank("STORAGE_PATH", "GRAPHIO_APP_STORAGE")
    _set_if_blank("GRAPHIO_APP_STORAGE_PATH", "GRAPHIO_APP_STORAGE")
    _set_if_blank("GRAPHIO_APP_STORAGE", "graphio_app_storage")
    _set_if_blank("GRAPHIO_APP_DB", "graphio_app.db")
    _set_if_blank("DOCUMENT_STORE", "document_store")
    _set_if_blank("EMBEDDING_MODEL", "text-embedding-3-small")
    _set_if_blank("LLM_MODEL", "google/gemma-4-31B-it")
    _set_if_blank("LLM_MODEL_TYPE", "openAI")
    _set_if_blank("LLM_API_ADDRESS", "http://192.168.109.254:32609/v1")
    _set_if_blank("GRAPHIO_ONTOLOGY_HOST", "192.168.109.254")
    _set_if_blank("GRAPHIO_ONTOLOGY_PORT", "32576")
    _set_if_blank("GRAPHIO_ONTOLOGY_USER", "neo4j")
    _set_if_blank("GRAPHIO_ONTOLOGY_PASS", "biris.manse")
    _set_if_blank("C_APP_PLATFORM_HOST", "http://192.168.109.254")
    _set_if_blank("C_APP_PLATFORM_PORT", "31557")
    _set_if_blank("C_APP_PLATFORM_TIMEOUT", "30")
    _set_if_blank("C_APP_PLATFORM_ADD_MESSAGE", "/graphio/app_platform/control/api/message/add")
    _set_if_blank("C_APP_PLATFORM_UPDATE_TITLE", "/graphio/app_platform/control/api/thread/edit")
    _set_if_blank("C_APP_PLATFORM_GET_APP_INFO", "/graphio/app_platform/control/api/detail")
    _set_if_blank("KNOWLEDGE_GRAPH_HOST", "http://192.168.109.254")
    _set_if_blank("KNOWLEDGE_GRAPH_PORT", "30058")
    _set_if_blank("KNOWLEDGE_GRAPH_TIMEOUT", "30")
    _set_if_blank("KNOWLEDGE_GRAPH_GET_KG_LIST", "/graphio/v1/knowledgeGraph/list")
    _set_if_blank("CONTAINER_MANAGER_MQ_HOST", "192.168.109.254")
    _set_if_blank("CONTAINER_MANAGER_MQ_PORT", "30988")
    _set_if_blank("CONTAINER_MANAGER_MQ_USER", "admin")
    _set_if_blank("CONTAINER_MANAGER_MQ_PASS", "admin123")
    _set_if_blank(
        "CONTAINER_MANAGER_MQ_HEADER_EXCHANGE",
        "graphioapp.container-service.exchange",
    )
    _set_if_blank("PID_FILE", "process.pid")
    _set_if_blank("C_MINIO_CLIENT_HOST", "192.168.109.254")
    _set_if_blank("C_MINIO_CLIENT_PORT", "30901")
    _set_if_blank("C_MINIO_CONTEXT_PATH", "download")
    _set_if_blank("C_MINIO_ACCESS_KEY", "root")
    _set_if_blank("C_MINIO_SECRET_KEY", "platform.manse")
    _set_if_blank("C_MINIO_BUCKET_NAME", "friday")
    _set_if_blank("C_DATABASE_HOST", "192.168.109.254")
    _set_if_blank("C_DATABASE_PORT", "31032")
    _set_if_blank("C_DATABASE_USER", "root")
    _set_if_blank("C_DATABASE_PASSWORD", "biris.manse")
    _set_if_blank("C_DATABASE_DB", "office_guide")
    _set_if_blank("C_DATABASE_SCHEMA", "friday")
    _set_if_blank("QNA_RAG_DB", "1")
    _set_if_blank("QNA_STRUCTURED_DB", "1")
    _set_if_blank("C_PHOENIX_USE", "true")
    _set_if_blank("C_PHOENIX_ENDPOINT", "http://192.168.109.254:31365/v1/traces")
    _set_if_blank("C_PHOENIX_PROJECT_NAME", "APP_TEST")
    _set_if_blank("RAG_SERVICE_URL", "http://192.168.101.20:9999")
    _set_if_blank("C_GOVERNANCE_ENDPOINT", "http://192.168.109.254:30880")
    _set_if_blank("C_GOVERNANCE_TIMEOUT", "30")
    _set_if_blank("C_RERANK_URL", "http://192.168.101.129:30659")
    _set_if_blank("C_RERANK_MODEL", "sigridjineth/ko-reranker-v1.1-preview")
    _set_if_blank("C_DOCUMENT_TEMPLATE_PATH", "document_template")
    _set_if_blank("C_DOCUMENT_TEMP_PATH", "/document_temp")
    _set_if_blank("C_CHAT_FILE_ACCESS_PATH", "/mount/vol/chat_files")
    _set_if_blank("LOG_CONSOLE_ENABLED", "True")

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
