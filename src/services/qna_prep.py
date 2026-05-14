# QnA 서브그래프 실행 전 인증·clarify 파라미터를 AgentState에 넣는 전처리 노드.
# 상위 graph(graph.py)는 ``ontology_config_info`` → ``qna_prep`` → 컴파일된 ``qna_graph``(노드명 qna) 순으로 연결한다.
# JWT에서 user_id를 못 얻으면 ``configurable["user_id"]`` 또는 state의 ``user_id``를 사용한다(사내 상위 에이전트 연동).
# 그것도 없으면 user_id를 빈 문자열로 두고, 서브그래프 ``qna_graph``가 에러 메시지를 반환한다.
from __future__ import annotations

import os

from langchain_core.runnables import RunnableConfig

from services.agent import AgentState
from services.qna_auth import is_authenticated_internal_user, user_id_from_token
from utils.logger import LOG


class QnAPrep:
    async def __call__(self, state: AgentState, config: RunnableConfig) -> dict:
        configurable = config.get("configurable", {})
        access_token = configurable.get("access_token")

        dbg_uid = (os.getenv("GRAPHIO_LOADER_DEBUG_USER_ID") or "").strip()
        if dbg_uid:
            prev_uid = (state.get("user_id") or "").strip() if isinstance(state, dict) else (
                (getattr(state, "user_id", None) or "").strip()
            )
            LOG.info(
                "QnAPrep: GRAPHIO_LOADER_DEBUG_USER_ID=%r — state.user_id before override=%r (토큰 경로 건너뜀)",
                dbg_uid,
                prev_uid or "(없음)",
            )
            active_tool = configurable.get("active_tool") or {}
            clarify_emp_nm = active_tool.get("clarify_emp_nm") if isinstance(active_tool, dict) else None
            clarify_dept_hint = active_tool.get("clarify_dept_hint") if isinstance(active_tool, dict) else None
            return {
                "user_id": dbg_uid,
                "clarify_emp_nm": clarify_emp_nm or None,
                "clarify_dept_hint": clarify_dept_hint or None,
            }

        user_id = user_id_from_token(access_token)

        if not is_authenticated_internal_user(user_id):
            cfg_uid = (configurable.get("user_id") or "").strip()
            state_uid = (
                (state.get("user_id") or "").strip()
                if isinstance(state, dict)
                else (getattr(state, "user_id", None) or "").strip()
            )
            trusted_uid = cfg_uid or state_uid
            if trusted_uid:
                user_id = trusted_uid
            else:
                dev_ok = os.getenv("QNA_DEV_ALLOW_NO_AUTH", "").strip().lower() in ("1", "true", "yes", "on")
                if dev_ok:
                    user_id = (os.getenv("QNA_DEV_FALLBACK_USER_ID") or "dev-anonymous").strip() or "dev-anonymous"
                    LOG.warning("QnAPrep: QNA_DEV_ALLOW_NO_AUTH — using fallback user_id=%r", user_id)
                else:
                    LOG.warning("QnAPrep: unauthenticated user")
                    user_id = ""

        active_tool = configurable.get("active_tool") or {}
        clarify_emp_nm = active_tool.get("clarify_emp_nm") if isinstance(active_tool, dict) else None
        clarify_dept_hint = active_tool.get("clarify_dept_hint") if isinstance(active_tool, dict) else None

        return {
            "user_id": user_id,
            "clarify_emp_nm": clarify_emp_nm or None,
            "clarify_dept_hint": clarify_dept_hint or None,
        }
