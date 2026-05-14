# 최상위 LangGraph 스레드 공유 상태(AgentState)를 정의한다.
# 가이드(주의사항): AgentState는 반드시 BaseAgentState를 상속해야 합니다.
from typing import Optional

from langgraph.managed.is_last_step import IsLastStep

from graphio_app_framework.states import BaseAgentState
from utils.store import VectorStore


class AgentState(BaseAgentState, total=False):
    """LangGraph Thread 상태. QnA·스튜디오·벡터스토어 등 확장 필드."""

    is_last_step: IsLastStep
    is_chart_last_step: IsLastStep
    vectorstore_manager: VectorStore
    studio_input: Optional[dict]
    studio_type: Optional[str]
    app_report_list: Optional[list]
    user_id: Optional[str]
    clarify_emp_nm: Optional[str]
    clarify_dept_hint: Optional[str]
