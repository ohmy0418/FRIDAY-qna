# graphio 전체 앱의 최상위 LangGraph 그래프를 조립한다.
# loader → ontology → ontology_config_info → qna_prep → qna(서브그래프) → END
# cleanup·타이틀·file_use 등은 graphio_app() 래퍼가 주입한다(가이드 준수).
import os

from services.deploy_env_defaults import load_repo_dotenv_then_apply_defaults

load_repo_dotenv_then_apply_defaults()

from langchain_core.messages import ChatMessage as LangchainChatMessage
from langgraph.graph import END, StateGraph

from graphio_app_framework.graph_base.graph import graphio_app
from service_utils.util_node import ontology_config_info, ontology_info
from services.agent import AgentState
from services.graphio_stream_input import schedule_parse_input_stream_token_bridge
from services.qna_graph import qna_graph
from services.qna_prep import QnAPrep


async def loader(state: AgentState):
    out: dict = {
        "messages": [
            LangchainChatMessage(
                content="질문을 분석하고 답변을 생성하는 중", role="assistant", additional_kwargs={"type": "loading"})
        ],
    }
    # 스모크: `GRAPHIO_LOADER_DEBUG_USER_ID` 설정 시 state.user_id를 넣어 qna_prep 이전 단계까지 전달되는지 확인.
    dbg_uid = (os.getenv("GRAPHIO_LOADER_DEBUG_USER_ID") or "").strip()
    if dbg_uid:
        out["user_id"] = dbg_uid
    return out


def build_qna_top_graph() -> StateGraph:
    """StateGraph 빌더만 반환한다. compile은 graphio_app()에 맡긴다."""
    agent = StateGraph(AgentState)
    agent.add_node("loader", loader)
    agent.add_node("ontology", ontology_info)
    agent.add_node("ontology_config_info", ontology_config_info)
    agent.add_node("qna_prep", QnAPrep())
    agent.add_node("qna", qna_graph)

    agent.set_entry_point("loader")
    agent.add_edge("loader", "ontology")
    agent.add_edge("ontology", "ontology_config_info")
    agent.add_edge("ontology_config_info", "qna_prep")
    agent.add_edge("qna_prep", "qna")
    agent.add_edge("qna", END)

    return agent


graphio_app_flow = graphio_app(build_qna_top_graph, state_type=AgentState)
schedule_parse_input_stream_token_bridge()
