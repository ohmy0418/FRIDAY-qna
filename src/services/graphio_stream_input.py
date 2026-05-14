# ``StreamInput.stream_tokens`` → ``RunnableConfig.configurable.active_tool`` 브리지.
# ``generator.stream_generator``가 ``stream_tokens=True``일 때 최종 AIMessage를 SSE로 보내지 않으므로
# ``qna_graph``가 이 플래그를 보고 ``loading_message`` + ``type: exception`` 디스패치로 본문을 올린다.
#
# ``graphio-app run`` 은 ``graphio_app_framework.api.generator`` 를 쓰고,
# ``uvicorn api.graphio_app:app`` 은 ``api.generator`` 를 쓰므로 둘 다 패치한다.

from __future__ import annotations

import sys
import threading
from typing import Any

GRAPHIO_STREAM_TOKENS_ACTIVE_TOOL_KEY = "__graphio_stream_tokens__"

# graphio-app CLI 쪽이 먼저 로드되는 경우가 많아 프레임워크 모듈을 앞에 둔다.
_PARSE_INPUT_BRIDGE_TARGETS: tuple[str, ...] = (
    "graphio_app_framework.api.generator",
    "api.generator",
)


def _merge_stream_tokens_into_kwargs(user_input: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    # graphio-app CLI → ``graphio_app_framework.api.schema.StreamInput``,
    # ``uvicorn api.graphio_app:app`` → ``api.schema.StreamInput`` — 동일 필드이나 클래스가 달라
    # ``isinstance(..., api.schema.StreamInput)`` 만 쓰면 프레임워크 요청에서 브리지가 전부 무시된다.
    if getattr(user_input, "stream_tokens", None) is None:
        return kwargs
    flag = bool(user_input.stream_tokens)
    cfg = kwargs.get("config")
    if cfg is None:
        return kwargs
    configurable = cfg.get("configurable") if isinstance(cfg, dict) else getattr(cfg, "configurable", None)
    if not isinstance(configurable, dict):
        return kwargs
    at = configurable.get("active_tool")
    if not isinstance(at, dict):
        at = {}
    new_conf = {
        **configurable,
        "active_tool": {**at, GRAPHIO_STREAM_TOKENS_ACTIVE_TOOL_KEY: flag},
    }
    if isinstance(cfg, dict):
        new_cfg = {**cfg, "configurable": new_conf}
    else:
        new_cfg = {**dict(cfg), "configurable": new_conf}
    return {**kwargs, "config": new_cfg}


def _install_parse_input_bridge(gen: Any) -> bool:
    if getattr(gen.parse_input, "_graphio_stream_token_bridge", False):
        return True
    orig = gen.parse_input

    async def parse_input(user_input: Any, access_token: str | None):
        out = await orig(user_input, access_token)
        kwargs, run_id, thread_id = out
        kwargs = _merge_stream_tokens_into_kwargs(user_input, kwargs)
        return kwargs, run_id, thread_id

    parse_input._graphio_stream_token_bridge = True  # type: ignore[attr-defined]
    gen.parse_input = parse_input
    return True


def _try_install_parse_input_bridge(attempt: int = 0) -> None:
    for mod_name in _PARSE_INPUT_BRIDGE_TARGETS:
        gen = sys.modules.get(mod_name)
        if gen is not None and callable(getattr(gen, "parse_input", None)):
            _install_parse_input_bridge(gen)
    if attempt < 120:
        threading.Timer(0.005, lambda: _try_install_parse_input_bridge(attempt + 1)).start()


def schedule_parse_input_stream_token_bridge() -> None:
    """``parse_input`` 이 로드된 뒤(앱·프레임워크 generator 모두) 브리지를 지연 설치한다."""
    threading.Timer(0.0, lambda: _try_install_parse_input_bridge(0)).start()
