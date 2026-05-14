# 검색 결과와 LLM 실행 결과를 바탕으로 최종 응답 상태(success·fallback·error)를 결정한다.
# failure_reason을 QNA4xx/5xx 형식 에러코드로 변환하는 매핑도 여기서 관리한다.
from __future__ import annotations

from services.qna_config import TimeoutAction
from services.qna_types import FailureReason, RouteType, StatusType

FAILURE_ERROR_CODE_MAP: dict[FailureReason, str] = {
    "routing_failed": "QNA422",
    "no_result": "QNA404",
    "permission_denied": "QNA403",
    "llm_timeout": "QNA504",
    "llm_error": "QNA502",
    "unknown": "QNA500",
    "structured_person_not_found": "QNA404",
    "structured_clarification": "QNA200",
    "off_topic": "QNA400",
}


def determine_status(
    route_type: RouteType,
    has_context: bool,
    failure_reason: FailureReason | None,
) -> tuple[StatusType, FailureReason | None]:
    if route_type == "unsupported":
        return "fallback", "routing_failed"
    if has_context:
        return "success", None
    if failure_reason:
        return "fallback", failure_reason
    return "error", "unknown"


def resolve_llm_failure(timeout_action: TimeoutAction, reason: FailureReason) -> tuple[StatusType, FailureReason]:
    if reason == "llm_timeout" and timeout_action == "fallback":
        return "fallback", reason
    return "error", reason


def failure_reason_to_error_code(reason: FailureReason | None) -> str | None:
    if reason is None:
        return None
    return FAILURE_ERROR_CODE_MAP.get(reason, "QNA500")
