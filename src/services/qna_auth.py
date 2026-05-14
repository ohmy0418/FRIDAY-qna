# 사용자 인증 유틸 모듈.
# JWT access token에서 user_id를 추출(user_id_from_token)하고
# 내부 사용자 여부를 확인(is_authenticated_internal_user)한다.
# Keycloak/OIDC 토큰은 보통 ``sub``에 주체 ID가 있고 ``id``는 없을 수 있다.
from __future__ import annotations

import base64
import json


def is_authenticated_internal_user(user_id: str | None) -> bool:
    return bool(user_id and user_id.strip())


def user_id_from_token(access_token: str | None) -> str | None:
    if not access_token:
        return None
    try:
        payload_b64 = access_token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        raw = payload.get("id") or payload.get("sub") or payload.get("preferred_username")
        if raw is None:
            return None
        s = str(raw).strip()
        return s or None
    except Exception:
        return None
