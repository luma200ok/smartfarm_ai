"""
Phase 3-4 — 디스코드 Webhook 알림.

조기경보(B)·진단 처방(C)을 디스코드 채널로 밀어준다(수동 버튼 트리거).
웹훅 URL은 .env의 DISCORD_WEBHOOK_URL(시크릿). 미설정이어도 앱은 죽지 않고 안내만.
절대 예외를 밖으로 전파하지 않는다(알림 실패가 처방/경보 흐름을 막지 않도록).
"""
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_log = logging.getLogger(__name__)
_TIMEOUT = 10
_COLOR = {"경고": 15158332, "주의": 15105570, "정상": 3066993}   # 빨강/주황/초록
_GRAY = 9807270      # 알 수 없는 경보수준(오탈자 등) — 정상(초록)로 오인 방지
_BLUE = 3447003


def _t(s, n: int = 1024) -> str:
    """디스코드 필드/제목 길이 제한 대비 truncate. 빈 값은 '-'로(빈 필드 거부 방지)."""
    s = (s or "").strip() or "-"
    return s if len(s) <= n else s[: n - 1] + "…"


def send_discord(embed: dict) -> tuple[bool, str]:
    """embed 1개를 웹훅으로 전송. (성공여부, 메시지). 예외 전파 없음."""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return False, "DISCORD_WEBHOOK_URL 미설정 (.env에 추가하세요)"
    try:
        r = requests.post(url, json={"embeds": [embed]}, timeout=_TIMEOUT)
        if 200 <= r.status_code < 300:
            return True, "디스코드로 전송했어요"
        return False, f"디스코드 응답 오류 {r.status_code}"
    except Exception as e:
        # str(e)에 웹훅 URL(토큰)이 섞일 수 있어 화면엔 예외 타입만, 상세는 로그로만.
        _log.exception("디스코드 전송 예외")
        return False, f"전송 실패 — 네트워크/설정 오류({type(e).__name__})"


def notify_warning(w) -> tuple[bool, str]:
    """조기경보(pipeline.Warning) → 디스코드 embed."""
    fields = [
        {"name": "경보수준", "value": _t(w.경보수준, 256), "inline": True},
        {"name": "위험병해", "value": _t(w.위험병해, 256), "inline": True},
        {"name": "이유", "value": _t(w.이유)},
    ]
    if (w.권장조치 or "").strip():
        fields.append({"name": "권장조치", "value": _t(w.권장조치)})
    embed = {"title": "⚠️ 토마토 조기 경보", "color": _COLOR.get(w.경보수준, _GRAY),
             "fields": fields}
    return send_discord(embed)


def notify_prescription(p) -> tuple[bool, str]:
    """진단 처방(prescribe.Prescription) → 디스코드 embed."""
    fields = [
        {"name": "원인", "value": _t(p.원인)},
        {"name": "즉시 조치", "value": _t(p.즉시조치)},
        {"name": "예방", "value": _t(p.예방)},
    ]
    if (p.재촬영시점 or "").strip():
        fields.append({"name": "재촬영 시점", "value": _t(p.재촬영시점)})
    # 필드 4×1024 + title 256 + footer 1024 ≈ 5.4k < embed 총합 6000 제한.
    embed = {"title": _t(f"💊 {p.진단요약}", 256), "color": _BLUE, "fields": fields}
    if p.근거출처:
        embed["footer"] = {"text": _t("근거: " + " · ".join(p.근거출처), 1024)}
    return send_discord(embed)
