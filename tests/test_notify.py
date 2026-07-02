"""src/llm/notify.py — 디스코드 Webhook 알림 (requests 모킹, 예외 전파 없음)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm import notify


def _warning(level="주의"):
    return SimpleNamespace(경보수준=level, 위험병해="잎곰팡이병", 이유="고습 지속", 권장조치="야간 환기")


def _presc():
    return SimpleNamespace(진단요약="잎곰팡이병 의심", 원인="습도", 즉시조치="감염 잎 제거",
                           예방="환기", 재촬영시점="3일 후", 근거출처=["가이드 — https://x"])


def test_no_url_skips_post(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    with patch("requests.post") as p:
        ok, msg = notify.send_discord({"title": "t"})
    assert ok is False and "미설정" in msg
    p.assert_not_called()


def test_notify_warning_posts_colored_embed(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    with patch("requests.post", return_value=MagicMock(status_code=204)) as p:
        ok, _ = notify.notify_warning(_warning("경고"))
    assert ok is True
    embed = p.call_args.kwargs["json"]["embeds"][0]
    assert embed["title"].startswith("⚠️")
    assert embed["color"] == 15158332                       # 경고 = 빨강
    assert {"경보수준", "위험병해", "이유"} <= {f["name"] for f in embed["fields"]}


def test_notify_prescription_embed_has_title_and_footer(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    with patch("requests.post", return_value=MagicMock(status_code=200)) as p:
        ok, _ = notify.notify_prescription(_presc())
    assert ok is True
    embed = p.call_args.kwargs["json"]["embeds"][0]
    assert "잎곰팡이병" in embed["title"]
    assert "footer" in embed                                # 근거출처 → footer


def test_non_2xx_is_graceful(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    with patch("requests.post", return_value=MagicMock(status_code=500)):
        ok, msg = notify.send_discord({"title": "t"})
    assert ok is False and "500" in msg


def test_exception_is_graceful(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook")
    with patch("requests.post", side_effect=Exception("boom")):
        ok, msg = notify.send_discord({"title": "t"})
    assert ok is False                                      # 예외 전파 안 함


def test_failure_message_does_not_leak_url(monkeypatch):
    """예외 메시지에 웹훅 URL/토큰이 섞여 화면에 노출되면 안 됨(P1 회귀)."""
    token = "https://discord.com/api/webhooks/123/SECRET_TOKEN_abc"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", token)
    with patch("requests.post", side_effect=Exception(f"Max retries with url: {token}")):
        ok, msg = notify.send_discord({"title": "t"})
    assert ok is False
    assert "SECRET_TOKEN_abc" not in msg and "webhooks" not in msg


def test_truncate_empty_and_long():
    assert notify._t("", 10) == "-"
    assert notify._t("a" * 2000, 10).endswith("…") and len(notify._t("a" * 2000, 10)) == 10
