"""src/llm/pipeline.py — 일일코치(A)·조기경보(B) (ollama·forecast·retrieve 모킹)."""
import json
from unittest.mock import patch

from llm import pipeline
from llm.pipeline import Coach, Warning


def test_daily_coach_returns_schema(monkeypatch):
    monkeypatch.setattr(pipeline, "get_forecast",
                        lambda: {"next_temp": 30.0, "trend": "유지", "humidity_risk": "보통", "humidity_mean": 70.0})
    out = json.dumps({"요약": "맑음", "오늘_할일": ["환기하기"], "근거": "온도 안정"}, ensure_ascii=False)
    with patch("ollama.chat", return_value={"message": {"content": out}}):
        c = pipeline.daily_coach()
    assert isinstance(c, Coach) and c.오늘_할일 == ["환기하기"]


def test_early_warning_high_humidity_uses_leafmold_rag(monkeypatch):
    monkeypatch.setattr(pipeline, "get_forecast",
                        lambda: {"next_temp": 30.0, "trend": "유지", "humidity_risk": "높음", "humidity_mean": 92.0})
    seen = {"disease": None}

    def _rt(q, disease=None, k=3):
        seen["disease"] = disease
        return [{"text": "환기하라", "title": "가이드", "source": ""}]

    monkeypatch.setattr(pipeline, "retrieve", _rt)
    out = json.dumps({"경보수준": "주의", "위험병해": "잎곰팡이병", "이유": "고습", "권장조치": "환기"}, ensure_ascii=False)
    with patch("ollama.chat", return_value={"message": {"content": out}}):
        w = pipeline.early_warning()
    assert seen["disease"] == "leaf_mold"       # 고습 → 잎곰팡이병 근거 검색
    assert isinstance(w, Warning) and w.경보수준 == "주의"


def test_early_warning_low_humidity_skips_rag(monkeypatch):
    monkeypatch.setattr(pipeline, "get_forecast",
                        lambda: {"next_temp": 24.0, "trend": "유지", "humidity_risk": "낮음", "humidity_mean": 45.0})
    calls = {"n": 0}
    monkeypatch.setattr(pipeline, "retrieve",
                        lambda q, disease=None, k=3: calls.__setitem__("n", calls["n"] + 1) or [])
    out = json.dumps({"경보수준": "정상", "위험병해": "없음", "이유": "저습", "권장조치": ""}, ensure_ascii=False)
    with patch("ollama.chat", return_value={"message": {"content": out}}):
        pipeline.early_warning()
    assert calls["n"] == 0                       # 저습 → RAG 미검색


def test_early_warning_no_forecast_returns_normal(monkeypatch):
    monkeypatch.setattr(pipeline, "get_forecast", lambda: {"unavailable": True, "reason": "x"})
    w = pipeline.early_warning()
    assert w.경보수준 == "정상" and w.위험병해 == "없음"
