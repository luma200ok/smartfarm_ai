"""src/llm/tools.py — function-calling tool + 게이트 파이프라인(실모델)."""
from dl import infer
from llm import tools


def test_get_diagnosis_passes_for_leaf(leaf_image):
    r = tools.get_diagnosis(leaf_image)
    assert r["ood_blocked"] is False
    assert r["label"] in infer.CLASSES
    assert r["label_kr"] == infer.LABEL_KR[r["label"]]
    assert r["part"] == "leaf"


def test_get_diagnosis_blocks_nonleaf(nonleaf_image):
    """과실 사진 → 부위 게이트②가 진단 대신 차단 사유."""
    r = tools.get_diagnosis(nonleaf_image)
    assert r["ood_blocked"] is True
    assert r.get("reason")


def test_get_diagnosis_blocks_ood(ood_image):
    """비식물(노이즈) → OOD 게이트①가 차단(환각 방어 ①)."""
    r = tools.get_diagnosis(ood_image)
    assert r["ood_blocked"] is True
    assert r.get("reason")


def test_get_detection_passes_for_leaf(leaf_image):
    r = tools.get_detection(leaf_image)
    assert r["ood_blocked"] is False
    assert r["lesion_count"] == len(r["boxes"])


def test_get_detection_blocks_ood(ood_image):
    """검출도 진단과 동일하게 비식물을 차단."""
    r = tools.get_detection(ood_image)
    assert r["ood_blocked"] is True


def test_get_forecast_structure(monkeypatch):
    import numpy as np
    monkeypatch.setattr(tools.infer, "latest_window",
                        lambda: np.zeros((7, 8), dtype="float32"))
    monkeypatch.setattr(tools.infer, "forecast",
                        lambda w: {"next_temp": 30.0, "trend": "유지",
                                   "humidity_risk": "보통", "recent_temp": 29.5, "humidity_mean": 70.0})
    r = tools.get_forecast()
    assert r["next_temp"] == 30.0 and r["humidity_risk"] == "보통"


def test_get_forecast_unavailable_without_data(monkeypatch):
    monkeypatch.setattr(tools.infer, "latest_window", lambda: None)
    assert tools.get_forecast()["unavailable"] is True


def test_get_forecast_uses_explicit_window(monkeypatch):
    """window 명시(가상 센서) 시 latest_window를 거치지 않고 그 창으로 예측."""
    import numpy as np
    monkeypatch.setattr(tools.infer, "latest_window",
                        lambda: (_ for _ in ()).throw(AssertionError("latest_window 호출되면 안 됨")))
    monkeypatch.setattr(tools.infer, "forecast",
                        lambda w: {"next_temp": 22.0, "trend": "하강", "humidity_risk": "높음",
                                   "recent_temp": 24.0, "humidity_mean": 88.0})
    r = tools.get_forecast(np.zeros((7, 8), dtype="float32"))
    assert r["next_temp"] == 22.0 and r["humidity_risk"] == "높음"


def test_get_weather_registered_in_tool_registry_and_schemas():
    assert tools.TOOL_REGISTRY["get_weather"] is tools.get_weather
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert "get_weather" in names


def test_get_weather_current_delegates_to_weather_module(monkeypatch):
    monkeypatch.setattr(tools.weather, "get_current", lambda: {"unavailable": False, "temp": 20.0})
    r = tools.get_weather("current")
    assert r == {"unavailable": False, "temp": 20.0}


def test_get_weather_forecast_delegates_to_weather_module(monkeypatch):
    monkeypatch.setattr(tools.weather, "get_forecast_3d", lambda: {"unavailable": False, "daily": []})
    r = tools.get_weather("forecast")
    assert r == {"unavailable": False, "daily": []}


def test_get_weather_default_kind_is_forecast(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.weather, "get_forecast_3d", lambda: calls.append("forecast") or {"unavailable": False})
    tools.get_weather()
    assert calls == ["forecast"]


def test_get_weather_swallows_exceptions(monkeypatch):
    def _boom():
        raise RuntimeError("boom")
    monkeypatch.setattr(tools.weather, "get_current", _boom)
    r = tools.get_weather("current")
    assert r["unavailable"] is True
