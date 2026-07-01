"""src/dl/infer.py forecast — 습도위험 밴드·trend·graceful (모델 모킹)."""
import numpy as np

from dl import infer


def _fake_model(monkeypatch):
    model = infer.TempLSTM(n_feat=len(infer.ENV_FEATURES)).eval().to(infer.device)
    meta = {"features": infer.ENV_FEATURES, "window": infer.WINDOW,
            "mu": [0.0] * 8, "sd": [1.0] * 8}
    monkeypatch.setattr(infer, "load_forecast_model", lambda: (model, meta))


def _window(humidity):
    w = np.zeros((infer.WINDOW, len(infer.ENV_FEATURES)), dtype="float32")
    w[:, 0] = 25.0                       # 온도내부_평균
    w[:, infer._HUM_IDX] = humidity      # 습도내부_평균
    return w


def test_forecast_returns_expected_keys(monkeypatch):
    _fake_model(monkeypatch)
    r = infer.forecast(_window(70))
    assert set(r) >= {"next_temp", "recent_temp", "trend", "humidity_risk", "humidity_mean"}
    assert r["trend"] in {"상승", "유지", "하강"}


def test_forecast_humidity_bands(monkeypatch):
    _fake_model(monkeypatch)
    assert infer.forecast(_window(85))["humidity_risk"] == "높음"
    assert infer.forecast(_window(70))["humidity_risk"] == "보통"
    assert infer.forecast(_window(50))["humidity_risk"] == "낮음"


def test_forecast_none_when_model_missing(monkeypatch):
    monkeypatch.setattr(infer, "load_forecast_model", lambda: None)
    assert infer.forecast(_window(70)) is None
