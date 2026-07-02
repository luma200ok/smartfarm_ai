"""src/llm/expect.py — pkl lazy-load 우아한 저하 + predict() 순수함수."""
from llm import expect


def test_load_model_none_when_pkl_missing(monkeypatch):
    monkeypatch.setattr(expect, "PKL_PATH", expect.ROOT / "models" / "__does_not_exist__.pkl")
    assert expect.load_model(force=True) is None


def test_expected_none_when_model_missing(monkeypatch):
    monkeypatch.setattr(expect, "PKL_PATH", expect.ROOT / "models" / "__does_not_exist__.pkl")
    expect.load_model(force=True)
    assert expect.expected({"온도외부_평균": 10.0, "일사량_평균": 5.0}, "2024-01-15") is None


class _FakeReg:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        import numpy as np
        return np.array([self.value] * len(X))


def _fake_model():
    return {
        "models": {"평균": _FakeReg(15.0), "최저": _FakeReg(10.0)},
        "features": ["온도외부_평균", "일사량_평균", "doy_sin", "doy_cos"],
        "resid_sigma": {"평균": 1.4, "최저": 1.9},
    }


def test_predict_returns_targets_and_sigma():
    model = _fake_model()
    out = expect.predict(model, {"온도외부_평균": 5.0, "일사량_평균": 3.0}, "2024-01-15")
    assert out == {"평균": 15.0, "최저": 10.0, "resid_sigma": {"평균": 1.4, "최저": 1.9}}


def test_predict_none_when_reading_missing_feature():
    model = _fake_model()
    assert expect.predict(model, {"온도외부_평균": 5.0}, "2024-01-15") is None


def test_predict_none_when_model_none():
    assert expect.predict(None, {"온도외부_평균": 5.0, "일사량_평균": 3.0}, "2024-01-15") is None


def test_doy_handles_multiple_date_formats():
    assert expect.doy("2024-01-01") == 1
    assert expect.doy("20240101") == 1
    assert expect.doy("2024-02-01") == 32


def test_z_score():
    assert expect.z_score(10.0, 15.0, 2.5) == -2.0
    assert expect.z_score(10.0, 15.0, 0) == 0.0
