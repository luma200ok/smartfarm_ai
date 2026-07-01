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
