"""app/views/prescribe.py — API 모드(SMARTFARM_API_URL) 렌더 스모크 테스트(code-reviewer P2-3).

`api_client.health_remote`/`diagnose_remote`/`prescribe_remote`를 몽키패치해 실 HTTP 호출 없이
4가지 분기를 검증한다:
  ① 정상 흐름 — diagnose_remote·prescribe_remote 몽키패치 → 처방까지 렌더
  ② diagnose_remote 실패 — st.error + 조기 종료(prescribe_remote 미호출)
  ③ /api/health의 ollama.online=False — 처방 버튼이 "DL 진단만" 모드로, 처방 섹션 미렌더
  ④ API 다운(health_remote 자체가 ApiClientError) — api_error 분기(연결 실패 안내)
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRESCRIBE_PAGE = ROOT / "app" / "views" / "prescribe.py"


def _health_online():
    return {"status": "ok", "ollama": {"online": True, "models": ["qwen2.5:14b"]},
            "artifacts": {"resnet18": True, "yolo": True, "part": True, "leaf_gate": True}}


def _health_ollama_offline():
    return {"status": "ok", "ollama": {"online": False, "models": None},
            "artifacts": {"resnet18": True, "yolo": True, "part": True, "leaf_gate": True}}


def _diagnosis_success():
    return {
        "ood_blocked": False, "reason": None, "label": "leaf_mold", "label_kr": "잎곰팡이병",
        "prob": 0.88, "probs": {"late_blight": 0.02, "leaf_mold": 0.88, "normal": 0.08, "tylcv": 0.02},
        "part": "leaf", "plant_score": None, "part_prob": None,
        "cam_png_base64": None, "yolo": None,
    }


def _prescription_success():
    return {"진단요약": "잎곰팡이병으로 보입니다(신뢰도 88%)", "원인": "다습한 환경에서 곰팡이가 번식해요",
            "즉시조치": "병든 잎을 제거하고 환기하세요", "예방": "습도를 60% 이하로 관리하세요",
            "재촬영시점": "1주일 후", "근거출처": ["농사로 재배가이드"]}


@pytest.fixture
def _api_mode(monkeypatch):
    """SMARTFARM_API_URL 설정 + api_client 모듈 확보(각 테스트가 필요한 함수만 개별 몽키패치)."""
    import sys
    for p in (ROOT / "src", ROOT / "app"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import api_client

    monkeypatch.setenv("SMARTFARM_API_URL", "http://fake-smartfarm-api:8000")
    return api_client


def _select_leaf_mold_sample(at):
    at.radio[0].set_value("leaf_mold").run(timeout=30)
    assert not at.exception, f"샘플 선택 중 예외 발생: {[str(e) for e in at.exception]}"


def _main_button(at):
    buttons = [b for b in at.button if "처방" in b.label or "진단" in b.label]
    assert len(buttons) == 1, f"처방/진단 버튼을 찾지 못함: {[b.label for b in at.button]}"
    return buttons[0]


# ── ① 정상 흐름 ─────────────────────────────────────────────────────────
def test_prescribe_page_api_mode_full_flow_renders_prescription(monkeypatch, _api_mode):
    monkeypatch.setattr(_api_mode, "health_remote", _health_online)
    monkeypatch.setattr(_api_mode, "diagnose_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg", include_visuals=True:
                         _diagnosis_success())
    monkeypatch.setattr(_api_mode, "prescribe_remote",
                         lambda question, diagnosis=None, image_bytes=None, filename="image.jpg":
                         _prescription_success())

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PRESCRIBE_PAGE))
    at.run(timeout=30)
    assert not at.exception, f"페이지 렌더 중 예외 발생: {[str(e) for e in at.exception]}"

    _select_leaf_mold_sample(at)
    _main_button(at).click().run(timeout=30)
    assert not at.exception, f"처방 흐름 중 예외 발생: {[str(e) for e in at.exception]}"

    body = " ".join(md.value for md in at.markdown) + " ".join(s.value for s in at.success)
    assert "잎곰팡이병으로 보입니다" in body  # 처방(Prescription.진단요약)
    assert "다습한 환경" in body  # 원인
    assert "잎곰팡이병" in body  # DL 진단 패널(diag['label_kr'])


# ── ② diagnose_remote 실패 → st.error + 조기 종료(prescribe_remote 미호출) ──
def test_prescribe_page_api_mode_diagnose_failure_stops_before_prescribe(monkeypatch, _api_mode):
    monkeypatch.setattr(_api_mode, "health_remote", _health_online)

    def _fail_diagnose(image_bytes, conf=0.25, filename="image.jpg", include_visuals=True):
        raise _api_mode.ApiClientError("서빙 API 연결 실패")

    monkeypatch.setattr(_api_mode, "diagnose_remote", _fail_diagnose)

    presc_calls = []
    monkeypatch.setattr(_api_mode, "prescribe_remote",
                         lambda *a, **kw: presc_calls.append((a, kw)) or _prescription_success())

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PRESCRIBE_PAGE))
    at.run(timeout=30)
    assert not at.exception

    _select_leaf_mold_sample(at)
    _main_button(at).click().run(timeout=30)
    assert not at.exception, f"진단 실패 처리 중 예외 발생: {[str(e) for e in at.exception]}"

    errors = " ".join(e.value for e in at.error)
    assert "서빙 API 연결 실패" in errors
    assert presc_calls == []  # 조기 종료 — 처방 호출 자체가 없어야 함
    body = " ".join(md.value for md in at.markdown) + " ".join(s.value for s in at.success)
    assert "잎곰팡이병으로 보입니다" not in body  # 처방 섹션이 렌더되지 않아야 함


# ── ③ /api/health가 ollama offline → 처방 스킵, DL 진단만 안내 ─────────────
def test_prescribe_page_api_mode_ollama_offline_skips_prescription(monkeypatch, _api_mode):
    monkeypatch.setattr(_api_mode, "health_remote", _health_ollama_offline)
    monkeypatch.setattr(_api_mode, "diagnose_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg", include_visuals=True:
                         _diagnosis_success())

    presc_calls = []
    monkeypatch.setattr(_api_mode, "prescribe_remote",
                         lambda *a, **kw: presc_calls.append((a, kw)) or _prescription_success())

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PRESCRIBE_PAGE))
    at.run(timeout=30)
    assert not at.exception

    warnings = " ".join(w.value for w in at.warning)
    assert "LLM 오프라인" in warnings
    assert _main_button(at).label == "🔬 DL 진단만 받기"

    _select_leaf_mold_sample(at)
    _main_button(at).click().run(timeout=30)
    assert not at.exception, f"DL 진단 전용 흐름 중 예외 발생: {[str(e) for e in at.exception]}"

    assert presc_calls == []  # online=False면 prescribe_remote를 아예 호출하지 않아야 함
    body = " ".join(md.value for md in at.markdown)
    assert "잎곰팡이병" in body  # DL 진단 패널은 렌더됨
    success_body = " ".join(s.value for s in at.success)
    assert "잎곰팡이병으로 보입니다" not in success_body  # LLM 처방 섹션(success)은 없어야 함


# ── ④ API 다운(health_remote 자체가 ApiClientError) — api_error 분기 ──────
def test_prescribe_page_api_mode_api_down_shows_connection_error(monkeypatch, _api_mode):
    def _fail_health():
        raise _api_mode.ApiClientError("서빙 API 연결 실패")

    monkeypatch.setattr(_api_mode, "health_remote", _fail_health)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PRESCRIBE_PAGE))
    at.run(timeout=30)
    assert not at.exception, f"페이지 렌더 중 예외 발생: {[str(e) for e in at.exception]}"

    errors = " ".join(e.value for e in at.error)
    assert "서빙 API 연결 실패" in errors
    captions = " ".join(c.value for c in at.caption)
    assert "서빙 API에 연결할 수 없어요" in captions
    # "Ollama 데몬 미가동"(온라인/오프라인 케이스 전용 문구)과는 구분돼야 한다
    warnings = " ".join(w.value for w in at.warning)
    assert "LLM 오프라인" not in warnings
