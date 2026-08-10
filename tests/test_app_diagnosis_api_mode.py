"""app/views/diagnosis.py — API 모드(SMARTFARM_API_URL) 렌더 스모크 테스트(이슈 #59 PR 3).

기존 test_app_diagnosis.py는 in-process 경로(로컬 모델 필요, 없으면 skip)만 다룬다. 이 파일은
`api_client.diagnose_remote`를 몽키패치해 API 모드 분기(`_render_diagnosis_api`/`_render_detect_api`)
가 예외 없이 완주하는지만 확인한다 — 로컬 모델 가중치가 없어도 항상 실행돼야 한다(API 모드의
존재 이유 자체가 클라이언트에 모델이 없어도 되는 것이므로 CKPT.exists() 에 의존하지 않는다).
"""
import base64
import io
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_PAGE = ROOT / "app" / "views" / "diagnosis.py"


def _png_base64(color):
    img = Image.new("RGB", (4, 4), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fake_diagnosis_response(ood_blocked=False):
    if ood_blocked:
        return {"ood_blocked": True, "reason": "식물·잎으로 보이지 않는 사진(OOD)", "label": None,
                "label_kr": None, "prob": None, "probs": None, "part": None,
                "cam_png_base64": None, "yolo": None}
    return {
        "ood_blocked": False,
        "reason": None,
        "label": "leaf_mold",
        "label_kr": "잎곰팡이병",
        "prob": 0.91,
        "probs": {"late_blight": 0.02, "leaf_mold": 0.91, "normal": 0.05, "tylcv": 0.02},
        "part": "leaf",
        "cam_png_base64": _png_base64((255, 0, 0)),
        "yolo": {"annotated_png_base64": _png_base64((0, 255, 0)),
                 "boxes": [{"label": "leaf_mold", "conf": 0.83}]},
    }


@pytest.fixture
def _api_mode(monkeypatch):
    """SMARTFARM_API_URL 설정 + api_client.diagnose_remote 몽키패치(실 HTTP 호출 없음)."""
    import sys
    for p in (ROOT / "src", ROOT / "app"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import api_client

    monkeypatch.setenv("SMARTFARM_API_URL", "http://fake-smartfarm-api:8000")
    monkeypatch.setattr(api_client, "diagnose_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg": _fake_diagnosis_response())
    return api_client


def test_diagnosis_page_api_mode_sample_click_renders_without_exception(_api_mode):
    """API 모드에서 예시 사진 클릭 → diagnose_remote 몽키패치 응답으로 진단 탭이 렌더돼야 한다."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception, f"페이지 렌더 중 예외 발생: {[str(e) for e in at.exception]}"

    buttons = [b for b in at.button if b.key == "samp_leaf_mold"]
    assert len(buttons) == 1, "예시 사진 버튼(samp_leaf_mold)이 없음"
    buttons[0].click().run(timeout=30)

    assert not at.exception, f"API 모드 진단 렌더 중 예외 발생: {[str(e) for e in at.exception]}"
    body = " ".join(md.value for md in at.subheader) + " ".join(md.value for md in at.markdown)
    assert "잎곰팡이병" in body


def test_diagnosis_page_api_mode_yolo_sample_click_renders_without_exception(_api_mode):
    """API 모드에서 탭2(YOLO) 예시 사진 클릭도 예외 없이 완주해야 한다(yolo 필드만 사용)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.key == "det_leaf_mold"]
    assert len(buttons) == 1, "YOLO 예시 사진 버튼(det_leaf_mold)이 없음"
    buttons[0].click().run(timeout=30)

    assert not at.exception, f"API 모드 검출 렌더 중 예외 발생: {[str(e) for e in at.exception]}"


def test_diagnosis_page_api_mode_ood_blocked_shows_error(monkeypatch, _api_mode):
    """게이트 차단 응답이면 st.error로 사유가 표시되고 예외는 없어야 한다."""
    import api_client

    monkeypatch.setattr(api_client, "diagnose_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg": _fake_diagnosis_response(
                             ood_blocked=True))

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.key == "samp_leaf_mold"]
    buttons[0].click().run(timeout=30)

    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "OOD" in errors or "보이지 않는" in errors
