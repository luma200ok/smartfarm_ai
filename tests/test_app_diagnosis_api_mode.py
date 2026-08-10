"""app/views/diagnosis.py — API 모드(SMARTFARM_API_URL) 렌더 스모크 테스트(이슈 #59 PR 3).

기존 test_app_diagnosis.py는 in-process 경로(로컬 모델 필요, 없으면 skip)만 다룬다. 이 파일은
`api_client.diagnose_remote`(탭1)·`api_client.detect_remote`(탭2, PR 3-1 분리)를 몽키패치해
API 모드 분기(`_render_diagnosis_api`/`_render_detect_api`)가 예외 없이 완주하는지만 확인한다
— 로컬 모델 가중치가 없어도 항상 실행돼야 한다(API 모드의 존재 이유 자체가 클라이언트에
모델이 없어도 되는 것이므로 CKPT.exists() 에 의존하지 않는다).
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


def _fake_diagnosis_response(ood_blocked=False, part_blocked=False):
    if ood_blocked:
        return {"ood_blocked": True, "reason": "식물·잎으로 보이지 않는 사진(OOD)", "label": None,
                "label_kr": None, "prob": None, "probs": None, "part": None,
                "plant_score": 0.012, "part_prob": None,
                "cam_png_base64": None, "yolo": None}
    if part_blocked:
        return {"ood_blocked": True, "reason": "잎이 아닌 부위로 판정(과실)", "label": None,
                "label_kr": None, "prob": None, "probs": None, "part": "fruit",
                "plant_score": 0.5, "part_prob": 0.87,
                "cam_png_base64": None, "yolo": None}
    return {
        "ood_blocked": False,
        "reason": None,
        "label": "leaf_mold",
        "label_kr": "잎곰팡이병",
        "prob": 0.91,
        "probs": {"late_blight": 0.02, "leaf_mold": 0.91, "normal": 0.05, "tylcv": 0.02},
        "part": "leaf",
        "plant_score": None,
        "part_prob": None,
        "cam_png_base64": _png_base64((255, 0, 0)),
        "yolo": {"annotated_png_base64": _png_base64((0, 255, 0)),
                 "boxes": [{"label": "leaf_mold", "conf": 0.83}]},
    }


def _fake_detection_response():
    return {"annotated_png_base64": _png_base64((0, 255, 0)),
            "boxes": [{"label": "leaf_mold", "conf": 0.77}]}


@pytest.fixture
def _api_mode(monkeypatch):
    """SMARTFARM_API_URL 설정 + api_client.diagnose_remote/detect_remote 몽키패치(실 HTTP 호출 없음)."""
    import sys
    for p in (ROOT / "src", ROOT / "app"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import api_client

    monkeypatch.setenv("SMARTFARM_API_URL", "http://fake-smartfarm-api:8000")
    monkeypatch.setattr(api_client, "diagnose_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg": _fake_diagnosis_response())
    monkeypatch.setattr(api_client, "detect_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg": _fake_detection_response())
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
    """API 모드에서 탭2(YOLO) 예시 사진 클릭 → detect_remote(게이트 없는 /api/detections 전용)
    몽키패치 응답으로 검출 탭이 렌더돼야 한다(이슈 #59 PR 3-1 — diagnose_remote가 아닌
    detect_remote가 호출되는지까지 검증: diagnose_remote 몽키패치의 label_kr이 아니라
    detect_remote 몽키패치의 boxes 내용이 화면에 나와야 한다)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.key == "det_leaf_mold"]
    assert len(buttons) == 1, "YOLO 예시 사진 버튼(det_leaf_mold)이 없음"
    buttons[0].click().run(timeout=30)

    assert not at.exception, f"API 모드 검출 렌더 중 예외 발생: {[str(e) for e in at.exception]}"
    body = " ".join(md.value for md in at.subheader) + " ".join(md.value for md in at.markdown)
    assert "검출 1건" in body
    assert "77" in body  # detect_remote 몽키패치의 conf=0.77 이 신뢰도 문자열로 보여야 함


def test_diagnosis_page_api_mode_ood_blocked_shows_unified_message(monkeypatch, _api_mode):
    """OOD 차단 응답 → in-process와 동일한 템플릿(퍼센트 포함)의 안내 문구가 떠야 한다(P2-2).

    plant_score=0.012 → "잎·식물 신호 1.2%"로 in-process `_ood_blocked_message()`와 같은
    포맷이어야 한다(문구 함수 공용화 검증)."""
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
    assert "토마토 잎으로 보이지 않아요" in errors
    assert "1.2%" in errors  # plant_score=0.012 → {score:.1%}
    assert "잎이 화면에 크게 보이도록 촬영해 업로드하세요" in errors


def test_diagnosis_page_api_mode_part_blocked_shows_unified_message(monkeypatch, _api_mode):
    """부위 게이트 차단 응답 → in-process `_part_blocked_message()`와 동일한 문구여야 한다(P2-2).

    part=fruit·part_prob=0.87 → "과실(열매)...부위 신뢰도 87%" + 범위 안내 문구까지 포함."""
    import api_client

    monkeypatch.setattr(api_client, "diagnose_remote",
                         lambda image_bytes, conf=0.25, filename="image.jpg": _fake_diagnosis_response(
                             part_blocked=True))

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.key == "samp_leaf_mold"]
    buttons[0].click().run(timeout=30)

    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "과실(열매)" in errors
    assert "87%" in errors  # part_prob=0.87 → {part_prob:.0%}
    assert "과실·꽃·줄기 병해는 현재 진단 범위가 아닙니다" in errors
