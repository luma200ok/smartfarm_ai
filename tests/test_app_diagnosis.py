"""app/views/diagnosis.py — 페이지 렌더 스모크 테스트(이슈 #59, 진단 로직 src 흡수).

infer.py 흡수 이후에도 화면 결과가 그대로인지 AppTest로 페이지 전체를 실행해 검증한다
(단순 `import` 성공만으로는 render() 내부에서 `from dl import infer` 뒤의 실제 게이트·
Grad-CAM·YOLO 호출 경로 오류를 못 잡는다 — 예시 사진 버튼까지 클릭해 실행까지 확인).
모델·샘플 사진이 로컬에 없으면 skip(models/*.pt 는 gitignore — conftest 스킵 패턴과 동일)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_PAGE = ROOT / "app" / "views" / "diagnosis.py"
CKPT = ROOT / "models" / "tomato_resnet18.pt"
YOLO_CKPT = ROOT / "models" / "tomato_yolov8n.pt"

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="진단 모델(models/tomato_resnet18.pt) 없음 — 로컬 전용")


def test_diagnosis_page_renders_without_exception_empty_state():
    """업로드·샘플 선택 전 빈 상태도 예외 없이 완주해야 한다."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)

    assert not at.exception, f"페이지 렌더 중 예외 발생: {[str(e) for e in at.exception]}"


def test_diagnosis_sample_click_runs_predict_with_cam_path():
    """'잎곰팡이병' 예시 사진 클릭 → OOD/부위 게이트 통과 → infer.predict_with_cam() 경로까지
    예외 없이 완주해야 한다(이슈 #59 — 구 뷰 중복 구현을 infer.py 호출로 교체한 핵심 경로)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.key == "samp_leaf_mold"]
    assert len(buttons) == 1, "예시 사진 버튼(samp_leaf_mold)이 없음 — app/samples/leaf_mold.jpg 확인"
    buttons[0].click().run(timeout=30)

    assert not at.exception, f"진단 렌더 중 예외 발생: {[str(e) for e in at.exception]}"
    body = " ".join(md.value for md in at.subheader) + " ".join(md.value for md in at.markdown)
    assert "진단" in body


@pytest.mark.skipif(not YOLO_CKPT.exists(), reason="검출 모델(models/tomato_yolov8n.pt) 없음 — 로컬 전용")
def test_diagnosis_yolo_sample_click_runs_detect_annotated_path():
    """YOLO 탭 예시 사진 클릭 → infer.detect_annotated() 경로까지 예외 없이 완주해야 한다."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DIAGNOSIS_PAGE))
    at.run(timeout=30)
    assert not at.exception

    buttons = [b for b in at.button if b.key == "det_leaf_mold"]
    assert len(buttons) == 1, "YOLO 예시 사진 버튼(det_leaf_mold)이 없음 — app/samples/yolo_leaf_mold.jpg 확인"
    buttons[0].click().run(timeout=30)

    assert not at.exception, f"검출 렌더 중 예외 발생: {[str(e) for e in at.exception]}"
