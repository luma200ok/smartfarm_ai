"""POST /api/detections — 이슈 #59 PR 3-1(검출 전용 엔드포인트 분리).

`/api/diagnoses`(진단+검출 결합, 게이트 있음)와 달리 게이트 없이 `infer.detect_annotated()`만
호출한다 — in-process `app/views/diagnosis.py` 탭2(YOLO 위치 검출)와 동일 의미. 업로드 캡·
픽셀 폭탄 방어·동시성 캡은 `/api/diagnoses`와 같은 공용 헬퍼를 재사용하므로 그 하드닝 테스트를
그대로 미러한다(test_api_diagnoses.py 참고).
"""
import base64
import io

from api.concurrency import MAX_CONCURRENT_INFERENCE, _inference_slots
from api.uploads import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from dl import infer
from PIL import Image


def _post_image(api_client, path, **kwargs):
    with open(path, "rb") as f:
        return api_client.post("/api/detections", files={"file": ("leaf.jpg", f, "image/jpeg")}, **kwargs)


def test_detections_returns_boxes_for_leaf(api_client, leaf_image):
    r = _post_image(api_client, leaf_image)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["boxes"], list)
    for box in body["boxes"]:
        assert set(box) == {"label", "conf"}
    annotated_png = base64.b64decode(body["annotated_png_base64"])
    assert annotated_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_detections_no_gate_for_nonleaf_part(api_client, nonleaf_image):
    """`/api/diagnoses`와 달리 게이트가 없다 — 잎이 아닌 사진도 200으로 그대로 검출 결과를 낸다
    (in-process 탭2 동작과 동일 의미 — 결과 동일성 원칙)."""
    r = _post_image(api_client, nonleaf_image)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["boxes"], list)
    assert "annotated_png_base64" in body


def test_detections_empty_file_is_400(api_client):
    r = api_client.post("/api/detections", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    assert r.json() == {"detail": "빈 파일입니다."}


def test_detections_non_image_file_is_400(api_client):
    r = api_client.post("/api/detections", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    assert r.json() == {"detail": "유효한 이미지 파일이 아닙니다."}


def test_detections_missing_file_is_422(api_client):
    r = api_client.post("/api/detections")
    assert r.status_code == 422


def test_detections_conf_query_param_forwarded(api_client, leaf_image, monkeypatch):
    captured = {}
    real_detect = infer.detect_annotated

    def _spy(pil, conf=0.25):
        captured["conf"] = conf
        return real_detect(pil, conf=conf)

    monkeypatch.setattr("api.routers.detections.infer.detect_annotated", _spy)
    r = _post_image(api_client, leaf_image, params={"conf": 0.9})
    assert r.status_code == 200
    assert captured["conf"] == 0.9


# ── 업로드 크기 캡(413) ────────────────────────────────────────────────
def test_detections_oversized_file_is_413(api_client):
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    r = api_client.post("/api/detections", files={"file": ("big.jpg", oversized, "image/jpeg")})
    assert r.status_code == 413
    assert r.json() == {"detail": "파일이 너무 큽니다(최대 10MB)."}


# ── 픽셀 폭탄 방어(400) ──────────────────────────────────────────────────
def test_detections_pixel_bomb_is_400(api_client):
    side = 6000
    assert side * side > MAX_IMAGE_PIXELS
    buf = io.BytesIO()
    Image.new("RGB", (side, side), color=(10, 20, 30)).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    assert len(png_bytes) < MAX_UPLOAD_BYTES

    r = api_client.post("/api/detections", files={"file": ("bomb.png", png_bytes, "image/png")})
    assert r.status_code == 400
    assert r.json() == {"detail": "이미지 해상도가 너무 큽니다(최대 2500만 픽셀)."}


# ── 동시성 캡(429) — diagnoses·prescriptions와 슬롯 공유 ───────────────────
def test_detections_returns_429_when_slots_exhausted(api_client, leaf_image):
    acquired = [_inference_slots.acquire(blocking=False) for _ in range(MAX_CONCURRENT_INFERENCE)]
    assert all(acquired)
    try:
        r = _post_image(api_client, leaf_image)
        assert r.status_code == 429
        assert r.json() == {"detail": "서버가 혼잡합니다. 잠시 후 다시 시도하세요."}
    finally:
        for _ in acquired:
            _inference_slots.release()
