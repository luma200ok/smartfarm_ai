"""POST /api/diagnoses — 이슈 #59 PR 2 + code-reviewer/security-reviewer 후속 픽스.

실모델 파이프라인(OOD → 부위 게이트 → 진단+YOLO) + 업로드 캡(P1-1)·픽셀 폭탄 방어(P1-2)·
동시성 캡(P2-2)·에러 상세 비노출(P2-3).
"""
import base64
import io

from api.concurrency import MAX_CONCURRENT_INFERENCE, _inference_slots
from api.uploads import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from dl import infer
from PIL import Image


def _post_image(api_client, path, **kwargs):
    with open(path, "rb") as f:
        return api_client.post("/api/diagnoses", files={"file": ("leaf.jpg", f, "image/jpeg")}, **kwargs)


def test_diagnoses_passes_for_leaf(api_client, leaf_image):
    r = _post_image(api_client, leaf_image)
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is False
    assert body["label"] in infer.CLASSES
    assert body["label_kr"] == infer.LABEL_KR[body["label"]]
    assert body["part"] == "leaf"
    assert 0.0 <= body["prob"] <= 1.0
    assert abs(sum(body["probs"].values()) - 1.0) < 1e-3
    # Grad-CAM PNG — 유효한 base64 이미지 바이트인지만 확인(픽셀 내용은 test_infer.py가 검증)
    png = base64.b64decode(body["cam_png_base64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert isinstance(body["yolo"]["boxes"], list)
    for box in body["yolo"]["boxes"]:
        assert set(box) == {"label", "conf"}
    annotated_png = base64.b64decode(body["yolo"]["annotated_png_base64"])
    assert annotated_png[:8] == b"\x89PNG\r\n\x1a\n"
    # 성공 응답엔 plant_score·part_prob를 안 싣는다(게이트 차단 안내 문구 전용 필드 — P2-2)
    assert body["plant_score"] is None
    assert body["part_prob"] is None


# ── P2-1: include_visuals=False — Grad-CAM·YOLO 계산 생략(처방 뷰 재사용 시 비용 절감) ──
def test_diagnoses_include_visuals_false_omits_cam_and_yolo(api_client, leaf_image):
    r = _post_image(api_client, leaf_image, params={"include_visuals": False})
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is False
    assert body["label"] in infer.CLASSES
    assert body["label_kr"] == infer.LABEL_KR[body["label"]]
    assert body["part"] == "leaf"
    assert 0.0 <= body["prob"] <= 1.0
    assert abs(sum(body["probs"].values()) - 1.0) < 1e-3
    assert body["cam_png_base64"] is None
    assert body["yolo"] is None


def test_diagnoses_include_visuals_true_still_default(api_client, leaf_image):
    """include_visuals 쿼리를 생략하면 기존(True) 동작 그대로 — 하위 호환."""
    r = _post_image(api_client, leaf_image)
    assert r.status_code == 200
    body = r.json()
    assert body["cam_png_base64"] is not None
    assert body["yolo"] is not None


def test_diagnoses_include_visuals_false_still_blocks_gates(api_client, ood_image):
    """include_visuals=False여도 OOD/부위 게이트는 그대로 적용된다(시각화만 생략)."""
    r = _post_image(api_client, ood_image, params={"include_visuals": False})
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is True


def test_diagnoses_blocks_ood(api_client, ood_image):
    r = _post_image(api_client, ood_image)
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is True
    assert body["reason"]
    assert body["label"] is None
    assert body["cam_png_base64"] is None
    assert body["yolo"] is None
    # code-reviewer P2-2 — plant_score를 실어 보내야 클라이언트가 in-process와 동일한
    # 퍼센트 포함 안내 문구를 만들 수 있다.
    assert body["plant_score"] is not None
    assert 0.0 <= body["plant_score"] < 1.0
    assert body["part_prob"] is None  # OOD 게이트에서 막혔으므로 부위 게이트는 아직 안 돌았음


def test_diagnoses_blocks_nonleaf_part(api_client, nonleaf_image):
    r = _post_image(api_client, nonleaf_image)
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is True
    assert body["part"] != "leaf"
    assert body["reason"]
    assert body["label"] is None
    # code-reviewer P2-2
    assert body["plant_score"] is not None
    assert body["part_prob"] is not None
    assert 0.0 <= body["part_prob"] <= 1.0


def test_diagnoses_empty_file_is_400(api_client):
    r = api_client.post("/api/diagnoses", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    assert r.json() == {"detail": "빈 파일입니다."}


def test_diagnoses_non_image_file_is_400(api_client):
    """P2-3 — 클라이언트엔 고정 문구만(예외 클래스명·메시지 등 내부 상세 비노출)."""
    r = api_client.post("/api/diagnoses", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    assert r.json() == {"detail": "유효한 이미지 파일이 아닙니다."}


def test_diagnoses_missing_file_is_422(api_client):
    r = api_client.post("/api/diagnoses")
    assert r.status_code == 422


def test_diagnoses_conf_query_param_forwarded(api_client, leaf_image, monkeypatch):
    """conf 쿼리 파라미터가 infer.detect_annotated(conf=...)로 그대로 전달되는지."""
    captured = {}
    real_detect = infer.detect_annotated

    def _spy(pil, conf=0.25):
        captured["conf"] = conf
        return real_detect(pil, conf=conf)

    monkeypatch.setattr("api.routers.diagnoses.infer.detect_annotated", _spy)
    r = _post_image(api_client, leaf_image, params={"conf": 0.9})
    assert r.status_code == 200
    assert captured["conf"] == 0.9


# ── P1-1: 업로드 크기 캡(413) ────────────────────────────────────────────
def test_diagnoses_oversized_file_is_413(api_client):
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)
    r = api_client.post("/api/diagnoses", files={"file": ("big.jpg", oversized, "image/jpeg")})
    assert r.status_code == 413
    assert r.json() == {"detail": "파일이 너무 큽니다(최대 10MB)."}


def test_diagnoses_file_at_cap_boundary_is_not_413(api_client):
    """정확히 MAX_UPLOAD_BYTES(경계값)는 413이 아니어야 한다(비이미지라 400은 됨 — 캡 통과만 확인)."""
    at_cap = b"x" * MAX_UPLOAD_BYTES
    r = api_client.post("/api/diagnoses", files={"file": ("at_cap.jpg", at_cap, "image/jpeg")})
    assert r.status_code != 413


# ── P1-2: 픽셀 폭탄 방어(400) ─────────────────────────────────────────────
def test_diagnoses_pixel_bomb_is_400(api_client):
    """압축 파일 크기는 작지만(단색 PNG) 디코드하면 MAX_IMAGE_PIXELS를 넘는 이미지는 400.

    Pillow 기본 Image.MAX_IMAGE_PIXELS(~89.5M)는 이 크기(36M px)를 그냥 통과시킨다 —
    그래서 앱 차원의 명시적 상한(uploads.MAX_IMAGE_PIXELS=2500만)이 필요하다(P1-2).
    """
    side = 6000  # 6000*6000 = 36,000,000 px > MAX_IMAGE_PIXELS(2500만), Pillow 기본 경고 임계 미만
    assert side * side > MAX_IMAGE_PIXELS
    buf = io.BytesIO()
    Image.new("RGB", (side, side), color=(10, 20, 30)).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    assert len(png_bytes) < MAX_UPLOAD_BYTES  # 단색이라 압축 후엔 매우 작음 — 크기 캡과는 무관한 케이스

    r = api_client.post("/api/diagnoses", files={"file": ("bomb.png", png_bytes, "image/png")})
    assert r.status_code == 400
    assert r.json() == {"detail": "이미지 해상도가 너무 큽니다(최대 2500만 픽셀)."}


# ── P2-2: 동시성 캡(429) ───────────────────────────────────────────────
def test_diagnoses_returns_429_when_slots_exhausted(api_client, leaf_image):
    """diagnoses·prescriptions가 슬롯(전역 세마포어)을 공유하므로, 슬롯을 미리 다 채우면 429."""
    acquired = [_inference_slots.acquire(blocking=False) for _ in range(MAX_CONCURRENT_INFERENCE)]
    assert all(acquired)
    try:
        r = _post_image(api_client, leaf_image)
        assert r.status_code == 429
        assert r.json() == {"detail": "서버가 혼잡합니다. 잠시 후 다시 시도하세요."}
    finally:
        for _ in acquired:
            _inference_slots.release()
