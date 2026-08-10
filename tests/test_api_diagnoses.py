"""POST /api/diagnoses — 이슈 #59 PR 2. 실모델 파이프라인(OOD → 부위 게이트 → 진단+YOLO) + 에러."""
import base64

from dl import infer


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


def test_diagnoses_blocks_ood(api_client, ood_image):
    r = _post_image(api_client, ood_image)
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is True
    assert body["reason"]
    assert body["label"] is None
    assert body["cam_png_base64"] is None
    assert body["yolo"] is None


def test_diagnoses_blocks_nonleaf_part(api_client, nonleaf_image):
    r = _post_image(api_client, nonleaf_image)
    assert r.status_code == 200
    body = r.json()
    assert body["ood_blocked"] is True
    assert body["part"] != "leaf"
    assert body["reason"]
    assert body["label"] is None


def test_diagnoses_empty_file_is_400(api_client):
    r = api_client.post("/api/diagnoses", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_diagnoses_non_image_file_is_400(api_client):
    r = api_client.post("/api/diagnoses", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    assert "detail" in r.json()


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
