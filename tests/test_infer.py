"""src/dl/infer.py 순수 추론 계층 — 실모델(MPS/CPU) 사용.

predict_with_cam·detect_annotated 는 이슈 #59(app/views/diagnosis.py 중복 흡수)로
추가된 UI 전용 반환 형태 — diagnose()/detect() 와 같은 게이트·모델을 재사용하는지도 검증한다.
"""
import numpy as np
from PIL import Image

from dl import infer


def test_diagnose_returns_valid_label(leaf_image):
    r = infer.diagnose(Image.open(leaf_image))
    assert r["label"] in infer.CLASSES
    assert 0.0 <= r["prob"] <= 1.0
    assert abs(sum(r["probs"].values()) - 1.0) < 1e-3
    assert set(r["probs"]) == set(infer.CLASSES)


def test_part_of_returns_known_part(leaf_image):
    part, prob = infer.part_of(Image.open(leaf_image))
    assert part in infer.PART_CLASSES
    assert 0.0 <= prob <= 1.0


def test_ood_score_in_range(leaf_image):
    score = infer.ood_plant_score(Image.open(leaf_image))
    assert 0.0 <= score <= 1.0


def test_detect_structure(leaf_image):
    boxes = infer.detect(Image.open(leaf_image))
    assert isinstance(boxes, list)
    for b in boxes:
        assert set(b) == {"cls", "conf", "region"}
        assert b["region"] in {"상단", "중단", "하단"}
        assert 0.0 <= b["conf"] <= 1.0


def test_predict_with_cam_returns_expected_shapes(leaf_image):
    """이슈 #59 — app/views/diagnosis.py 가 그대로 언팩하는 (label, prob, probs, cam, img) 5튜플."""
    label, prob, probs, cam, img = infer.predict_with_cam(Image.open(leaf_image))
    assert label in infer.CLASSES
    assert 0.0 <= prob <= 1.0
    assert probs.shape == (len(infer.CLASSES),)
    assert abs(float(probs.sum()) - 1.0) < 1e-3
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert img.shape == (224, 224, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0 + 1e-6


def test_predict_with_cam_agrees_with_diagnose(leaf_image):
    """Grad-CAM 경로와 diagnose() 경로가 같은 load_diagnosis_model()을 쓰므로 라벨이 갈리면 안 된다."""
    pil = Image.open(leaf_image)
    label_cam, prob_cam, _, _, _ = infer.predict_with_cam(pil)
    d = infer.diagnose(pil)
    assert label_cam == d["label"]
    assert abs(prob_cam - d["prob"]) < 1e-4


def test_detect_annotated_structure(leaf_image):
    pil = Image.open(leaf_image)
    annotated, dets = infer.detect_annotated(pil)
    assert isinstance(annotated, np.ndarray)
    assert annotated.ndim == 3 and annotated.shape[2] == 3
    assert isinstance(dets, list)
    for lab, c in dets:
        assert isinstance(lab, str)
        assert 0.0 <= c <= 1.0


def test_detect_annotated_agrees_with_detect_classes(leaf_image):
    """detect()/detect_annotated() 는 같은 YOLO 로더를 쓰므로 검출 클래스 집합이 같아야 한다."""
    pil = Image.open(leaf_image)
    boxes = infer.detect(pil)
    _, dets = infer.detect_annotated(pil)
    assert sorted(b["cls"] for b in boxes) == sorted(lab for lab, _ in dets)
