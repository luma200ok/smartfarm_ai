"""
Phase 2 (DL) — Streamlit 데모: 토마토 잎 사진 → 진단(+Grad-CAM) · 위치 검출(YOLO)

탭1 진단: resnet18(전이학습) 추론 → 정상/질병 + '어디를 보고 판단했나' Grad-CAM 히트맵
탭2 검출: YOLOv8n → 장면에서 잎을 찾아 박스 + 정상/질병 라벨 + 신뢰도
멀티페이지: app/streamlit_app.py 가 render() 를 호출(set_page_config 는 엔트리에서 1회).
단독 실행: streamlit run app/phase2_dl.py   (프로젝트 루트에서)

모델: models/tomato_resnet18.pt (진단) · models/tomato_yolov8n.pt (검출)
  없으면 → prepare_tomato.py → 02_core.py --chunk 2-5 / prepare_tomato_yolo.py → 05_detect.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "tomato_resnet18.pt"
YOLO_CKPT = ROOT / "models" / "tomato_yolov8n.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES = ["leaf_mold", "normal", "tylcv"]       # ImageFolder 알파벳순(학습과 동일)
LABEL_KR = {"leaf_mold": "🦠 잎곰팡이병", "normal": "🌿 정상",
            "tylcv": "🦠 황화잎말이바이러스"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model():
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location=device))
    return m.eval().to(device)


def predict_with_cam(model, pil):
    """추론 + Grad-CAM → (label, prob, probs, cam224, img224)."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = tf(pil.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)

    store = {}
    layer = model.layer4[-1]

    def _save(m, i, o):
        o.retain_grad()
        store["act"] = o
    handle = layer.register_forward_hook(_save)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    cls = int(probs.argmax())
    model.zero_grad()
    logits[0, cls].backward()

    act = store["act"].detach()[0]
    grad = store["act"].grad[0]
    w = grad.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * act).sum(0))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(224, 224),
                        mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
    handle.remove()

    # 표시용 원본(정규화 역연산)
    img = x.detach()[0].cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
    return CLASSES[cls], float(probs[cls]), probs.detach().cpu().numpy(), cam, img


def overlay(img, cam):
    """원본 위에 jet 히트맵을 반투명 합성."""
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return (0.55 * img + 0.45 * heat).clip(0, 1)


@st.cache_resource
def load_yolo():
    """검출용 YOLOv8 모델(05_detect.py 산출물)."""
    from ultralytics import YOLO
    return YOLO(str(YOLO_CKPT))


def detect(yolo, pil, conf=0.25):
    """YOLO 검출 → (박스 그려진 RGB 이미지, [(label, conf), ...])."""
    res = yolo.predict(pil.convert("RGB"), device=device, conf=conf, verbose=False)[0]
    annotated = res.plot()[:, :, ::-1]                 # BGR → RGB
    names = res.names
    dets = [(names[int(b.cls)], float(b.conf)) for b in res.boxes]
    return annotated, dets


# ── 페이지 렌더 (멀티페이지 엔트리가 호출) ───────────────────────────────────
def render():
    st.title("🍅 토마토 잎 병해 — 진단(Grad-CAM) · 위치 검출(YOLO)")
    st.caption("ML로는 불가능한 '사진 진단' + 설명가능 AI(어느 병반을 보고 판단했나) + 장면 속 잎 위치 검출")

    tab_diag, tab_detect = st.tabs(["🔬 진단 + Grad-CAM", "🎯 위치 검출 (YOLO)"])

    # ── 탭 1: 분류 진단 + Grad-CAM ──
    with tab_diag:
        if not CKPT.exists():
            st.error(f"진단 모델이 없습니다: {CKPT}\n\n터미널에서 먼저 실행하세요:\n"
                     "1) `python src/dl/prepare_tomato.py`\n"
                     "2) `python src/dl/02_core.py --chunk 2-5`")
        else:
            model = load_model()
            up = st.file_uploader("토마토 잎 사진 업로드", type=["jpg", "jpeg", "png"], key="diag")
            if up:
                pil = Image.open(up)
                label, prob, probs, cam, img = predict_with_cam(model, pil)

                st.subheader(f"진단: {LABEL_KR[label]}  (확률 {prob:.1%})")
                if label != "normal":
                    st.warning(f"{LABEL_KR[label]}이(가) 의심됩니다. 오른쪽 히트맵의 붉은 영역(병반 추정)을 확인하세요.")
                else:
                    st.success("정상으로 판단됩니다.")

                c1, c2 = st.columns(2)
                c1.image(img, caption="입력(224×224)", use_container_width=True)
                c2.image(overlay(img, cam), caption="Grad-CAM — 판단 근거 영역", use_container_width=True)

                st.markdown("**클래스별 확률**")
                st.bar_chart({c: float(p) for c, p in zip([LABEL_KR[c] for c in CLASSES], probs)})
                st.caption("⚠️ Grad-CAM은 보조 지표 — 잎맥·배경 등 비병변 영역에 반응할 수 있어 사람 검수가 필요합니다.")
            else:
                st.info("잎 사진을 업로드하면 진단 결과와 Grad-CAM 히트맵이 표시됩니다.")

    # ── 탭 2: YOLO 위치 검출 ──
    with tab_detect:
        if not YOLO_CKPT.exists():
            st.error(f"검출 모델이 없습니다: {YOLO_CKPT}\n\n터미널에서 먼저 실행하세요:\n"
                     "1) `python src/dl/prepare_tomato_yolo.py`\n"
                     "2) `python src/dl/05_detect.py`")
        else:
            yolo = load_yolo()
            up2 = st.file_uploader("토마토 잎 사진 업로드", type=["jpg", "jpeg", "png"], key="detect")
            conf = st.slider("신뢰도 임계값(conf)", 0.05, 0.9, 0.25, 0.05)
            if up2:
                pil2 = Image.open(up2)
                annotated, dets = detect(yolo, pil2, conf=conf)
                st.image(annotated, caption="YOLO 검출 — 박스 위치 + 정상/질병 + 신뢰도",
                         use_container_width=True)
                if dets:
                    st.subheader(f"검출 {len(dets)}건")
                    for lab, c in dets:
                        st.write(f"- {LABEL_KR.get(lab, lab)} — 신뢰도 {c:.1%}")
                else:
                    st.info("임계값 이상으로 검출된 잎이 없습니다. conf 슬라이더를 낮춰 보세요.")
            else:
                st.info("잎 사진을 업로드하면 잎 위치 박스와 정상/질병 라벨이 표시됩니다.")


if __name__ == "__main__":
    # 단독 실행 시에만 페이지 설정(멀티페이지에선 엔트리가 담당)
    st.set_page_config(page_title="토마토 잎 진단 (Phase 2 DL)", page_icon="🍅")
    render()
