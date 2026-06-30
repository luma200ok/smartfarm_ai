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
PART_CKPT = ROOT / "models" / "tomato_part.pt"   # 부위 분류기(과실/꽃/잎/줄기) — 잎 진단 게이트
YOLO_CKPT = ROOT / "models" / "tomato_yolov8n.pt"
SAMPLE_DIR = ROOT / "app" / "samples"            # 예시 사진(사진 없는 외부 체험용)
SAMPLE_CLASSES = ["normal", "leaf_mold", "tylcv"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES = ["leaf_mold", "normal", "tylcv"]       # ImageFolder 알파벳순(학습과 동일)
LABEL_KR = {"leaf_mold": "🦠 잎곰팡이병", "normal": "🌿 정상",
            "tylcv": "🦠 황화잎말이바이러스"}
PART_CLASSES = ["flower", "fruit", "leaf", "stem"]   # 부위 분류기 학습 순서(ImageFolder 알파벳순)
PART_KR = {"flower": "🌼 꽃", "fruit": "🍅 과실(열매)", "leaf": "🌿 잎", "stem": "🌱 줄기"}
# OOD 게이트: 닫힌 3-클래스 분류기는 잎이 아닌 이미지도 한 클래스로 찍는다.
# → 진단 전에 "토마토 잎인가"를 ImageNet 사전학습 분류기로 판별(식물·잎·채소 클래스 softmax 합).
#   우리 3-클래스 모델의 logit/feature 로는 잎·OOD 가 겹쳐(실측 ~17%) 못 가르므로, 1000클래스 지식을 빌린다.
#   추가 학습·설치 없음(torchvision 사전학습 가중치 재사용). 실측: 합성 OOD 100% 차단·진짜 잎 오차단 ~4%.
PLANT_THRESHOLD = 0.04                             # 식물 클래스 확률 합이 이 값 미만이면 "잎 아님"으로 차단
# ImageNet 1000클래스 중 잎·식물·채소·과일 관련(토마토 잎은 ImageNet에 없어 인접 녹색식물로 잡힘)
PLANT_KEYWORDS = ["leaf", "cardoon", "nettle", "cabbage", "broccoli", "cauliflower", "cucumber",
                  "artichoke", "zucchini", "corn", "pot ", "plant", "acorn", "fig", "pineapple",
                  "buckeye", "ear", "hay", "daisy", "mushroom", "bell pepper", "granny smith",
                  "custard apple", "hip", "head cabbage", "spaghetti squash", "butternut"]
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


@st.cache_resource
def load_leaf_gate():
    """OOD 게이트용 ImageNet 사전학습 resnet18 + 식물 클래스 인덱스(진단 모델과 별개)."""
    from torchvision import models
    w = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=w).eval().to(device)
    cats = w.meta["categories"]
    idx = sorted({i for i, c in enumerate(cats)
                  if any(k in c.lower() for k in PLANT_KEYWORDS)})
    return net, w.transforms(), idx


def plant_score(pil):
    """입력이 잎·식물일 정도(ImageNet 식물 클래스 softmax 합, 0~1). 낮을수록 OOD."""
    net, tf, idx = load_leaf_gate()
    with torch.no_grad():
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
        return float(torch.softmax(net(x)[0], 0)[idx].sum())


@st.cache_resource
def load_part_model():
    """부위 분류기(과실/꽃/잎/줄기) resnet18 — 잎 진단 앞단 게이트."""
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(PART_CLASSES))
    m.load_state_dict(torch.load(PART_CKPT, map_location=device))
    return m.eval().to(device)


def predict_part(pil):
    """입력 사진의 토마토 부위 추론 → (부위코드, 확률). leaf 가 아니면 잎 진단을 막는다.

    부위 모델(PART_CKPT)이 없으면 게이트를 건너뛴다('leaf' 취급) — 모델 파일 미배포 시
    앱 전체가 죽지 않게 한다. 비잎 입력은 앞단 plant_score(OOD) 게이트가 1차로 거른다.
    """
    if not PART_CKPT.exists():
        return "leaf", 0.0
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    model = load_part_model()
    with torch.no_grad():
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x)[0], 0)
    idx = int(probs.argmax())
    return PART_CLASSES[idx], float(probs[idx])


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
            st.caption("ℹ️ 이 모델은 **토마토 잎 사진 전용**입니다. "
                       "업로드하면 먼저 **잎/비잎을 판별**해, 잎이 아닌 이미지는 진단하지 않습니다.")
            up = st.file_uploader("토마토 잎 사진 업로드", type=["jpg", "jpeg", "png"], key="diag")

            # 예시 사진은 결과 아래에 렌더(클릭 시 rerun → 위쪽 결과가 갱신됨)
            samples = [(c, SAMPLE_DIR / f"{c}.jpg") for c in SAMPLE_CLASSES]
            samples = [(c, p) for c, p in samples if p.exists()]

            # 입력 결정: 업로드가 있으면 업로드 우선(샘플 선택 해제), 없으면 선택한 예시 사용
            if up:
                st.session_state.pop("diag_sample", None)
                pil = Image.open(up)
            elif st.session_state.get("diag_sample"):
                pil = Image.open(SAMPLE_DIR / f"{st.session_state['diag_sample']}.jpg")
            else:
                pil = None

            if pil is not None:
                st.divider()
                # OOD 게이트: ImageNet 식물 판별기로 잎/비잎을 먼저 거름(임계값 미만이면 진단 차단)
                score = plant_score(pil)
                if score < PLANT_THRESHOLD:
                    pcol, _ = st.columns([1, 2])
                    pcol.image(pil, caption="🔍 분석한 사진", use_container_width=True)
                    st.error(
                        f"🚫 **토마토 잎으로 보이지 않습니다**(잎·식물 신호 {score:.1%}). 진단을 진행하지 않습니다.\n\n"
                        "이 진단기는 토마토 잎 전용입니다. 잎이 화면에 크게 보이도록 촬영해 업로드하세요."
                    )
                elif (part := predict_part(pil))[0] != "leaf":
                    # 식물이지만 잎이 아닌 부위(과실·꽃·줄기) → 잎 진단 차단(과육 오분류 방지)
                    pcol, _ = st.columns([1, 2])
                    pcol.image(pil, caption="🔍 분석한 사진", use_container_width=True)
                    st.error(
                        f"🚫 이 사진은 **{PART_KR[part[0]]}**(으)로 보입니다(부위 신뢰도 {part[1]:.0%}). "
                        "이 진단기는 **토마토 잎 전용**입니다 — 잎이 화면에 크게 보이도록 촬영해 업로드하세요.\n\n"
                        "※ 과실·꽃·줄기 병해는 현재 진단 범위가 아닙니다(학습 데이터가 잎 병해뿐)."
                    )
                else:
                    label, prob, probs, cam, img = predict_with_cam(model, pil)

                    st.subheader(f"진단: {LABEL_KR[label]}  (확률 {prob:.1%})")
                    # 분석한 사진 + 판단 근거(Grad-CAM)를 같은 라인에 나란히 배치
                    icol, gcol = st.columns(2)
                    icol.image(pil, caption="🔍 분석한 사진", use_container_width=True)
                    gcol.image(overlay(img, cam), caption="Grad-CAM — 판단 근거(붉을수록 주목한 영역)",
                               use_container_width=True)
                    if label != "normal":
                        st.warning(f"{LABEL_KR[label]}이(가) 의심됩니다. 위 히트맵의 붉은 영역(병반 추정)을 확인하세요.")
                    else:
                        st.success("정상으로 판단됩니다.")

                    st.markdown("**클래스별 확률**")
                    st.bar_chart({c: float(p) for c, p in zip([LABEL_KR[c] for c in CLASSES], probs)})
                    st.caption("⚠️ Grad-CAM은 보조 지표 — 잎맥·배경 등 비병변 영역에 반응할 수 있어 사람 검수가 필요합니다.")
            else:
                st.info("잎 사진을 업로드하거나 아래 예시 사진을 클릭하면 진단 결과와 Grad-CAM 히트맵이 표시됩니다.")

            # 예시 사진 갤러리 — 결과 아래에 배치(사진 없는 외부 체험용, 클릭하면 위 결과가 갱신됨)
            if samples:
                st.divider()
                st.markdown("**예시 사진으로 바로 체험** (클릭하면 위쪽에 결과가 표시됩니다)")
                for col, (c, p) in zip(st.columns(len(samples)), samples):
                    col.image(str(p), caption=LABEL_KR[c], use_container_width=True)
                    if col.button("이 사진으로 진단", key=f"samp_{c}", use_container_width=True):
                        st.session_state["diag_sample"] = c
                        st.toast(f"예시 ‘{LABEL_KR[c]}’ 선택 — 위쪽 결과를 확인하세요", icon="🔬")
                        st.rerun()

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

            # YOLO 전용 예시 — 진단용 클로즈업과 달리 '잎이 여러 개 보이는 장면'(yolo_*.jpg)
            det_samples = [(c, SAMPLE_DIR / f"yolo_{c}.jpg") for c in SAMPLE_CLASSES]
            det_samples = [(c, p) for c, p in det_samples if p.exists()]

            # 입력 결정: 업로드 우선(샘플 선택 해제), 없으면 선택한 예시 사용
            if up2:
                st.session_state.pop("detect_sample", None)
                pil2 = Image.open(up2)
            elif st.session_state.get("detect_sample"):
                pil2 = Image.open(SAMPLE_DIR / f"yolo_{st.session_state['detect_sample']}.jpg")
            else:
                pil2 = None

            if pil2 is not None:
                st.divider()
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
                st.info("잎 사진을 업로드하거나 아래 예시 사진을 클릭하면 잎 위치 박스와 정상/질병 라벨이 표시됩니다.")

            # 예시 사진 갤러리 — 결과 아래에 배치(클릭하면 위쪽 검출 결과가 갱신됨)
            if det_samples:
                st.divider()
                st.markdown("**예시 사진으로 바로 체험** (클릭하면 위쪽에 검출 결과가 표시됩니다)")
                for col, (c, p) in zip(st.columns(len(det_samples)), det_samples):
                    col.image(str(p), caption=LABEL_KR[c], use_container_width=True)
                    if col.button("이 사진으로 검출", key=f"det_{c}", use_container_width=True):
                        st.session_state["detect_sample"] = c
                        st.toast(f"예시 ‘{LABEL_KR[c]}’ 선택 — 위쪽 검출 결과를 확인하세요", icon="🎯")
                        st.rerun()


if __name__ == "__main__":
    # 단독 실행 시에만 페이지 설정(멀티페이지에선 엔트리가 담당)
    st.set_page_config(page_title="토마토 잎 진단 (Phase 2 DL)", page_icon="🍅")
    render()
