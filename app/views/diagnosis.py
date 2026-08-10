"""
잎 병해 진단 페이지 — 토마토 잎 사진 → 진단(+Grad-CAM) · 위치 검출(YOLO)

탭1 진단: resnet18(전이학습) 추론 → 정상/질병 + '어디를 보고 판단했나' Grad-CAM 히트맵
탭2 검출: YOLOv8n → 장면에서 잎을 찾아 박스 + 정상/질병 라벨 + 신뢰도
멀티페이지: app/streamlit_app.py 가 render() 를 호출(set_page_config 는 엔트리에서 1회).
단독 실행: streamlit run app/views/diagnosis.py   (프로젝트 루트에서)

추론(진단·Grad-CAM·OOD/부위 게이트·검출)은 src/dl/infer.py(streamlit 비의존 순수 함수)로
흡수됐다(이슈 #59) — 이 파일은 UI 렌더 + st.cache_resource 모델 로더 래핑만 담당한다.

모델: models/tomato_resnet18.pt (진단) · models/tomato_yolov8n.pt (검출)
  없으면 → prepare_tomato.py → 02_core.py --chunk 2-5 / prepare_tomato_yolo.py → 05_detect.py
"""
import sys
from pathlib import Path

from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CKPT = ROOT / "models" / "tomato_resnet18.pt"
YOLO_CKPT = ROOT / "models" / "tomato_yolov8n.pt"
SAMPLE_DIR = ROOT / "app" / "samples"            # 예시 사진(사진 없는 외부 체험용)
SAMPLE_CLASSES = ["normal", "leaf_mold", "tylcv", "late_blight"]

LABEL_KR = {"late_blight": "🦠 잎마름역병", "leaf_mold": "🦠 잎곰팡이병",
            "normal": "🌿 정상", "tylcv": "🦠 황화잎말이바이러스"}
PART_KR = {"flower": "🌼 꽃", "fruit": "🍅 과실(열매)", "leaf": "🌿 잎", "stem": "🌱 줄기"}


@st.cache_resource
def _load_model():
    """진단 모델(resnet18) 로더 래핑 — 실제 로드·캐시는 infer.load_diagnosis_model()(lru_cache)."""
    from dl import infer
    return infer.load_diagnosis_model()


@st.cache_resource
def _load_yolo():
    """검출 모델(YOLOv8) 로더 래핑 — 실제 로드·캐시는 infer.load_yolo()(lru_cache)."""
    from dl import infer
    return infer.load_yolo()


def overlay(img, cam):
    """원본 위에 jet 히트맵을 반투명 합성."""
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return (0.55 * img + 0.45 * heat).clip(0, 1)


# ── 페이지 렌더 (멀티페이지 엔트리가 호출) ───────────────────────────────────
def render():
    from dl import infer

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
            _load_model()
            st.caption("이 모델은 토마토 잎 사진 전용이에요. "
                       "업로드하면 먼저 잎/비잎을 판별해, 잎이 아닌 이미지는 진단하지 않아요.")
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
                score = infer.ood_plant_score(pil)
                if score < infer.PLANT_THRESHOLD:
                    pcol, _ = st.columns([1, 2])
                    pcol.image(pil, caption="🔍 분석한 사진", use_container_width=True)
                    st.error(
                        f"토마토 잎으로 보이지 않아요(잎·식물 신호 {score:.1%}). 진단을 진행하지 않아요.\n\n"
                        "이 진단기는 토마토 잎 전용이에요. 잎이 화면에 크게 보이도록 촬영해 업로드하세요."
                    )
                elif (part := infer.part_of(pil))[0] != "leaf":
                    # 식물이지만 잎이 아닌 부위(과실·꽃·줄기) → 잎 진단 차단(과육 오분류 방지)
                    pcol, _ = st.columns([1, 2])
                    pcol.image(pil, caption="🔍 분석한 사진", use_container_width=True)
                    st.error(
                        f"🚫 이 사진은 **{PART_KR[part[0]]}**(으)로 보입니다(부위 신뢰도 {part[1]:.0%}). "
                        "이 진단기는 **토마토 잎 전용**입니다 — 잎이 화면에 크게 보이도록 촬영해 업로드하세요.\n\n"
                        "※ 과실·꽃·줄기 병해는 현재 진단 범위가 아닙니다(학습 데이터가 잎 병해뿐)."
                    )
                else:
                    label, prob, probs, cam, img = infer.predict_with_cam(pil)

                    st.subheader(f"진단: {LABEL_KR[label]}  (확률 {prob:.1%})")
                    # 분석한 사진 + 판단 근거(Grad-CAM)를 같은 라인에 나란히 배치
                    icol, gcol = st.columns(2)
                    icol.image(pil, caption="🔍 분석한 사진", use_container_width=True)
                    gcol.image(overlay(img, cam), caption="Grad-CAM — 판단 근거(붉을수록 주목한 영역)",
                               use_container_width=True)
                    if label != "normal":
                        st.warning(f"{LABEL_KR[label]}이(가) 의심돼요. 위 히트맵의 붉은 영역(병반 추정)을 확인하세요.")
                    else:
                        st.success("정상으로 판단돼요.")

                    st.markdown("**클래스별 확률**")
                    st.bar_chart({c: float(p) for c, p in zip([LABEL_KR[c] for c in infer.CLASSES], probs)})
                    st.caption("Grad-CAM은 보조 지표예요 — 잎맥·배경 등 비병변 영역에 반응할 수 있어 사람 검수가 필요해요.")
            else:
                st.info("잎 사진을 업로드하거나 아래 예시 사진을 클릭하면 진단 결과와 Grad-CAM 히트맵을 볼 수 있어요.")

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
            _load_yolo()
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
                annotated, dets = infer.detect_annotated(pil2, conf=conf)
                st.image(annotated, caption="YOLO 검출 — 박스 위치 + 정상/질병 + 신뢰도",
                         use_container_width=True)
                if dets:
                    st.subheader(f"검출 {len(dets)}건")
                    for lab, c in dets:
                        st.write(f"- {LABEL_KR.get(lab, lab)} — 신뢰도 {c:.1%}")
                else:
                    st.info("임계값 이상으로 검출된 잎이 없어요. conf 슬라이더를 낮춰 보세요.")
            else:
                st.info("잎 사진을 업로드하거나 아래 예시 사진을 클릭하면 잎 위치 박스와 정상/질병 라벨을 볼 수 있어요.")

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
    st.set_page_config(page_title="잎 병해 진단", page_icon="🍅")
    render()
