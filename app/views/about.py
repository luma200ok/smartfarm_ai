"""
프로젝트 개요·성과 페이지 — [프로젝트 기록] 그룹. 구 app/home.py(대문) 내용을 이관(이슈 #10 C3).

ML→DL→LLM 통합 성과 요약. Phase별로 묶고, 각 Phase의 핵심 결과·대표 그림을 바로 보여준다.
"Phase N" 명칭은 이 페이지(프로젝트 기록) 내부에서만 허용.
"""
from pathlib import Path

import streamlit as st

from ui import page_header  # 페이지 헤더 배너 통일(#76) — streamlit_app이 app/을 sys.path에 올려둠

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figures"
REPO = "https://github.com/luma200ok/smartfarm_ai"


def _img(rel, caption=None):
    """docs/figures/<rel> 이미지를 안전하게 표시(없으면 조용히 건너뜀)."""
    p = FIGS / rel
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def render():
    # ── Hero ──
    page_header(
        "📈 프로젝트 개요·성과",
        "환경 센서 + 잎 사진을 학습해 관수·병해 진단·환기를 처방하는 멀티모달 AI. "
        "한 작물(토마토)을 ML→DL→LLM으로 관통하며 단계마다 새 모달리티(정형→이미지→언어)를 도입했어요.",
    )

    # ── 핵심 지표 스트립(맨 위에 바로) ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ML 작물분류 (test F1)", "0.68", help="정직한 일반화 GroupKFold F1 0.49")
    m2.metric("DL 잎진단 (val acc)", "0.97", help="서빙 ResNet18 · ROC-AUC 0.997 · 백본 best mobilenet_v2 0.987")
    m3.metric("YOLO 검출 (mAP@50)", "0.78")
    m4.metric("LSTM 예측 (MAE)", "1.18℃", "baseline 1.25℃", delta_color="inverse")

    st.divider()

    # ════════════════ Phase 1 · ML ════════════════
    with st.container(border=True):
        h, b = st.columns([0.75, 0.25])
        h.subheader("🌱 Phase 1 · ML — 환경 센서로 작물 9종 분류")
        b.success("✅ 완료")
        st.caption("농진청 스마트팜 현장 농가 데이터(2022~24 다년) · 288만 시간별 → 11.6만 일별 집계")

        left, right = st.columns([0.52, 0.48])
        with left:
            k1, k2, k3 = st.columns(3)
            k1.metric("test F1", "0.68")
            k2.metric("GroupKFold F1", "0.49")
            k3.metric("다년 효과", "+0.073")
            st.markdown(
                "- **데이터 누수 실증** — 랜덤 분리 F1 **0.67** vs 농가 단위(GroupKFold) **0.49**. "
                "정직한 일반화 성능은 0.49예요.\n"
                "- **데이터 양 효과** — 단년→다년(3.5배)으로 공통 8작물 F1 **+0.073**, 누수 격차 36%p→18%p 완화, 수박 신규 커버.\n"
                "- 트리 부스팅(XGBoost)이 선형 대비 압도 — 환경↔작물은 비선형이에요."
            )
        with right:
            _img("phase1_ml/confusion_matrix.png", "작물별 혼동행렬 (XGBoost)")

        with st.expander("더 보기 — 데이터 양 효과 · 모델 비교"):
            c1, c2 = st.columns(2)
            with c1:
                _img("phase1_ml/year_compare.png", "단년 vs 다년 — 데이터 양 효과")
            with c2:
                _img("phase1_ml/model_compare.png", "모델 3종 비교")
        st.caption("👈 «작물 환경 추천» 페이지에서 환경값 슬라이더 데모를 실행하거나 «ML 실험 기록»에서 상세 평가를 확인하세요  ·  "
                   f"[📄 수행내역서]({REPO}/blob/main/docs/phase1_ml.md)")

    # ════════════════ Phase 2 · DL ════════════════
    with st.container(border=True):
        h, b = st.columns([0.75, 0.25])
        h.subheader("🍃 Phase 2 · DL — 잎 병해 진단 + 위치 검출 + 환경 시계열")
        b.success("✅ 완료")
        st.caption("AI Hub 071 + PlantVillage 토마토 잎 4분류(정상·잎곰팡이병·황화잎말이·잎마름역병) · 전이학습 · 설명가능 AI")

        left, right = st.columns([0.52, 0.48])
        with left:
            k1, k2, k3 = st.columns(3)
            k1.metric("4분류 acc", "0.96", help="ROC-AUC 0.997 · 잎마름역병 포함")
            k2.metric("YOLO mAP@50", "0.78")
            k3.metric("부위 게이트", "0.932")
            st.markdown(
                "- **전이학습 + 데이터 정제** — 원천 정상에서 잎(area3)만 선별, 백본을 **MLflow**로 비교"
                "(mobilenet_v2 0.987·서빙 resnet18 0.971).\n"
                "- **Grad-CAM 설명** + **YOLO 위치 검출** — 진단 → 근거 → 위치.\n"
                "- **2단 게이트** — 식물(plant_score) + 부위 분류기(0.932)로 과육·비잎 오진 차단.\n"
                "- **다변량 LSTM** — 8변수·485개 다년 시계열로 baseline(1.25℃) 추월(**1.18℃**)."
            )
        with right:
            _img("phase2_dl/06_gradcam.png", "Grad-CAM — 모델의 판단 근거(붉을수록 주목)")

        with st.expander("더 보기 — 평가 · YOLO 검출 · 시계열"):
            c1, c2, c3 = st.columns(3)
            with c1:
                _img("phase2_dl/09_eval.png", "혼동행렬 · ROC/AUC 0.997")
            with c2:
                _img("phase2_dl/10_yolo_detect.png", "YOLO 병해 잎 위치 검출")
            with c3:
                _img("phase2_dl/08_lstm_forecast.png", "LSTM 다음날 온도 예측")
        st.caption("👈 «잎 병해 진단» 페이지에서 잎 사진 업로드 → 진단+Grad-CAM · YOLO 검출 데모를 실행하거나 «DL 실험 기록»에서 상세 평가를 확인하세요  ·  "
                   f"[📄 수행내역서]({REPO}/blob/main/docs/phase2_dl.md)")

    # ════════════════ Phase 3 · LLM ════════════════
    with st.container(border=True):
        h, b = st.columns([0.75, 0.25])
        h.subheader("💬 Phase 3 · LLM — 진단·예측을 자연어 처방으로")
        b.success("✅ 완료")
        st.caption("CNN 진단 + LSTM 예측 + 재배가이드(RAG) → LLM 자연어 처방 → 디스코드 알림 · 날씨 인지 모니터링")

        left, right = st.columns([0.52, 0.48])
        with left:
            st.markdown(
                "- **Ollama(qwen2.5:14b)** — 진단·예측 숫자/라벨 → 자연어 처방 생성\n"
                "- **RAG(bge-m3)** — 농사로 재배가이드 검색 → 근거 있는 조언\n"
                "- **날씨 인지 모니터링** — 기대값 회귀(MAE 1.11℃) 기반 원인 구분 경보·사전 경보\n"
                "- **알림·대시보드** — 디스코드 Webhook 알림 + Streamlit 통합 대시보드"
            )
        with right:
            st.markdown("**처방 예시**")
            st.success(
                "잎곰팡이병 의심(87%) — 감염 잎 제거·습도 낮추기\n\n"
                "토양수분 30% 낮음 — 관수\n\n"
                "2시간 뒤 32℃ — 환기 준비"
            )
        st.caption("👈 «AI 처방»·«환경 관제» 페이지에서 데모를 실행하세요  ·  "
                   f"[📄 로드맵]({REPO}/blob/main/docs/roadmap.md)")

    st.divider()
    st.caption(
        f"🔗 [GitHub 레포]({REPO})  ·  "
        f"[README]({REPO}/blob/main/README.md)  ·  "
        f"[로드맵]({REPO}/blob/main/docs/roadmap.md)  ·  "
        "👈 왼쪽 사이드바 «서비스» 메뉴에서 인터랙티브 데모를 실행하세요."
    )


if __name__ == "__main__":
    st.set_page_config(page_title="프로젝트 개요·성과", page_icon="📈", layout="wide")
    render()
