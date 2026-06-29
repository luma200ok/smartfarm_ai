"""
SmartFarm AI — 대문(홈) 페이지

프로젝트 한 줄 소개 + 3단계(ML→DL→LLM) 카드 + 핵심 지표.
멀티페이지: app/streamlit_app.py 가 render() 를 호출.
"""
import streamlit as st


def render():
    st.title("🌱 SmartFarm AI — 스마트팜 재배 도우미")
    st.caption(
        "환경 센서 + 잎 사진을 학습해 \"관수·병해 진단·환기\"를 처방하는 AI. "
        "하나의 도메인으로 ML→DL→LLM을 쌓되 단계마다 새 모달리티(정형→이미지→언어)를 도입합니다."
    )

    st.divider()

    # ── 3단계 카드 ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🌱 Phase 1 · ML")
        st.caption("환경 센서 → 작물 9종 분류")
        st.markdown(
            "- scikit-learn · XGBoost\n"
            "- test F1 **0.68**\n"
            "- GroupKFold F1 **0.49** (누수 교훈)"
        )
        st.success("완료")
    with c2:
        st.subheader("🍃 Phase 2 · DL")
        st.caption("잎 사진 진단 + 환경 시계열 예측")
        st.markdown(
            "- 전이학습 3분류 val **0.94**\n"
            "- YOLO 검출 mAP@50 **0.78**\n"
            "- LSTM MAE **1.18℃** < baseline"
        )
        st.success("완료")
    with c3:
        st.subheader("💬 Phase 3 · LLM")
        st.caption("진단·예측 → 자연어 처방 + 알림")
        st.markdown(
            "- Claude API · RAG\n"
            "- 통합 처방 파이프라인\n"
            "- 텔레그램 알림 · 대시보드"
        )
        st.info("예정")

    st.divider()

    # ── 핵심 지표 스트립 ──
    st.markdown("#### 핵심 지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ML 작물분류 (test F1)", "0.68", help="GroupKFold(현실적) F1 0.49")
    m2.metric("DL 3분류 (val acc)", "0.94", help="ROC-AUC 0.99")
    m3.metric("YOLO 검출 (mAP@50)", "0.78")
    m4.metric("LSTM 예측 (MAE)", "1.18℃", "baseline 1.25℃", delta_color="inverse")

    st.divider()
    st.markdown(
        "👈 왼쪽 사이드바에서 **Phase별 데모**를 둘러보세요.  \n"
        "자세한 내용은 레포 `README.md` · `docs/roadmap.md` 참고."
    )


if __name__ == "__main__":
    st.set_page_config(page_title="SmartFarm AI", page_icon="🌱", layout="wide")
    render()
