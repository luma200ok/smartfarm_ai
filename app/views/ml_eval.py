"""
ML 실험 기록 페이지 — [프로젝트 기록] 그룹. 모델 평가·데이터 누수 실증 + EDA.

구 app/phase1_ml.py 탭3(모델 평가)·탭4(EDA)를 그대로 이관(이슈 #10 C3).
"Phase N" 명칭은 이 페이지(프로젝트 기록) 내부에서만 허용.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "phase1_crop_env_clf.pkl"
FIG = ROOT / "docs" / "figures" / "phase1_ml"


@st.cache_resource
def load_payload():
    import joblib
    return joblib.load(MODEL_PATH)


def tab_eval(payload):
    model_name = payload.get("model_name", "XGBoost")
    st.subheader("모델 3종 비교 (2022~2024 다년 · 9작물)")
    st.table(pd.DataFrame({
        "모델": ["LogisticRegression", "RandomForest", f"**{model_name} (베스트)**"],
        "Test Accuracy": ["0.38", "0.75", "0.72"],
        "Test F1(macro)": ["0.33", "0.70", "0.68"],
    }))

    img_compare = FIG / "model_compare.png"
    if img_compare.exists():
        st.image(str(img_compare), caption="모델별 Accuracy / F1(macro) 비교")

    st.subheader("⭐ 평가 3겹 — 데이터 누수의 교훈")
    st.table(pd.DataFrame({
        "평가 방법": [
            "① Test set (stratify)",
            "② StratifiedKFold(5) — 낙관적",
            "③ GroupKFold(연도+농가+작기) — 현실적",
        ],
        "F1(macro)": ["0.68", "0.67", "**0.49**"],
        "의미": [
            "단일 분할",
            "같은 농가가 train·test에 섞임 → 과대평가",
            "처음 보는 농가로 평가 → 진짜 일반화",
        ],
    }))
    st.info(
        "랜덤 분리 0.67 vs 농가 단위 분리 0.49 — **18%p 격차**.\n\n"
        "모델이 '농가·작기 고유 패턴'을 외워 성능을 부풀려요. "
        "새 농가에 대한 진짜 일반화 성능은 **0.49**가 정직한 값이에요.\n\n"
        "단년(2022)일 땐 0.77 vs 0.41(36%p)로 더 컸으나, **다년 결합으로 18%p까지 완화**됐어요."
    )

    st.subheader("혼동행렬 · 피처 중요도")
    img_cm  = FIG / "confusion_matrix.png"
    img_imp = FIG / "feature_importance.png"
    if img_cm.exists():
        st.image(str(img_cm), caption=f"혼동행렬 — {model_name}")
    if img_imp.exists():
        st.image(str(img_imp), caption="피처 중요도 (RandomForest 기준)")

    st.subheader("단년 vs 다년 — 데이터 양 효과")
    img_yc = FIG / "year_compare.png"
    if img_yc.exists():
        st.image(str(img_yc),
                 caption="2022 단년 → 2022~24 다년(데이터 3.5배): 공통 8작물 recall 비교 — "
                         "macro-F1 0.44→0.51(+0.073), 데이터 적던 작물일수록 향상 + 수박 신규 커버")


def tab_eda():
    st.subheader("탐색적 데이터 분석 (EDA)")

    eda_items = [
        ("eda_class_distribution.png",
         "작물별 표본 수. 방울토마토(25,241) ~ 수박(1,023) 약 25배 불균형 → F1(macro) 평가 중심."),
        ("eda_feature_distributions.png",
         "피처 8종 분포. 내부 온·습도는 작물별로 제어되는 값이라 분포 폭이 다양해요."),
        ("eda_correlation.png",
         "피처 상관 히트맵. 내부 온도 통계끼리 높은 상관(0.8+), CO2는 상대적으로 독립적이에요."),
        ("eda_crop_env_compare.png",
         "작물별 환경 평균 비교. 참외·딸기는 CO2 낮고 일사량 높음, 파프리카는 온도 낮음 등 작물별 차이 확인."),
    ]

    for fname, caption in eda_items:
        p = FIG / fname
        if p.exists():
            st.image(str(p), caption=caption)
        else:
            st.warning(f"{fname} 파일 없음 — `python src/ml/eda.py` 먼저 실행하세요.")


def render():
    st.title("📊 ML 실험 기록")
    st.caption(
        "환경 센서(온·습도·CO2·일사량) → XGBoost로 재배 작물 9종 예측(Phase 1) 실험 기록 · "
        "농진청 스마트팜 현장 데이터(2022~2024) · test F1 0.68 · GroupKFold F1 0.49"
    )

    if not MODEL_PATH.exists():
        st.caption("ℹ️ ML 실험 기록 사용 불가 — 모델 파일(models/phase1_crop_env_clf.pkl) 없음")
        return

    payload = load_payload()
    tab1, tab2 = st.tabs(["📊 모델 평가", "📑 EDA"])
    with tab1:
        tab_eval(payload)
    with tab2:
        tab_eda()


if __name__ == "__main__":
    st.set_page_config(page_title="ML 실험 기록", page_icon="📊", layout="wide")
    render()
