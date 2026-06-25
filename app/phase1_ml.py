"""
Phase 1 (ML) — Streamlit 데모: 환경 센서 → 재배 작물 8종 분류

흐름: 슬라이더 입력 → 저장된 XGBoost 묶음(.pkl) 로드 → 예측(정수→작물명 매핑) → 결과 표시
실행: streamlit run app/phase1_ml.py  (프로젝트 루트에서)

⚠️ XGBoost는 LabelEncoder로 인코딩된 정수 y로 학습됨.
   predict() → 정수 인덱스 → payload["labels"][idx] 로 작물명 변환 필수.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "phase1_crop_env_clf.pkl"
FIG = ROOT / "docs" / "figures" / "phase1_ml"

FEATURES = [
    "온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
    "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균",
]

# 슬라이더 표시용 레이블
FEAT_LABELS = {
    "온도내부_평균":     "내부 온도 평균 (°C)",
    "온도내부_최저":     "내부 온도 최저 (°C)",
    "온도내부_최고":     "내부 온도 최고 (°C)",
    "온도내부_표준편차":  "내부 온도 표준편차",
    "습도내부_평균":     "내부 습도 평균 (%)",
    "co2_평균":         "CO2 평균 (ppm)",
    "온도외부_평균":     "외부 온도 평균 (°C)",
    "일사량_평균":       "일사량 평균",
}


# ── 데이터·모델 캐시 ──────────────────────────────────────────────────────
@st.cache_resource
def load_payload():
    return joblib.load(MODEL_PATH)


# 슬라이더 범위·작물별 통계는 모델 pkl에 동봉돼 있다(payload["ranges"]·["crop_mean"]…).
# → 배포 환경에 원본 csv가 없어도 데모가 자립 동작.


# ── 공통 CSS (초록 테마) ──────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 12px 24px;
        background-color: #F1F8E9;
        color: #2E5A1C;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4C9A2A;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── 탭1 🔮 예측 ──────────────────────────────────────────────────────────
def tab_predict(payload):
    st.subheader("환경값을 입력하면 재배 중일 작물을 추천합니다.")
    ranges = payload["ranges"]
    crop_mean = payload["crop_mean"]

    model  = payload["model"]
    labels = payload["labels"]   # sorted 작물명 리스트 — XGBoost 정수 → 이름 매핑 키

    cols = st.columns(2)
    values = []
    for i, feat in enumerate(FEATURES):
        lo, hi, med = ranges[feat]
        with cols[i % 2]:
            v = st.slider(FEAT_LABELS[feat], float(lo), float(hi), float(med), key=f"slider_{feat}")
        values.append(v)

    if st.button("작물 추천 받기", type="primary"):
        X = pd.DataFrame([values], columns=FEATURES)

        # XGBoost: predict() → 정수 인덱스 → labels 로 변환
        raw_pred = model.predict(X)[0]
        if isinstance(raw_pred, (int, np.integer)):
            pred_crop = labels[int(raw_pred)]
        else:
            # RandomForest 등 문자열 직접 반환 시 그대로 사용
            pred_crop = str(raw_pred)

        st.success(f"### 🌾 예측 작물 : **{pred_crop}**")

        # 신뢰도 Top 3
        proba = model.predict_proba(X)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        st.subheader("예측 신뢰도 Top 3")
        rows = []
        for idx in top3_idx:
            rows.append({"작물": labels[idx], "신뢰도": f"{proba[idx]:.1%}"})
        st.table(pd.DataFrame(rows).set_index("작물"))

        # 내 입력 vs 예측 작물 평균 비교
        if pred_crop in crop_mean.index:
            st.subheader(f"💡 내 입력 vs {pred_crop} 평균 환경")
            typical = crop_mean.loc[pred_crop]
            cmp = pd.DataFrame(
                {"내 입력": values,
                 f"{pred_crop} 평균": [round(typical[f], 2) for f in FEATURES]},
                index=[FEAT_LABELS[f] for f in FEATURES],
            )
            st.table(cmp)
            st.caption("두 값이 비슷할수록 그 작물에 적합한 환경입니다.")


# ── 탭2 🌾 작물별 환경 가이드 ────────────────────────────────────────────
def tab_guide(payload):
    st.subheader("🌾 작물별 적합 환경 가이드")
    st.caption("작물을 선택하면 해당 작물의 환경 피처 평균·최소·최대를 보여줍니다.")

    crop_mean, crop_min, crop_max = payload["crop_mean"], payload["crop_min"], payload["crop_max"]
    crops = sorted(crop_mean.index.tolist())
    sel = st.selectbox("작물 선택", crops)

    tbl = pd.DataFrame(
        {
            "평균": [round(crop_mean.loc[sel, f], 2) for f in FEATURES],
            "최소": [round(crop_min.loc[sel, f], 2) for f in FEATURES],
            "최대": [round(crop_max.loc[sel, f], 2) for f in FEATURES],
        },
        index=[FEAT_LABELS[f] for f in FEATURES],
    )
    st.table(tbl)


# ── 탭3 📊 모델 평가 ─────────────────────────────────────────────────────
def tab_eval(payload):
    model_name = payload.get("model_name", "XGBoost")
    st.subheader("모델 3종 비교")
    st.table(pd.DataFrame({
        "모델": ["LogisticRegression", "RandomForest", f"**{model_name} (베스트)**"],
        "Test Accuracy": ["0.41", "0.80", "0.80"],
        "Test F1(macro)": ["0.37", "0.77", "0.78"],
    }))

    img_compare = FIG / "model_compare.png"
    if img_compare.exists():
        st.image(str(img_compare), caption="모델별 Accuracy / F1(macro) 비교")

    st.subheader("⭐ 평가 3겹 — 데이터 누수의 교훈")
    st.table(pd.DataFrame({
        "평가 방법": [
            "① Test set (stratify)",
            "② StratifiedKFold(5) — 낙관적",
            "③ GroupKFold(농가+작기) — 현실적",
        ],
        "F1(macro)": ["0.78", "0.77", "**0.41**"],
        "의미": [
            "단일 분할",
            "같은 농가가 train·test에 섞임 → 과대평가",
            "처음 보는 농가로 평가 → 진짜 일반화",
        ],
    }))
    st.info(
        "🔑 랜덤 분리 0.77 vs 농가 단위 분리 0.41 — **36%p 폭락**.\n\n"
        "모델이 '농가·작기 고유 패턴'을 외워 성능을 부풀린다. "
        "새 농가에 대한 진짜 일반화 성능은 **0.41**이 정직한 값."
    )

    st.subheader("혼동행렬 · 피처 중요도")
    img_cm  = FIG / "confusion_matrix.png"
    img_imp = FIG / "feature_importance.png"
    if img_cm.exists():
        st.image(str(img_cm), caption=f"혼동행렬 — {model_name}")
    if img_imp.exists():
        st.image(str(img_imp), caption="피처 중요도 (RandomForest 기준)")


# ── 탭4 📑 EDA ───────────────────────────────────────────────────────────
def tab_eda():
    st.subheader("탐색적 데이터 분석 (EDA)")

    eda_items = [
        ("eda_class_distribution.png",
         "작물별 표본 수. 완숙토마토(7,920) ~ 가지(717) 약 11배 불균형 → F1(macro) 평가 중심."),
        ("eda_feature_distributions.png",
         "피처 8종 분포. 내부 온·습도는 작물별로 제어되는 값이라 분포 폭이 다양하다."),
        ("eda_correlation.png",
         "피처 상관 히트맵. 내부 온도 통계끼리 높은 상관(0.8+), CO2는 상대적으로 독립적."),
        ("eda_crop_env_compare.png",
         "작물별 환경 평균 비교. 참외·딸기는 CO2 낮고 일사량 높음, 파프리카는 온도 낮음 등 작물별 차이 확인."),
    ]

    for fname, caption in eda_items:
        p = FIG / fname
        if p.exists():
            st.image(str(p), caption=caption)
        else:
            st.warning(f"{fname} 파일 없음 — `python src/ml/eda.py` 먼저 실행하세요.")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="스마트팜 작물 추천 (Phase 1 ML)", page_icon="🌱", layout="wide")
    inject_css()

    st.title("🌱 스마트팜 작물 분류 데모 — Phase 1 ML")
    st.caption(
        "환경 센서(온·습도·CO2·일사량) 입력 → XGBoost로 재배 작물 8종 예측 | "
        "농진청 스마트팜 현장 데이터(2022) · test F1 0.78"
    )

    payload = load_payload()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 예측", "🌾 작물별 환경 가이드", "📊 모델 평가", "📑 EDA"]
    )
    with tab1:
        tab_predict(payload)
    with tab2:
        tab_guide(payload)
    with tab3:
        tab_eval(payload)
    with tab4:
        tab_eda()


if __name__ == "__main__":
    main()
