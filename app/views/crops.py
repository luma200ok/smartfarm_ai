"""
작물 환경 추천 페이지 — 환경 센서 → 재배 작물 9종 분류/가이드 (2022~2024 다년).

흐름: 슬라이더 입력 → 저장된 XGBoost 묶음(.pkl) 로드 → 예측(정수→작물명 매핑) → 결과 표시
멀티페이지: app/streamlit_app.py 가 render() 를 호출(set_page_config 는 엔트리에서 1회).
단독 실행: streamlit run app/views/crops.py  (프로젝트 루트에서)

⚠️ XGBoost는 LabelEncoder로 인코딩된 정수 y로 학습됨.
   predict() → 정수 인덱스 → payload["labels"][idx] 로 작물명 변환 필수.

구 app/phase1_ml.py 탭1(예측)·탭2(가이드)를 이관(이슈 #10 C3). 탭3·4(모델 평가·EDA)는
ml_eval.py(프로젝트 기록)로 분리.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from ui import page_header, unavailable

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "phase1_crop_env_clf.pkl"

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


# ── 탭1 🔮 예측 ──────────────────────────────────────────────────────────
def tab_predict(payload):
    st.subheader("환경값을 입력하면 재배 중일 작물을 추천해요.")
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
            st.subheader(f"내 입력 vs {pred_crop} 평균 환경")
            typical = crop_mean.loc[pred_crop]
            cmp = pd.DataFrame(
                {"내 입력": values,
                 f"{pred_crop} 평균": [round(typical[f], 2) for f in FEATURES]},
                index=[FEAT_LABELS[f] for f in FEATURES],
            )
            st.table(cmp)
            st.caption("두 값이 비슷할수록 그 작물에 적합한 환경이에요.")


# ── 탭2 🌾 작물별 환경 가이드 ────────────────────────────────────────────
def tab_guide(payload):
    st.subheader("🌾 작물별 적합 환경 가이드")
    st.caption("작물을 선택하면 해당 작물의 환경 피처 평균·최소·최대를 보여줘요.")

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


# ── 페이지 렌더 (멀티페이지 엔트리가 호출) ───────────────────────────────────
def render():
    page_header(
        "🌾 작물 환경 추천",
        "환경 센서(온·습도·CO2·일사량) 입력 → XGBoost로 재배 작물 9종 예측 | "
        "농진청 스마트팜 현장 데이터(2022~2024) · test F1 0.68 · GroupKFold F1 0.49",
    )

    if not MODEL_PATH.exists():
        unavailable("작물 환경 추천", "모델 파일(models/phase1_crop_env_clf.pkl) 없음")
        return

    payload = load_payload()

    tab1, tab2 = st.tabs(["🔮 예측", "🌾 작물별 환경 가이드"])
    with tab1:
        tab_predict(payload)
    with tab2:
        tab_guide(payload)


if __name__ == "__main__":
    # 단독 실행 시에만 페이지 설정(멀티페이지에선 엔트리가 담당)
    st.set_page_config(page_title="작물 환경 추천", page_icon="🌾", layout="wide")
    render()
