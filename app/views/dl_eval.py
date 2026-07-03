"""
DL 실험 기록 페이지 — [프로젝트 기록] 그룹. 잎 진단·YOLO 검출·LSTM 예측 평가 그림.

구 app/home.py의 Phase 2 "더 보기" 섹션(혼동행렬·ROC·YOLO 검출·LSTM 예측)을 이관(이슈 #10 C3).
백본 비교는 MLflow로 추적(docs/_local/mlflow.md) — "Phase N" 명칭은 이 페이지 내부에서만 허용.
"""
from pathlib import Path

import streamlit as st

from ui import page_header, unavailable

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures" / "phase2_dl"


def _img(name, caption=None):
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.caption(f"ℹ️ {name} 사용 불가 — 그림 파일 없음")


def render():
    page_header(
        "📊 DL 실험 기록",
        "AI Hub 071 + PlantVillage 토마토 잎 4분류(정상·잎곰팡이병·황화잎말이·잎마름역병) 전이학습 실험 기록(Phase 2).",
    )

    st.subheader("혼동행렬 · ROC/AUC")
    _img("09_eval.png", "혼동행렬 · ROC/AUC 0.997")

    st.subheader("YOLO 검출")
    _img("10_yolo_detect.png", "YOLO 병해 잎 위치 검출")

    st.subheader("LSTM 다음날 온도 예측")
    _img("08_lstm_forecast.png", "LSTM 다음날 온도 예측(baseline 1.25℃ → 1.18℃)")

    st.divider()
    st.markdown(
        "- 백본 비교(mobilenet_v2 0.987 · 서빙 resnet18 0.971)는 **MLflow**로 실험 추적했어요.\n"
        "- 2단 게이트(식물 판별 + 부위 분류기 0.932)로 과육·비잎 오진을 차단했어요."
    )
    if not (ROOT / "mlruns").exists() and not (ROOT / "mlflow.db").exists():
        unavailable("MLflow 실험 UI 링크", "로컬 mlruns/mlflow.db 없음", "docs/_local/mlflow.md 참고")


if __name__ == "__main__":
    st.set_page_config(page_title="DL 실험 기록", page_icon="📊", layout="wide")
    render()
