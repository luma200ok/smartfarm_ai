"""
SmartFarm AI — Streamlit 멀티페이지 엔트리포인트

st.navigation 2그룹 구성(이슈 #10):
  [서비스] 농장 대시보드 / 잎 병해 진단 / AI 처방 / 환경 모니터링 / 작물 환경 추천
  [프로젝트 기록] 프로젝트 개요·성과 / ML 실험 기록 / DL 실험 기록
set_page_config 는 여기서 1회만 호출(각 view의 render() 에서는 호출 금지).

실행:  streamlit run app/streamlit_app.py   (프로젝트 루트에서)
"""
import sys
from pathlib import Path

import streamlit as st

# app/ 와 app/views/ 를 직접 임포트할 수 있도록 경로에 추가(app/views/ 자체는 st.navigation과
# 충돌하는 자동 멀티페이지 디렉터리가 아니라 일반 패키지 디렉터리)
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_APP_DIR / "views"))

st.set_page_config(page_title="SmartFarm AI", page_icon="🌱", layout="wide")

from ui import inject_css               # noqa: E402
from dashboard import render as dashboard   # noqa: E402
from diagnosis import render as diagnosis   # noqa: E402
from prescribe import render as prescribe   # noqa: E402
from monitor import render as monitor       # noqa: E402
from crops import render as crops           # noqa: E402
from about import render as about           # noqa: E402
from ml_eval import render as ml_eval       # noqa: E402
from dl_eval import render as dl_eval       # noqa: E402

inject_css()

nav = st.navigation({
    "서비스": [
        st.Page(dashboard, title="농장 대시보드",    icon="🏠", url_path="home", default=True),
        st.Page(diagnosis, title="잎 병해 진단",      icon="🔬", url_path="diagnosis"),
        st.Page(prescribe, title="AI 처방",          icon="💊", url_path="prescribe"),
        st.Page(monitor,   title="환경 모니터링",     icon="🌡️", url_path="monitor"),
        st.Page(crops,     title="작물 환경 추천",    icon="🌾", url_path="crops"),
    ],
    "프로젝트 기록": [
        st.Page(about,   title="프로젝트 개요·성과", icon="📈", url_path="about"),
        st.Page(ml_eval, title="ML 실험 기록",        icon="📊", url_path="ml-eval"),
        st.Page(dl_eval, title="DL 실험 기록",        icon="📊", url_path="dl-eval"),
    ],
})
nav.run()
