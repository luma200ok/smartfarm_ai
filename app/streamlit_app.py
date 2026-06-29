"""
SmartFarm AI — Streamlit 멀티페이지 엔트리포인트

한 URL에서 홈 + Phase 1(ML) + Phase 2(DL) + Phase 3(LLM) 을 사이드바로 전환.
set_page_config 는 여기서 1회만 호출(각 페이지 render() 에서는 호출 금지).

실행:  streamlit run app/streamlit_app.py   (프로젝트 루트에서)
"""
import sys
from pathlib import Path

import streamlit as st

# 페이지 모듈을 직접 임포트할 수 있도록 app/ 를 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="SmartFarm AI", page_icon="🌱", layout="wide")

from home import render as home          # noqa: E402
from phase1_ml import render as ml       # noqa: E402
from phase2_dl import render as dl       # noqa: E402
from phase3_llm import render as llm     # noqa: E402

# 페이지 함수가 모두 render() 라 URL 경로 추론이 충돌 → url_path 명시 필수
nav = st.navigation([
    st.Page(home, title="홈", icon="🏠", url_path="home", default=True),
    st.Page(ml,   title="Phase 1 · ML 작물분류", icon="🌱", url_path="ml"),
    st.Page(dl,   title="Phase 2 · DL 잎병해",   icon="🍃", url_path="dl"),
    st.Page(llm,  title="Phase 3 · LLM 처방",    icon="💬", url_path="llm"),
])
nav.run()
