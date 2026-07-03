"""
SmartFarm AI — Streamlit 멀티페이지 엔트리포인트

st.navigation 2그룹 구성(이슈 #10):
  [서비스] 농장 대시보드 / 잎 병해 진단 / AI 처방 / 환경 관제 / 작물 환경 추천
  [프로젝트 기록] 프로젝트 개요·성과 / ML 실험 기록 / DL 실험 기록
set_page_config 는 여기서 1회만 호출(각 view의 render() 에서는 호출 금지).
페이지 객체(st.Page)는 app/nav.py가 단일 진실로 소유 — st.page_link에서도 동일 인스턴스를 참조한다.

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

from ui import inject_css   # noqa: E402
import nav                  # noqa: E402

inject_css()

st.navigation(nav.NAV_GROUPS).run()
