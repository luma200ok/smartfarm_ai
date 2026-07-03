"""
페이지 객체 레지스트리 — st.navigation(streamlit_app.py)과 st.page_link(각 view, 예:
dashboard.py 기능 바로가기 카드)가 동일한 st.Page 인스턴스를 공유하기 위한 단일 진실.

이슈 #10 P1-1 픽스: st.page_link에 파일 경로 문자열을 넘기면 streamlit_app.py가
st.Page(콜러블)로 등록한 페이지와 매칭되지 않아 StreamlitPageNotFoundError로 크래시한다.
반드시 이 모듈의 st.Page 객체를 그대로 넘겨야 한다.

순환 참조 방지: 이 모듈은 각 views/*.py의 render 함수를 임포트하지만, views/*.py는
이 모듈을 최상위(import 시점)에서 임포트하지 않는다(dashboard.py는 함수 내부에서 지연 임포트).
"""
import streamlit as st

from dashboard import render as _dashboard
from diagnosis import render as _diagnosis
from prescribe import render as _prescribe
from monitor import render as _monitor
from crops import render as _crops
from about import render as _about
from ml_eval import render as _ml_eval
from dl_eval import render as _dl_eval

PAGE_DASHBOARD = st.Page(_dashboard, title="농장 대시보드", icon="🏠", url_path="home", default=True)
PAGE_DIAGNOSIS = st.Page(_diagnosis, title="잎 병해 진단", icon="🔬", url_path="diagnosis")
PAGE_PRESCRIBE = st.Page(_prescribe, title="AI 처방", icon="💊", url_path="prescribe")
PAGE_MONITOR = st.Page(_monitor, title="환경 관제", icon="🌡️", url_path="monitor")
PAGE_CROPS = st.Page(_crops, title="작물 환경 추천", icon="🌾", url_path="crops")
PAGE_ABOUT = st.Page(_about, title="프로젝트 개요·성과", icon="📈", url_path="about")
PAGE_ML_EVAL = st.Page(_ml_eval, title="ML 실험 기록", icon="📊", url_path="ml-eval")
PAGE_DL_EVAL = st.Page(_dl_eval, title="DL 실험 기록", icon="📊", url_path="dl-eval")

NAV_GROUPS = {
    "서비스": [PAGE_DASHBOARD, PAGE_DIAGNOSIS, PAGE_PRESCRIBE, PAGE_MONITOR, PAGE_CROPS],
    "프로젝트 기록": [PAGE_ABOUT, PAGE_ML_EVAL, PAGE_DL_EVAL],
}
