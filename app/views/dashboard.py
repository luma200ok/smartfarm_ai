"""
농장 대시보드 페이지 — 서비스 그룹 기본(default) 페이지. (C4에서 상세 구현 예정)
"""
import streamlit as st

from ui import page_header


def render():
    page_header("🏠 농장 대시보드", "핵심 지표·경보·기능 바로가기 — 준비 중입니다.")


if __name__ == "__main__":
    st.set_page_config(page_title="농장 대시보드", page_icon="🏠", layout="wide")
    render()
