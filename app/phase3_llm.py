"""
Phase 3 (LLM) — 플레이스홀더 페이지

아직 미구현. Phase 3 로드맵만 안내(발표 슬라이드 16번과 동일 톤).
구현 시작 시 이 파일의 render() 를 실제 처방 UI로 채운다.
멀티페이지: app/streamlit_app.py 가 render() 를 호출.
"""
import streamlit as st


def render():
    st.title("💬 Phase 3 · LLM — 자연어 처방 + 알림")
    st.info("🚧 준비 중 — Phase 2(DL)까지 완료. 아래는 다음 단계 로드맵입니다.")

    st.markdown("#### 처리 흐름")
    st.markdown(
        "> **CNN 진단 + LSTM 예측 + 재배가이드(RAG) → LLM 자연어 처방 → 🔔 알림**"
    )

    st.markdown("#### 단계")
    st.markdown(
        "- **3-1 Claude API** — 진단·예측 숫자/라벨 → 자연어 처방 생성\n"
        "- **3-2 RAG** — 농사로 재배가이드 검색 → 근거 있는 조언\n"
        "- **3-3 통합 파이프라인** — ML/LSTM 예측 + CNN 진단 + RAG → 처방 통합\n"
        "- **3-4 알림·대시보드** — 텔레그램 알림봇 + Streamlit 통합 대시보드"
    )

    st.markdown("#### 처방 예시 (목표 출력)")
    st.success(
        "🔬 잎곰팡이병 의심(87%) — 감염 잎 제거·습도↓  ·  "
        "💧 토양수분 30% 낮음 — 관수  ·  "
        "🌡️ 2시간 뒤 32℃ — 환기 준비"
    )

    st.caption("진행 상황은 레포 `docs/roadmap.md` Phase 3 섹션에서 갱신됩니다.")


if __name__ == "__main__":
    st.set_page_config(page_title="Phase 3 · LLM", page_icon="💬", layout="wide")
    render()
