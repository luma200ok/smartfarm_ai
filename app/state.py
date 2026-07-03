"""
session_state 키 상수 + 가상센서 헬퍼 — 이슈 #10 Streamlit 앱 전면 정리.

네임스페이스 형식: "{페이지}.{용도}" (예: K_LAST_DIAG = "presc.last_diag").
vsensor 생성 로직은 구 phase3_llm.py에서 문자 그대로 이관(리스크: 순서·조건 변경 금지).
"""
import streamlit as st

# ── prescribe.py ──
K_LAST_DIAG = "presc.last_diag"
K_LAST_PRESC = "presc.last_presc"

# ── monitor.py ──
K_VSENSOR = "monitor.vsensor"
K_LAST_WARNING = "monitor.last_warning"
K_WEATHER_QA_ANSWER = "monitor.weather_qa_answer"

# ── monitor.py — 관제형 대시보드(이슈 #17) ──
K_SETPOINTS = "monitor.setpoints"
K_DEVICE_STATES = "monitor.device_states"
K_CONTROL_LOG = "monitor.control_log"
K_CONTROL_ACTIVE = "monitor.control_active_alerts"
K_CONTROL_LAST_DATE = "monitor.control_last_date"
K_RECENT_READINGS = "monitor.recent_readings"
K_DISCORD_CONTROL_TOGGLE = "monitor.discord_control_toggle"
K_CONTROL_YEAR = "monitor.control_year"   # 연도별 제어 상태 리셋 판단용(이슈 #17 P2-1)
K_LIVE_DEVICE_STATES = "monitor.live_device_states"  # 오늘 운영 탭 전용 장치 상태(이슈 #25) — 시뮬용(K_DEVICE_STATES)과 분리


def get_vsensor(year):
    """연도별 VirtualSensor 인스턴스 확보(캐시 재사용) — 구 phase3의 vsensor 생성 로직 그대로 이관.

    반환: (vs, error) — 실패 시 vs=None, error=예외 메시지 문자열.
    """
    from sim.virtual_sensor import VirtualSensor

    vs = st.session_state.get(K_VSENSOR)
    if vs is None or vs.year != year:
        try:
            vs = VirtualSensor(year)
            st.session_state[K_VSENSOR] = vs
        except ValueError as e:                 # 그 작기에 7일↑ 시계열 없음
            return None, str(e)
    return vs, None
