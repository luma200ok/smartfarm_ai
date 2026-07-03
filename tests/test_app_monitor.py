"""app/views/monitor.py — 페이지 렌더 스모크 테스트(P1-1 회귀 방지, 이슈 #23).

render_live_tab()이 KMA 조회 성공 경로를 타는 시점에 datetime.now() 등 monitor.py
모듈 심볼이 실제로 임포트돼 있는지 AppTest로 페이지 전체를 렌더해 검증한다
(단순 `import` 성공만으로는 render_live_tab() 내부 NameError를 못 잡는다 — 함수 실행까지
확인해야 함).
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MONITOR_PAGE = ROOT / "app" / "views" / "monitor.py"


def _forecast_success(date_str="20260703"):
    hourly = [{"date": date_str, "time": f"{h:02d}00", "temp": 25.0, "humidity": 65.0}
              for h in range(24)]
    return {"unavailable": False, "hourly": hourly, "daily": []}


@pytest.fixture
def _kma_success(monkeypatch):
    """오늘 운영 탭이 render_live_tab()의 KMA 성공 경로(cur = ... datetime.now() 사용)를
    반드시 타도록 llm.weather·llm.expect를 mock. app/·app/views/도 sys.path에 추가
    (monitor.py가 app/state.py·app/ui.py를 임포트하므로 — streamlit_app.py 진입점과 동일)."""
    import sys
    for p in (ROOT / "src", ROOT / "app", ROOT / "app" / "views"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from datetime import date as _date

    from llm import expect as expect_mod
    from llm import weather

    today_str = _date.today().strftime("%Y%m%d")
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: _forecast_success(today_str))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: None)  # 폴백 경로(외기 그대로)


def test_monitor_page_renders_without_exception_on_kma_success(_kma_success):
    """AppTest로 /monitor 스크립트 전체 실행 — render_live_tab()이 예외 없이 완주해야 한다
    (수정 전엔 `datetime` 미임포트로 NameError가 나던 지점)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(MONITOR_PAGE))
    at.run(timeout=30)

    assert not at.exception, f"페이지 렌더 중 예외 발생: {[str(e) for e in at.exception]}"


def test_monitor_module_defines_datetime_symbol_used_by_render_live_tab():
    """render_live_tab()이 참조하는 datetime 심볼이 모듈 전역에 실제로 바인딩돼 있는지 확인
    (import 라인 자체의 회귀를 가장 직접적으로 잡는 저비용 가드)."""
    import importlib
    import sys

    if str(ROOT / "app") not in sys.path:
        sys.path.insert(0, str(ROOT / "app"))
    if str(ROOT / "app" / "views") not in sys.path:
        sys.path.insert(0, str(ROOT / "app" / "views"))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    monitor_mod = importlib.import_module("monitor")
    import datetime as datetime_module
    assert monitor_mod.datetime is datetime_module.datetime
