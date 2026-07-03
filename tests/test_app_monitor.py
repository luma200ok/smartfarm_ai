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


def test_live_trend_baseline_toggle_default_off_and_togglable(_kma_success):
    """이슈 #37 — "제어 없을 때와 비교해서 보기" 체크박스는 기본 꺼짐(False)이고,
    켜면 페이지가 예외 없이 재실행돼야 한다(비제어 기준선 라인 추가 렌더 경로)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(MONITOR_PAGE))
    at.run(timeout=30)
    assert not at.exception

    checkboxes = [cb for cb in at.checkbox if cb.key == "live_trend_show_baseline"]
    assert len(checkboxes) == 1
    assert checkboxes[0].value is False

    checkboxes[0].set_value(True).run(timeout=30)
    assert not at.exception


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


def _import_monitor_module():
    import importlib
    import sys

    for p in (ROOT / "app", ROOT / "app" / "views", ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("monitor")


def test_split_events_hour_equal_now_hour_is_done():
    """경계: hour==now_hour는 실행(done) 쪽으로 분류된다(이슈 #31)."""
    monitor_mod = _import_monitor_module()
    items = [{"hour": 8}, {"hour": 9}, {"hour": 10}]
    done, planned = monitor_mod._split_events(items, now_hour=9)
    assert [i["hour"] for i in done] == [8, 9]
    assert [i["hour"] for i in planned] == [10]


def test_split_events_empty_list():
    monitor_mod = _import_monitor_module()
    done, planned = monitor_mod._split_events([], now_hour=9)
    assert done == [] and planned == []


def test_split_events_now_hour_zero_midnight_boundary():
    """경계: 자정 직후(now_hour=0)엔 hour 0만 실행, 1시 이후는 전부 예정(이슈 #31)."""
    monitor_mod = _import_monitor_module()
    items = [{"hour": 0}, {"hour": 1}, {"hour": 2}]
    done, planned = monitor_mod._split_events(items, now_hour=0)
    assert [i["hour"] for i in done] == [0]
    assert [i["hour"] for i in planned] == [1, 2]


def test_split_events_all_done_or_all_planned():
    monitor_mod = _import_monitor_module()
    items = [{"hour": 0}, {"hour": 1}]
    done, planned = monitor_mod._split_events(items, now_hour=23)
    assert len(done) == 2 and planned == []

    done2, planned2 = monitor_mod._split_events(items, now_hour=-1)
    assert done2 == [] and len(planned2) == 2


# ── _split_control_segments — 실행/계획 분리(이슈 #35 리뷰 P2) ──────────────
def _long_df(hours, value_cols):
    """{hour: {col: val, ...}, ...} → melt된 {hour,구분,값} DataFrame."""
    import pandas as pd
    rows = [{"hour": h, **vals} for h, vals in hours.items()]
    df = pd.DataFrame(rows)
    return df.melt(id_vars=["hour"], value_vars=value_cols, var_name="구분", value_name="값")


def test_split_control_segments_now_hour_shared_by_both_segments():
    """now_hour(=2) 데이터 포인트가 실행/계획 양쪽 세그먼트에 공유돼(hour<=now, hour>=now)
    선이 끊기지 않고 이어진다."""
    monitor_mod = _import_monitor_module()
    value_cols = ["외기", "제어 후"]
    long_df = _long_df({h: {"외기": 20.0, "제어 후": 21.0 + h} for h in range(5)}, value_cols)
    out_df, _domain, _range = monitor_mod._split_control_segments(long_df, value_cols, "제어 후", now_hour=2)

    exec_rows = out_df[out_df["구분"] == "제어 후(실행)"]
    planned_rows = out_df[out_df["구분"] == "제어 후(계획)"]
    assert sorted(exec_rows["hour"].tolist()) == [0, 1, 2]      # hour<=now_hour
    assert sorted(planned_rows["hour"].tolist()) == [2, 3, 4]   # hour>=now_hour
    # now_hour(2) 값이 양쪽에 동일하게 존재(끊김 없이 이어짐)
    assert exec_rows[exec_rows["hour"] == 2]["값"].iloc[0] == pytest.approx(23.0)
    assert planned_rows[planned_rows["hour"] == 2]["값"].iloc[0] == pytest.approx(23.0)
    # 분리 대상이 아닌 계열("외기")은 원래 이름 그대로, 전체 hour 보유
    assert set(out_df[out_df["구분"] == "외기"]["hour"]) == {0, 1, 2, 3, 4}


def test_split_control_segments_boundary_now_hour_zero_and_last():
    """경계: now_hour=0이면 실행은 hour 0 하나뿐, 계획은 전체(0~N)."""
    monitor_mod = _import_monitor_module()
    value_cols = ["제어 후"]
    long_df = _long_df({h: {"제어 후": 20.0 + h} for h in range(4)}, value_cols)
    out_df, _domain, _range = monitor_mod._split_control_segments(long_df, value_cols, "제어 후", now_hour=0)
    exec_rows = out_df[out_df["구분"] == "제어 후(실행)"]
    planned_rows = out_df[out_df["구분"] == "제어 후(계획)"]
    assert sorted(exec_rows["hour"].tolist()) == [0]
    assert sorted(planned_rows["hour"].tolist()) == [0, 1, 2, 3]

    # now_hour이 마지막 시간이면 계획은 그 시간 하나뿐, 실행은 전체
    out_df2, _, _ = monitor_mod._split_control_segments(long_df, value_cols, "제어 후", now_hour=3)
    exec_rows2 = out_df2[out_df2["구분"] == "제어 후(실행)"]
    planned_rows2 = out_df2[out_df2["구분"] == "제어 후(계획)"]
    assert sorted(exec_rows2["hour"].tolist()) == [0, 1, 2, 3]
    assert sorted(planned_rows2["hour"].tolist()) == [3]


def test_split_control_segments_color_domain_range_length_matches():
    """범례 domain/range 길이가 항상 일치하고, split_col은 실행/계획 두 항목으로 대체되며
    실행=_EXEC_COLOR(빨강), 계획=_PLANNED_COLOR(회색) — 색으로 명확히 구분된다."""
    monitor_mod = _import_monitor_module()
    value_cols = ["외기", "제어 전(기준선)", "제어 후"]
    long_df = _long_df({h: {c: 20.0 for c in value_cols} for h in range(3)}, value_cols)
    out_df, domain, color_range = monitor_mod._split_control_segments(long_df, value_cols, "제어 후", now_hour=1)

    assert len(domain) == len(color_range)
    assert len(domain) == len(value_cols) + 1  # split_col 1개가 실행/계획 2개로 대체(+1)
    assert "제어 후(실행)" in domain and "제어 후(계획)" in domain
    assert "제어 후" not in domain
    idx_exec = domain.index("제어 후(실행)")
    idx_planned = domain.index("제어 후(계획)")
    assert color_range[idx_exec] == monitor_mod._EXEC_COLOR    # 실행=빨강
    assert color_range[idx_planned] == monitor_mod._PLANNED_COLOR  # 계획=회색
    assert set(out_df["구분"].unique()) == {"외기", "제어 전(기준선)", "제어 후(실행)", "제어 후(계획)"}


def test_split_control_segments_no_split_col_returns_unchanged():
    """split_col=None이면 long_df·domain·range가 원본 그대로(변경 없음)."""
    monitor_mod = _import_monitor_module()
    value_cols = ["외기", "제어 후"]
    long_df = _long_df({h: {c: 20.0 for c in value_cols} for h in range(2)}, value_cols)
    out_df, domain, color_range = monitor_mod._split_control_segments(long_df, value_cols, None, now_hour=1)
    assert domain == value_cols
    assert len(domain) == len(color_range)
    assert list(out_df["구분"].unique()) == value_cols
