"""app/views/dashboard.py — 핵심 지표를 관제 오늘 운영(제어 후) 값으로 연결(이슈 #48).

_today_live_kpi()는 monitor.py의 render_live_tab()과 동일 방식(assemble_today_timeline)으로
오늘 제어 후 값을 계산하는 순수에 가까운 헬퍼라 Streamlit 실행 컨텍스트 없이 단위 테스트
가능하다. render_metric_cards()도 ui.kpi_cards만 monkeypatch로 갈아 끼우면(내부에서 st.*를
직접 쓰지 않음) 실제 렌더 없이 값 매핑 로직을 검증할 수 있다.
"""
import importlib
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _import_dashboard_module():
    for p in (ROOT / "src", ROOT / "app", ROOT / "app" / "views"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("dashboard")


class _FakeVS:
    """render_metric_cards()가 요구하는 최소 인터페이스(reading/window)만 갖춘 더블."""

    def __init__(self, reading):
        self._reading = reading

    def reading(self):
        return self._reading

    def window(self):
        return None  # dl.infer.forecast을 monkeypatch로 대체하므로 실제 형태는 무관


def _patch_forecast(monkeypatch, result):
    from dl import infer
    monkeypatch.setattr(infer, "forecast", lambda window: result)


# ── _today_live_kpi ──────────────────────────────────────────────────────
def test_today_live_kpi_none_when_kma_unavailable(monkeypatch):
    """KMA 키 미설정·조회 실패(outdoor=None)면 None — 호출측이 가상센서로 폴백한다."""
    dashboard_mod = _import_dashboard_module()
    monkeypatch.setattr(dashboard_mod, "_cached_today_outdoor", lambda: None)
    assert dashboard_mod._today_live_kpi() is None


def test_today_live_kpi_none_when_timeline_empty(monkeypatch):
    dashboard_mod = _import_dashboard_module()
    from control import live as live_mod

    monkeypatch.setattr(dashboard_mod, "_cached_today_outdoor",
                         lambda: [{"hour": 0, "temp": 20.0, "humidity": 60.0}])
    monkeypatch.setattr(live_mod, "assemble_today_timeline", lambda *a, **k: [])
    assert dashboard_mod._today_live_kpi() is None


def test_today_live_kpi_returns_ctrl_values_on_success(monkeypatch):
    dashboard_mod = _import_dashboard_module()
    from control import live as live_mod

    monkeypatch.setattr(dashboard_mod, "_cached_today_outdoor",
                         lambda: [{"hour": 9, "temp": 21.0, "humidity": 55.0}])
    timeline = [{"hour": 9, "ctrl_temp": 23.4, "ctrl_hum": 66.0, "out_temp": 18.2}]
    monkeypatch.setattr(live_mod, "assemble_today_timeline", lambda *a, **k: timeline)

    live = dashboard_mod._today_live_kpi()
    assert live is not None
    assert live["ctrl_temp"] == 23.4
    assert live["ctrl_hum"] == 66.0
    assert live["out_temp"] == 18.2
    assert live["today"] == date.today()
    assert live["setpoints"].temp_low == 20.0  # data/control_setpoints.json 없을 때 기본값


def test_today_live_kpi_falls_back_to_last_row_when_current_hour_not_in_timeline(monkeypatch):
    """timeline에 현재 시각 행이 없으면(과거 스냅샷만 있는 등) 마지막 행을 쓴다 —
    monitor.py의 render_live_tab()과 동일한 폴백 규칙."""
    dashboard_mod = _import_dashboard_module()
    from control import live as live_mod

    monkeypatch.setattr(dashboard_mod, "_cached_today_outdoor",
                         lambda: [{"hour": 5, "temp": 21.0, "humidity": 55.0}])
    # 존재할 수 없는 hour(999)만 있는 timeline이라 next()가 못 찾고 timeline[-1]로 폴백된다.
    timeline = [{"hour": 999, "ctrl_temp": 19.9, "ctrl_hum": 61.0, "out_temp": 12.0}]
    monkeypatch.setattr(live_mod, "assemble_today_timeline", lambda *a, **k: timeline)

    live = dashboard_mod._today_live_kpi()
    assert live is not None
    assert live["ctrl_temp"] == 19.9
    assert live["ctrl_hum"] == 61.0
    assert live["out_temp"] == 12.0


def test_today_live_kpi_none_when_ctrl_values_missing(monkeypatch):
    dashboard_mod = _import_dashboard_module()
    from control import live as live_mod

    monkeypatch.setattr(dashboard_mod, "_cached_today_outdoor",
                         lambda: [{"hour": 9, "temp": 21.0, "humidity": 55.0}])
    timeline = [{"hour": 9, "ctrl_temp": None, "ctrl_hum": 66.0, "out_temp": 18.2}]
    monkeypatch.setattr(live_mod, "assemble_today_timeline", lambda *a, **k: timeline)

    assert dashboard_mod._today_live_kpi() is None


def test_today_live_kpi_none_on_unexpected_exception(monkeypatch):
    """setpoints.load()가 예기치 못한 예외를 던져도 대시보드가 죽지 않고 None 폴백
    (이슈 #10 C4 — 무크래시 원칙)."""
    dashboard_mod = _import_dashboard_module()
    from control import setpoints as setpoints_mod

    monkeypatch.setattr(dashboard_mod, "_cached_today_outdoor",
                         lambda: [{"hour": 9, "temp": 21.0, "humidity": 55.0}])

    def _boom(*a, **k):
        raise RuntimeError("설정 파일 손상")

    monkeypatch.setattr(setpoints_mod, "load", _boom)
    assert dashboard_mod._today_live_kpi() is None


# ── render_metric_cards ───────────────────────────────────────────────────
def test_render_metric_cards_uses_live_values_when_available(monkeypatch):
    """live가 있으면 내부온도/습도/외기는 제어 후 값, CO2만 가상센서 원본을 쓴다(이슈 #48)."""
    dashboard_mod = _import_dashboard_module()
    from control.setpoints import Setpoints

    _patch_forecast(monkeypatch, None)
    captured = {}
    monkeypatch.setattr(dashboard_mod, "kpi_cards", lambda items: captured.setdefault("items", items))

    vs = _FakeVS({"온도내부_평균": 99.0, "습도내부_평균": 99.0, "온도외부_평균": 99.0, "co2_평균": 500.0})
    live = {"ctrl_temp": 23.4, "ctrl_hum": 66.0, "out_temp": 18.2,
            "today": date.today(), "setpoints": Setpoints()}

    dashboard_mod.render_metric_cards(vs, live)

    items = captured["items"]
    assert items[0]["value"] == "23.4"   # 내부 온도 = ctrl_temp(원본 99.0 아님)
    assert items[1]["value"] == "66"     # 내부 습도 = ctrl_hum
    assert items[2]["value"] == "500"    # CO₂는 KMA에 없어 가상센서 원본 그대로
    assert items[3]["value"] == "18.2"   # 외부 온도 = out_temp


def test_render_metric_cards_falls_back_to_vsensor_when_live_none(monkeypatch):
    """live=None(KMA 미설정·실패)이면 기존 그대로 가상센서 원본을 쓴다(회귀 방지)."""
    dashboard_mod = _import_dashboard_module()
    _patch_forecast(monkeypatch, None)
    captured = {}
    monkeypatch.setattr(dashboard_mod, "kpi_cards", lambda items: captured.setdefault("items", items))

    vs = _FakeVS({"온도내부_평균": 21.5, "습도내부_평균": 70.0, "온도외부_평균": 15.0, "co2_평균": 420.0})

    dashboard_mod.render_metric_cards(vs, None)

    items = captured["items"]
    assert items[0]["value"] == "21.5"
    assert items[1]["value"] == "70"
    assert items[2]["value"] == "420"
    assert items[3]["value"] == "15.0"


def test_render_metric_cards_temp_chip_ok_within_setpoints_band(monkeypatch):
    dashboard_mod = _import_dashboard_module()
    from control.setpoints import Setpoints

    _patch_forecast(monkeypatch, {"next_temp": 24.0, "trend": "상승"})
    captured = {}
    monkeypatch.setattr(dashboard_mod, "kpi_cards", lambda items: captured.setdefault("items", items))

    vs = _FakeVS({"온도내부_평균": 99.0, "습도내부_평균": 50.0, "온도외부_평균": 99.0, "co2_평균": 500.0})
    live = {"ctrl_temp": 22.0, "ctrl_hum": 50.0, "out_temp": 18.0,
            "today": date.today(), "setpoints": Setpoints(temp_low=20.0, temp_high=25.0)}

    dashboard_mod.render_metric_cards(vs, live)

    temp_item = captured["items"][0]
    assert temp_item["chip_level"] == "ok"
    assert temp_item["chip"].startswith("정상")


def test_render_metric_cards_temp_chip_warn_outside_setpoints_band(monkeypatch):
    """제어 후 값이 setpoints 밴드를 벗어나면 온도 칩도 '주의'로 재판정된다(이슈 #48)."""
    dashboard_mod = _import_dashboard_module()
    from control.setpoints import Setpoints

    _patch_forecast(monkeypatch, {"next_temp": 24.0, "trend": "상승"})
    captured = {}
    monkeypatch.setattr(dashboard_mod, "kpi_cards", lambda items: captured.setdefault("items", items))

    vs = _FakeVS({"온도내부_평균": 99.0, "습도내부_평균": 50.0, "온도외부_평균": 99.0, "co2_평균": 500.0})
    live = {"ctrl_temp": 30.0, "ctrl_hum": 50.0, "out_temp": 18.0,
            "today": date.today(), "setpoints": Setpoints(temp_low=20.0, temp_high=25.0)}

    dashboard_mod.render_metric_cards(vs, live)

    temp_item = captured["items"][0]
    assert temp_item["chip_level"] == "warn"
    assert temp_item["chip"].startswith("주의")


def test_render_metric_cards_humidity_chip_thresholds_apply_on_fallback_path(monkeypatch):
    """습도 칩(HUM_WARN/CRIT) 판정은 폴백(가상센서) 경로에서도 그대로 적용된다(회귀 방지)."""
    dashboard_mod = _import_dashboard_module()
    _patch_forecast(monkeypatch, None)
    captured = {}
    monkeypatch.setattr(dashboard_mod, "kpi_cards", lambda items: captured.setdefault("items", items))

    vs = _FakeVS({"온도내부_평균": 22.0, "습도내부_평균": 92.0, "온도외부_평균": 18.0, "co2_평균": 420.0})
    dashboard_mod.render_metric_cards(vs, None)

    hum_item = captured["items"][1]
    assert hum_item["chip_level"] == "danger"


def test_render_metric_cards_humidity_chip_danger_on_live_path(monkeypatch):
    """습도 칩 임계 판정은 live(제어 후) 경로에서도 동일 소스(HUM_WARN/CRIT)를 쓴다."""
    dashboard_mod = _import_dashboard_module()
    from control.setpoints import Setpoints

    _patch_forecast(monkeypatch, None)
    captured = {}
    monkeypatch.setattr(dashboard_mod, "kpi_cards", lambda items: captured.setdefault("items", items))

    vs = _FakeVS({"온도내부_평균": 99.0, "습도내부_평균": 10.0, "온도외부_평균": 99.0, "co2_평균": 500.0})
    live = {"ctrl_temp": 22.0, "ctrl_hum": 92.0, "out_temp": 18.0,
            "today": date.today(), "setpoints": Setpoints()}

    dashboard_mod.render_metric_cards(vs, live)

    hum_item = captured["items"][1]
    assert hum_item["value"] == "92"
    assert hum_item["chip_level"] == "danger"


# ── render_alert_banner (이슈 #51 — 관제 제어 후 값 기준 경보) ──────────────
def test_render_alert_banner_no_alert_when_timeline_within_band(monkeypatch):
    """제어 후 값이 밴드 안(제어 한계 초과 아님)이면 emergency_hours가 빈 목록을 반환해
    "현재 경보 없음"만 떠야 한다 — 잘 제어되고 있으면 경보가 자연히 사라진다."""
    dashboard_mod = _import_dashboard_module()
    from control import live as live_mod

    now_hour = 9
    timeline = [{"hour": now_hour, "ctrl_temp": 22.0, "ctrl_hum": 70.0, "devices_on": []}]
    setpoints = object()
    monkeypatch.setattr(dashboard_mod, "_today_live_timeline", lambda: (timeline, setpoints))
    monkeypatch.setattr(live_mod, "emergency_hours", lambda tl, sp: [])

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 7, 5, now_hour, 30)

    monkeypatch.setattr(dashboard_mod, "datetime", _FixedDatetime)

    calls = []
    monkeypatch.setattr(dashboard_mod, "alert_strip",
                         lambda level, text, severity_label=None: calls.append((level, text)))

    dashboard_mod.render_alert_banner(vs=None)
    assert calls == [("ok", "현재 경보 없음 — 환경이 정상 범위예요.")]


def test_render_alert_banner_danger_when_current_hour_is_emergency(monkeypatch):
    """제어 한계 초과(장치 풀가동에도 밴드 밖) 시각이 현재 시각과 일치하면 danger로 표시."""
    dashboard_mod = _import_dashboard_module()
    from control import live as live_mod

    now_hour = 14
    timeline = [{"hour": now_hour, "ctrl_temp": 30.0, "ctrl_hum": 70.0,
                 "devices_on": ["cooling_fan"]}]
    setpoints = object()
    monkeypatch.setattr(dashboard_mod, "_today_live_timeline", lambda: (timeline, setpoints))
    emg_reason = "제어 한계 초과 — 냉방 풀가동에도 고온 지속(30.0℃)"
    monkeypatch.setattr(live_mod, "emergency_hours",
                         lambda tl, sp: [{"hour": now_hour, "kind": "temp_high", "reason": emg_reason}])

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 7, 5, now_hour, 0)

    monkeypatch.setattr(dashboard_mod, "datetime", _FixedDatetime)

    calls = []
    monkeypatch.setattr(dashboard_mod, "alert_strip",
                         lambda level, text, severity_label=None: calls.append((level, text, severity_label)))

    dashboard_mod.render_alert_banner(vs=None)
    assert calls == [("danger", emg_reason, "경고")]


def test_render_alert_banner_falls_back_to_vsensor_assess_when_no_timeline(monkeypatch):
    """KMA 실패로 타임라인이 없으면(제어 후 값 없음) 기존 vsensor 원본 기반 assess() 경로로
    폴백한다(회귀 방지 — 로컬/데모에서도 경보 배너가 동작해야 함)."""
    dashboard_mod = _import_dashboard_module()
    from llm import expect as expect_mod
    from llm import monitor as monitor_mod

    monkeypatch.setattr(dashboard_mod, "_today_live_timeline", lambda: (None, None))
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: {"dummy": True})
    monkeypatch.setattr(expect_mod, "predict", lambda model, r, d: {"평균": 20.0})
    monkeypatch.setattr(
        monitor_mod, "assess",
        lambda r, exp: [{"level": "경고", "reason": "고습 93% 지속", "cause": None}])

    calls = []
    monkeypatch.setattr(dashboard_mod, "alert_strip",
                         lambda level, text, severity_label=None: calls.append((level, text)))

    vs = _FakeVS({"온도내부_평균": 22.0, "습도내부_평균": 93.0})
    vs.date = lambda: date.today()
    dashboard_mod.render_alert_banner(vs)
    assert calls == [("danger", "고습 93% 지속")]


def test_render_alert_banner_fallback_ok_when_no_alerts(monkeypatch):
    """폴백 경로에서도 경보가 없으면 기존과 동일하게 '현재 경보 없음'을 띄운다."""
    dashboard_mod = _import_dashboard_module()
    from llm import expect as expect_mod
    from llm import monitor as monitor_mod

    monkeypatch.setattr(dashboard_mod, "_today_live_timeline", lambda: (None, None))
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: {"dummy": True})
    monkeypatch.setattr(expect_mod, "predict", lambda model, r, d: {"평균": 20.0})
    monkeypatch.setattr(monitor_mod, "assess", lambda r, exp: [])

    calls = []
    monkeypatch.setattr(dashboard_mod, "alert_strip",
                         lambda level, text, severity_label=None: calls.append((level, text)))

    vs = _FakeVS({"온도내부_평균": 22.0, "습도내부_평균": 70.0})
    vs.date = lambda: date.today()
    dashboard_mod.render_alert_banner(vs)
    assert calls == [("ok", "현재 경보 없음 — 환경이 정상 범위예요.")]


# ── render() — 이슈 #51 리뷰 P2-4: 타임라인 중복 계산 방지 ───────────────────
def test_render_calls_today_live_timeline_only_once(monkeypatch):
    """render() 1회 호출에 _today_live_timeline()(KMA baseline+simulate_control 조립,
    비용이 드는 작업)이 정확히 1번만 실행돼야 하고, 경보 배너·핵심 지표가 같은 결과를
    공유해야 한다(각자 내부에서 재조회하면 render() 1회에 2번 도는 회귀가 있었다)."""
    dashboard_mod = _import_dashboard_module()
    import streamlit as st

    calls = []
    timeline = [{"hour": 9, "ctrl_temp": 22.0, "ctrl_hum": 70.0, "out_temp": 18.0}]
    setpoints = object()

    def _fake_timeline():
        calls.append(1)
        return timeline, setpoints

    banner_args = []
    kpi_args = []

    monkeypatch.setattr(dashboard_mod, "_today_live_timeline", _fake_timeline)
    monkeypatch.setattr(dashboard_mod, "page_header", lambda *a, **k: None)
    monkeypatch.setattr(dashboard_mod, "section", lambda *a, **k: None)
    monkeypatch.setattr(dashboard_mod, "render_alert_banner",
                         lambda vs, live_timeline=None: banner_args.append(live_timeline))
    monkeypatch.setattr(dashboard_mod, "_today_live_kpi",
                         lambda live_timeline=None: kpi_args.append(live_timeline) or None)
    monkeypatch.setattr(dashboard_mod, "render_metric_cards", lambda vs, live=None: None)
    monkeypatch.setattr(dashboard_mod, "render_recent_chart", lambda vs: None)
    monkeypatch.setattr(dashboard_mod, "render_shortcuts", lambda: None)
    fake_vs = _FakeVS({})
    fake_vs.date = lambda: date.today()
    monkeypatch.setattr(dashboard_mod, "_latest_vsensor", lambda: (fake_vs, 2024))
    monkeypatch.setattr(st, "divider", lambda: None)

    dashboard_mod.render()

    assert len(calls) == 1
    assert banner_args == [(timeline, setpoints)]
    assert kpi_args == [(timeline, setpoints)]


def test_render_metric_cards_unavailable_on_vsensor_reading_failure(monkeypatch):
    """vs.reading()이 예외를 던지면(가상 센서 조회 실패) unavailable 안내로 조용히 종료한다
    (이슈 #10 C4 — live 유무와 무관하게 항상 무크래시)."""
    dashboard_mod = _import_dashboard_module()

    calls = []
    monkeypatch.setattr(dashboard_mod, "unavailable",
                         lambda feature, reason, hint=None: calls.append((feature, reason)))

    class _BrokenVS:
        def reading(self):
            raise RuntimeError("센서 오류")

    dashboard_mod.render_metric_cards(_BrokenVS(), live={"ctrl_temp": 1, "ctrl_hum": 1,
                                                          "out_temp": 1, "today": date.today(),
                                                          "setpoints": None})
    assert calls and calls[0][0] == "핵심 지표"
