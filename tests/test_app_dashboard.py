"""app/views/dashboard.py — 핵심 지표를 관제 오늘 운영(제어 후) 값으로 연결(이슈 #48).

_today_live_kpi()는 monitor.py의 render_live_tab()과 동일 방식(assemble_today_timeline)으로
오늘 제어 후 값을 계산하는 순수에 가까운 헬퍼라 Streamlit 실행 컨텍스트 없이 단위 테스트
가능하다. render_metric_cards()도 ui.kpi_cards만 monkeypatch로 갈아 끼우면(내부에서 st.*를
직접 쓰지 않음) 실제 렌더 없이 값 매핑 로직을 검증할 수 있다.
"""
import importlib
import sys
from datetime import date
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
