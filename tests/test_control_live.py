"""src/control/live.py — 오늘 운영 모드(이슈 #23) 테스트. KMA·expect는 mock."""
import json

import pytest

from control import live
from control.actuators import default_states
from control.setpoints import Setpoints


def _sp():
    return Setpoints()  # 온도 20~25℃, 습도 60~85%


def _forecast(hours_temp: dict, date_str="20260703"):
    """hour(int) -> temp 매핑으로 get_forecast_3d() 형태 mock 응답 구성."""
    hourly = [{"date": date_str, "time": f"{h:02d}00", "temp": t, "humidity": 70.0}
              for h, t in hours_temp.items()]
    return {"unavailable": False, "hourly": hourly, "daily": []}


# ── today_outdoor ────────────────────────────────────────────────────────
def test_today_outdoor_none_when_kma_unavailable(monkeypatch):
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: {"unavailable": True, "reason": "키 없음"})
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    assert live.today_outdoor() is None


def test_today_outdoor_replaces_current_hour(monkeypatch):
    from datetime import date, datetime
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 20.0 + h * 0.1 for h in range(24)}))
    monkeypatch.setattr(weather, "get_current",
                         lambda: {"unavailable": False, "temp": 99.0, "humidity": 55.0})
    now = datetime(2026, 7, 3, 14, 30)
    out = live.today_outdoor(today=date(2026, 7, 3), now=now)
    assert out is not None
    item14 = next(i for i in out if i["hour"] == 14)
    assert item14["temp"] == 99.0
    assert item14["humidity"] == 55.0
    assert len(out) == 24


# ── indoor_baseline ──────────────────────────────────────────────────────
def _patch_expect_model(monkeypatch, slope=0.8, intercept=2.0):
    """expect.load_model()이 predict()에서 온도외부_평균*slope+intercept를 돌려주는 더미 모델."""
    from llm import expect as expect_mod

    class _DummyModel:
        def predict(self, X):
            import numpy as np
            return np.array([X[0][0] * slope + intercept])

    payload = {
        "features": ["온도외부_평균", "일사량_평균", "doy_sin", "doy_cos"],
        "models": {"평균": _DummyModel(), "최저": _DummyModel()},
        "resid_sigma": {"평균": 1.0, "최저": 1.0},
        "doy_solar_climatology": {},
    }
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: payload)


def test_indoor_baseline_uses_expect_model(monkeypatch):
    from datetime import date
    _patch_expect_model(monkeypatch)
    outdoor = [{"hour": 12, "temp": 30.0, "humidity": 50.0}]
    baseline = live.indoor_baseline(outdoor, date=date(2026, 7, 3))
    assert baseline[0]["base_temp"] == pytest.approx(30.0 * 0.8 + 2.0)
    assert baseline[0]["base_hum"] == pytest.approx(60.0)  # 50 + INDOOR_HUMIDITY_OFFSET(10)


def test_indoor_baseline_falls_back_to_outdoor_temp_when_no_model(monkeypatch):
    from datetime import date
    from llm import expect as expect_mod
    monkeypatch.setattr(expect_mod, "load_model", lambda force=False: None)
    outdoor = [{"hour": 12, "temp": 33.0, "humidity": 40.0}]
    baseline = live.indoor_baseline(outdoor, date=date(2026, 7, 3))
    assert baseline[0]["base_temp"] == 33.0


# ── simulate_control ─────────────────────────────────────────────────────
def _baseline(hours):
    """hour -> (base_temp, base_hum) dict → baseline 리스트."""
    return [{"hour": h, "out_temp": t, "out_hum": hum, "base_temp": t, "base_hum": hum}
            for h, (t, hum) in hours.items()]


def test_simulate_control_turns_on_device_when_over_band_and_moves_ctrl_toward_band():
    # 매 시간 30℃(밴드 상한 25℃ 초과) 유지 — cooling_fan/vent가 계속 ON, ctrl_temp가
    # base_temp보다 낮아져야 함(냉방 효과로 밴드 방향 조정).
    baseline = _baseline({h: (30.0, 70.0) for h in range(5)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    # 첫 시간엔 ctrl_temp==base_temp(기준선 그대로 시작)이고, 온도가 밴드 상한을 넘으므로
    # cooling_fan·vent가 즉시 ON 판정돼야 한다.
    assert timeline[0]["ctrl_temp"] == pytest.approx(30.0)
    assert "cooling_fan" in timeline[0]["devices_on"]
    assert "vent" in timeline[0]["devices_on"]
    later = timeline[-1]
    assert later["ctrl_temp"] < later["base_temp"]


def test_simulate_control_devices_off_within_band():
    baseline = _baseline({h: (22.0, 70.0) for h in range(3)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    for item in timeline:
        assert item["devices_on"] == []
        assert item["events"] == []


def test_simulate_control_emergency_when_full_blast_cannot_recover():
    # 밴드 상한을 훨씬 초과하는 외기·기준선이 지속돼 냉방+환기 풀가동으로도 못 잡는 상황을 구성.
    baseline = _baseline({h: (45.0, 70.0) for h in range(6)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    emg = live.emergency_hours(timeline, _sp())
    assert len(emg) > 0
    assert all("고온 지속" in e["reason"] for e in emg)


def test_simulate_control_no_chattering_at_boundary():
    # 정확히 밴드 상한(25.0)에 붙어있는 시퀀스 — 히스테리시스로 채터링(ON/OFF 반복) 없어야 함.
    baseline = _baseline({0: (26.0, 70.0), 1: (24.8, 70.0), 2: (24.9, 70.0), 3: (24.7, 70.0)})
    states = default_states()
    timeline = live.simulate_control(baseline, _sp(), states, date="2026-07-03")
    on_flags = ["cooling_fan" in t["devices_on"] for t in timeline]
    # 한 번 ON 되면(temp>25) 데드밴드(24.5) 안쪽으로 복귀하기 전까진 계속 ON 유지 —
    # 24.8/24.9/24.7 모두 24.5보다 크므로 OFF로 전환되지 않아야 한다(채터링 없음).
    assert on_flags == [True, True, True, True]


def test_simulate_control_deterministic_when_states_deepcopied_between_calls():
    """오늘 운영 탭(app/views/monitor.py) P1 회귀 방지 — simulate_control()은 전달받은
    states를 in-place mutate하므로, 세션 원본을 그대로 재사용하면 두 번째 호출의 "0시
    시작" 상태가 첫 번째 호출의 "마지막 시간 결과"로 오염돼 타임라인이 리런마다 드리프트
    한다. monitor.py는 매 렌더 `deepcopy(states)`로 시뮬용 사본을 분리해 세션 states는
    항상 default_states() 그대로 유지한다 — 이 테스트는 그 패턴으로 동일 입력 2회 호출 시
    타임라인이 완전히 동일함을 검증한다."""
    from copy import deepcopy

    baseline = _baseline({h: (30.0, 70.0) for h in range(6)})  # 밴드 상한 초과 지속 — 장치 ON 유발
    session_states = default_states()

    sim_states_1 = deepcopy(session_states)
    timeline_1 = live.simulate_control(baseline, _sp(), sim_states_1, date="2026-07-03")

    # 세션 원본(session_states)은 첫 호출로 오염되지 않아야 한다.
    assert all(not s.on for s in session_states.values())

    sim_states_2 = deepcopy(session_states)
    timeline_2 = live.simulate_control(baseline, _sp(), sim_states_2, date="2026-07-03")

    def _strip(timeline):
        return [{k: v for k, v in item.items() if k != "events"} for item in timeline]

    assert _strip(timeline_1) == _strip(timeline_2)


def test_simulate_control_without_deepcopy_drifts_across_calls():
    """대조군 — deepcopy 없이 같은 states 객체를 재사용하면(수정 전 버그) 직전 호출이 장치를
    ON으로 남긴 채 끝난 뒤, 그 상태를 이어받아 재실행하면 히스테리시스 데드밴드 때문에
    "0시부터 밴드 정상 범위"인 새 타임라인의 첫 시간조차 잘못 ON으로 판정된다."""
    hot_baseline = _baseline({h: (30.0, 70.0) for h in range(6)})   # 밴드 상한 초과 지속 — fan ON 유발
    normal_baseline = _baseline({h: (24.7, 70.0) for h in range(3)})  # 데드밴드(24.5) 안쪽, 밴드 자체는 정상

    states_fresh = default_states()
    fresh_timeline = live.simulate_control(normal_baseline, _sp(), states_fresh, date="2026-07-03")
    assert fresh_timeline[0]["devices_on"] == []  # 새 states로 시작하면 정상 범위라 OFF

    states_polluted = default_states()
    live.simulate_control(hot_baseline, _sp(), states_polluted, date="2026-07-03")
    assert states_polluted["cooling_fan"].on  # 직전 호출로 오염 — ON 상태로 남음

    polluted_timeline = live.simulate_control(normal_baseline, _sp(), states_polluted, date="2026-07-03")
    # 같은 normal_baseline인데 오염된 states로 재실행하면 첫 시간 devices_on이 fresh와 달라진다
    # (데드밴드 안쪽이라 was_high=True 이력이 남은 fan이 계속 ON으로 판정됨) — 회귀 재현.
    assert polluted_timeline[0]["devices_on"] != fresh_timeline[0]["devices_on"]


# ── run_notify — 상태 파일 dedup ─────────────────────────────────────────
@pytest.fixture
def _isolated_state(tmp_path, monkeypatch):
    state_path = tmp_path / "control_live_state.json"
    monkeypatch.setattr(live, "STATE_PATH", state_path)
    return state_path


@pytest.fixture
def _isolated_setpoints(tmp_path, monkeypatch):
    from control import setpoints as setpoints_mod
    sp_path = tmp_path / "control_setpoints.json"
    monkeypatch.setattr(setpoints_mod, "SETPOINTS_PATH", sp_path)
    return sp_path


def _patch_notify(monkeypatch, sent: list):
    from llm import notify
    def _fake_send(embed):
        sent.append(embed)
        return True, "ok"
    monkeypatch.setattr(notify, "send_discord", _fake_send)


def test_run_notify_sends_on_first_run_and_dedups_second_run(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 30.0 for h in range(24)}, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    n1 = live.run_notify(dry_run=False, today=today)
    assert n1 >= 1  # 첫 실행 — cooling_fan 등 장치 전환 발송

    n2 = live.run_notify(dry_run=False, today=today)
    assert n2 == 0  # 같은 상태 재실행 — 전환 없음, 발송 0건


def test_run_notify_uses_current_hour_not_timeline_end_for_transitions(
        monkeypatch, _isolated_state, _isolated_setpoints):
    """P1-2 회귀 방지 — 현재 시각(now.hour)에는 밴드 내(장치 OFF)인데 미래 시간대만
    밴드 초과(장치 ON)인 프로파일. 픽스 전엔 timeline 마지막(미래) 상태를 "지금 전환"으로
    오인해 cooling_fan ON 알림을 보냈지만, 픽스 후엔 현재 시각 기준 OFF라 발송 0건이어야 한다."""
    from datetime import date, datetime
    from llm import weather

    # 09시(현재로 고정)는 밴드 내(22℃), 10시 이후만 밴드 초과(30℃) — 미래에만 장치가 켜짐.
    hours_temp = {h: (22.0 if h <= 9 else 30.0) for h in range(24)}
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast(hours_temp, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    today = date(2026, 7, 3)
    now = datetime(2026, 7, 3, 9, 0)   # 09시 = 아직 밴드 내(장치 OFF)여야 하는 "지금"
    live.run_notify(dry_run=False, today=today, now=now)
    # 미래(10시 이후) 밴드 초과로 인한 긴급 알림은 별개로 발송될 수 있지만, "장치 전환"
    # 알림(🎛)은 지금(09시) 기준으로는 없어야 한다 — 픽스 전엔 timeline 마지막(미래, 30℃
    # 지속)의 cooling_fan ON을 "지금 전환"으로 오인해 이 임베드가 섞여 들어갔다.
    device_transition_embeds = [e for e in sent if e["title"].startswith("🎛")]
    assert device_transition_embeds == []


def test_run_notify_resets_on_new_date(monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 30.0 for h in range(24)}, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    sent = []
    _patch_notify(monkeypatch, sent)

    live.run_notify(dry_run=False, today=date(2026, 7, 3))
    state = json.loads(_isolated_state.read_text())
    assert state["date"] == "2026-07-03"

    # 다음날 같은 forecast(date_str 다름으로 today_outdoor가 None 반환하지 않도록 갱신)
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 30.0 for h in range(24)}, date_str="20260704"))
    n = live.run_notify(dry_run=False, today=date(2026, 7, 4))
    assert n >= 1  # 날짜가 바뀌어 상태 리셋 → 다시 전환으로 인식돼 발송
    state2 = json.loads(_isolated_state.read_text())
    assert state2["date"] == "2026-07-04"


def test_run_notify_dry_run_does_not_write_state_or_call_discord(
        monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d",
                         lambda: _forecast({h: 30.0 for h in range(24)}, date_str="20260703"))
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})
    _patch_expect_model(monkeypatch, slope=1.0, intercept=0.0)

    from llm import notify
    called = []
    monkeypatch.setattr(notify, "send_discord", lambda embed: called.append(embed) or (True, "ok"))

    n = live.run_notify(dry_run=True, today=date(2026, 7, 3))
    assert n >= 1
    assert called == []            # 실제 discord 호출 없음
    assert not _isolated_state.exists()  # 상태 파일도 쓰지 않음


def test_run_notify_kma_unavailable_graceful(monkeypatch, _isolated_state, _isolated_setpoints):
    from datetime import date
    from llm import weather
    monkeypatch.setattr(weather, "get_forecast_3d", lambda: {"unavailable": True, "reason": "키 없음"})
    monkeypatch.setattr(weather, "get_current", lambda: {"unavailable": True})

    sent = []
    _patch_notify(monkeypatch, sent)

    n1 = live.run_notify(dry_run=False, today=date(2026, 7, 3))
    assert n1 == 0     # 1회차 실패는 아직 이상 알림 안 보냄(연속 2회 임계치)
    n2 = live.run_notify(dry_run=False, today=date(2026, 7, 3))
    assert n2 == 1     # 2회 연속 실패 → 이상 알림 1건
