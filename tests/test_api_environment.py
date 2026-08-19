"""GET /api/environment/today — smartfarm_ai#66. 무거운 연산(assemble_today_timeline 전체
재계산) 없이 상태/스냅샷 파일 + KMA 실황 단건 조회만으로 조립되는지 검증한다.

기존 control_live 테스트(tests/test_control_live.py)의 `_isolated_state` 패턴을 그대로
따른다 — 실제 파일시스템 STATE_PATH를 건드리지 않도록 tmp_path로 격리."""
import json

import pytest
from control import live


@pytest.fixture
def _isolated_state(tmp_path, monkeypatch):
    state_path = tmp_path / "control_live_state.json"
    monkeypatch.setattr(live, "STATE_PATH", state_path)
    return state_path


def _write_state(path, date_str: str, snapshots: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": date_str, "snapshots": snapshots, "version": 2}),
                    encoding="utf-8")


def test_environment_today_full_data_ok(api_client, monkeypatch, _isolated_state):
    """KMA 실황 + 오늘 스냅샷(지금 시각)이 모두 있으면 전 필드가 채워진다."""
    from llm import weather

    monkeypatch.setattr(weather, "get_current",
                        lambda: {"unavailable": False, "temp": 28.5, "humidity": 55.0})

    import datetime as _dt
    fixed_now = _dt.datetime(2026, 8, 20, 14, 0, 0)

    class _FixedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    from api.routers import environment
    monkeypatch.setattr(environment, "datetime", _FixedDatetime)

    today_str = fixed_now.date().isoformat()
    _write_state(_isolated_state, today_str, {
        "14": {
            "out_temp": 27.0, "out_hum": 50.0, "base_temp": 29.0, "base_hum": 60.0,
            "ctrl_temp": 26.5, "ctrl_hum": 58.0,
            "devices_on": ["cooling_fan"],
            "events": [], "source": "sim", "recorded_at": "2026-08-20T14:00:05",
        },
    })

    r = api_client.get("/api/environment/today")
    assert r.status_code == 200
    body = r.json()
    assert body["demo"] is True
    assert body["outdoor"] == {"temp": 28.5, "humidity": 55.0}
    assert body["indoor"] == {"temp": 26.5, "humidity": 58.0, "controlled": True}
    assert {"name": "cooling_fan", "on": True} in body["devices"]
    assert {"name": "heater", "on": False} in body["devices"]
    assert len(body["devices"]) == 4
    assert body["alerts"] == []
    assert body["updated_at"] == "2026-08-20T14:00:05"


def test_environment_today_kma_unavailable_graceful_200(api_client, monkeypatch, _isolated_state):
    """KMA 실황 불가 — 200 유지, outdoor는 None 필드 + alerts에 사유."""
    from llm import weather

    monkeypatch.setattr(weather, "get_current",
                        lambda: {"unavailable": True, "reason": "KMA_SERVICE_KEY 미설정"})
    r = api_client.get("/api/environment/today")
    assert r.status_code == 200
    body = r.json()
    assert body["outdoor"] == {"temp": None, "humidity": None}
    assert any("외기 실황 조회 실패" in a for a in body["alerts"])


def test_environment_today_no_snapshot_graceful_200(api_client, monkeypatch, _isolated_state):
    """오늘 상태 파일이 없거나 스냅샷이 하나도 없으면 indoor/devices가 비고 alerts에 안내."""
    from llm import weather

    monkeypatch.setattr(weather, "get_current",
                        lambda: {"unavailable": False, "temp": 20.0, "humidity": 40.0})
    r = api_client.get("/api/environment/today")
    assert r.status_code == 200
    body = r.json()
    assert body["indoor"] == {"temp": None, "humidity": None, "controlled": True}
    assert body["devices"] == []
    assert "오늘 기록된 운영 데이터 없음" in body["alerts"]


def test_environment_today_uses_latest_past_hour_when_current_hour_missing(
        api_client, monkeypatch, _isolated_state):
    """지금(now_hour) 스냅샷이 아직 없으면 그 이전 최신 스냅샷으로 폴백하고, alerts에 "갱신
    없음"을 남긴다(오래된 데이터임을 서비스가 알 수 있게)."""
    from llm import weather

    monkeypatch.setattr(weather, "get_current",
                        lambda: {"unavailable": False, "temp": 25.0, "humidity": 50.0})

    import datetime as _dt
    fixed_now = _dt.datetime(2026, 8, 20, 15, 0, 0)

    class _FixedDatetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    from api.routers import environment
    monkeypatch.setattr(environment, "datetime", _FixedDatetime)

    today_str = fixed_now.date().isoformat()
    _write_state(_isolated_state, today_str, {
        "13": {
            "ctrl_temp": 24.0, "ctrl_hum": 55.0, "devices_on": [],
            "recorded_at": "2026-08-20T13:00:00",
        },
    })

    r = api_client.get("/api/environment/today")
    assert r.status_code == 200
    body = r.json()
    assert body["indoor"]["temp"] == 24.0
    assert any("13시" in a and "갱신 없음" in a for a in body["alerts"])


def test_environment_today_state_from_other_day_is_ignored(api_client, monkeypatch, _isolated_state):
    """상태 파일 date가 오늘과 다르면(load_today_snapshots 계약) 어제 스냅샷을 쓰지 않는다."""
    from llm import weather

    monkeypatch.setattr(weather, "get_current",
                        lambda: {"unavailable": False, "temp": 22.0, "humidity": 45.0})
    _write_state(_isolated_state, "2020-01-01", {
        "10": {"ctrl_temp": 99.0, "ctrl_hum": 99.0, "devices_on": ["heater"]},
    })
    r = api_client.get("/api/environment/today")
    assert r.status_code == 200
    body = r.json()
    assert body["indoor"] == {"temp": None, "humidity": None, "controlled": True}
    assert body["devices"] == []
