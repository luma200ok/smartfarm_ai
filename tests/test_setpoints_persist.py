"""설정 밴드 영속화(저장/로드) 테스트 — 이슈 #21."""
import json

import control.setpoints as setpoints_mod
from control.setpoints import Setpoints, load, save, save_changed


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "sp.json"
    sp = Setpoints(temp_low=18.0, temp_high=27.0, hum_low=55.0, hum_high=90.0)
    save(sp, path)
    loaded = load(path)
    assert loaded.temp_low == 18.0
    assert loaded.temp_high == 27.0
    assert loaded.hum_low == 55.0
    assert loaded.hum_high == 90.0
    # deadband는 파일 값과 무관하게 항상 기본값
    assert loaded.temp_deadband == Setpoints().temp_deadband
    assert loaded.hum_deadband == Setpoints().hum_deadband


def test_load_missing_file_falls_back_to_default(tmp_path):
    path = tmp_path / "missing.json"
    loaded = load(path)
    assert loaded == Setpoints()


def test_load_corrupt_json_falls_back_to_default(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    loaded = load(path)
    assert loaded == Setpoints()


def test_load_missing_keys_falls_back_to_default(tmp_path):
    path = tmp_path / "missing_keys.json"
    path.write_text(json.dumps({"temp_low": 15.0}), encoding="utf-8")
    loaded = load(path)
    assert loaded == Setpoints()


def test_load_out_of_range_values_are_clamped(tmp_path):
    path = tmp_path / "oob.json"
    path.write_text(json.dumps({
        "temp_low": -10.0, "temp_high": 100.0,
        "hum_low": -5.0, "hum_high": 200.0,
    }), encoding="utf-8")
    loaded = load(path)
    assert loaded.temp_low == 0.0
    assert loaded.temp_high == 40.0
    assert loaded.hum_low == 0.0
    assert loaded.hum_high == 100.0


def test_load_low_greater_than_high_falls_back_to_default_band(tmp_path):
    path = tmp_path / "inverted.json"
    path.write_text(json.dumps({
        "temp_low": 30.0, "temp_high": 20.0,
        "hum_low": 90.0, "hum_high": 50.0,
    }), encoding="utf-8")
    loaded = load(path)
    default = Setpoints()
    assert loaded.temp_low == default.temp_low
    assert loaded.temp_high == default.temp_high
    assert loaded.hum_low == default.hum_low
    assert loaded.hum_high == default.hum_high


def test_save_writes_atomically_no_leftover_tmp(tmp_path):
    path = tmp_path / "sp.json"
    save(Setpoints(), path)
    assert path.exists()
    tmp_file = path.with_suffix(path.suffix + ".tmp")
    assert not tmp_file.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["temp_low"] == Setpoints().temp_low


def test_save_write_failure_is_swallowed(tmp_path, monkeypatch):
    """P2-1: os.replace가 OSError를 던져도 save()는 예외 없이 반환한다."""
    path = tmp_path / "sp.json"

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    from control import state_io
    monkeypatch.setattr(state_io.os, "replace", _boom)
    save(Setpoints(), path)  # 예외 없이 반환되어야 함
    assert not path.exists()


def test_save_changed_merges_concurrent_session_edits(tmp_path):
    """P2-2: 세션 B가 습도를 먼저 저장한 뒤, 세션 A가 온도만 바꿔 저장하면
    파일에는 두 변경(온도+습도)이 모두 남아야 한다(lost-update 방지)."""
    path = tmp_path / "sp.json"
    base = Setpoints()
    save(base, path)

    # 세션 B: 습도만 변경 후 저장
    b_prev = load(path)
    b_now = Setpoints(**{**b_prev.__dict__, "hum_low": 50.0, "hum_high": 95.0})
    save_changed(b_now, b_prev, path)

    # 세션 A: 자신의 stale 스냅샷(base) 기준으로 온도만 변경
    a_prev = base
    a_now = Setpoints(**{**a_prev.__dict__, "temp_low": 15.0, "temp_high": 30.0})
    merged = save_changed(a_now, a_prev, path)

    assert merged.temp_low == 15.0
    assert merged.temp_high == 30.0
    assert merged.hum_low == 50.0
    assert merged.hum_high == 95.0

    on_disk = load(path)
    assert on_disk.temp_low == 15.0
    assert on_disk.temp_high == 30.0
    assert on_disk.hum_low == 50.0
    assert on_disk.hum_high == 95.0
