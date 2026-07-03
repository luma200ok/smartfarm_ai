"""설정 밴드 영속화(저장/로드) 테스트 — 이슈 #21."""
import json

from control.setpoints import Setpoints, load, save


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
