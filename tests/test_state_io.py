"""src/control/state_io.py — 원자적 상태 파일 읽기/쓰기 공통 유틸 테스트(이슈 #40)."""
import json

from control import state_io


def test_save_json_atomic_roundtrip(tmp_path):
    path = tmp_path / "sub" / "state.json"
    data = {"a": 1, "b": "값"}
    state_io.save_json_atomic(path, data)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_save_json_atomic_swallows_oserror(tmp_path, monkeypatch):
    path = tmp_path / "state.json"

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(state_io.os, "replace", _boom)
    state_io.save_json_atomic(path, {"a": 1})  # 예외 없이 반환
    assert not path.exists()


def test_load_json_missing_file_returns_empty(tmp_path):
    assert state_io.load_json(tmp_path / "missing.json") == {}


def test_load_json_corrupted_file_returns_empty(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert state_io.load_json(path) == {}


def test_load_json_valid_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"x": 1}), encoding="utf-8")
    assert state_io.load_json(path) == {"x": 1}
