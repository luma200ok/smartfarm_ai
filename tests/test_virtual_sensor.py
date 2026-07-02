"""src/sim/virtual_sensor.py — 재생·tick·wrap·최장 시계열 선택·데이터 부족 (합성 DF 모킹)."""
import numpy as np
import pandas as pd
import pytest

from dl import infer
from sim import virtual_sensor as vsmod


def _rows(farm, n, year=2024):
    return [{"도": "A", "시군": "B", "농가명": farm, "작기": 1, "품목": "방울토마토", "연도": year,
             "날짜": f"{year}-01-{i + 1:02d}", **{f: float(i) for f in infer.ENV_FEATURES}}
            for i in range(n)]


def _one_farm():
    return pd.DataFrame(_rows("C", 12))


def _multi_farm():
    return pd.DataFrame(_rows("C1", 10) + _rows("C2", 15))   # C2 가 최장


def _too_short():
    return pd.DataFrame(_rows("C", 4))                       # < WINDOW(7)


def test_window_shape_and_tick(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _one_farm)
    vs = vsmod.VirtualSensor(2024)
    assert vs.window().shape == (vsmod.WINDOW, len(infer.ENV_FEATURES))
    d0 = vs.date()
    vs.tick()
    assert vs.date() != d0
    assert vs.window().shape == (vsmod.WINDOW, len(infer.ENV_FEATURES))


def test_cursor_wraps_at_end(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _one_farm)
    vs = vsmod.VirtualSensor(2024)
    for _ in range(50):
        vs.tick()
    assert vsmod.WINDOW - 1 <= vs.cursor < 12
    assert vs.window().shape == (vsmod.WINDOW, len(infer.ENV_FEATURES))


def test_reading_has_all_features(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _one_farm)
    assert set(vsmod.VirtualSensor(2024).reading()) == set(infer.ENV_FEATURES)


def test_picks_longest_series(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _multi_farm)
    vs = vsmod.VirtualSensor(2024)
    assert vs.farm[2] == "C2" and len(vs.series) == 15    # SEQ_KEYS 농가명=index 2


def test_raises_when_insufficient_data(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _too_short)
    with pytest.raises(ValueError):
        vsmod.VirtualSensor(2024)


def test_seek_moves_to_target_date(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _one_farm)
    vs = vsmod.VirtualSensor(2024)
    target = vs.dates[9]
    assert vs.seek(target) == target
    assert vs.date() == target


def test_seek_clamps_out_of_range(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _one_farm)
    vs = vsmod.VirtualSensor(2024)
    vs.seek(-100)
    assert vs.cursor == vsmod.WINDOW - 1
    vs.seek(9999)
    assert vs.cursor == len(vs.series) - 1


def test_seek_keeps_window_length(monkeypatch):
    monkeypatch.setattr(vsmod, "_tomato_df", _one_farm)
    vs = vsmod.VirtualSensor(2024)
    vs.seek(vs.dates[5])
    assert vs.window().shape == (vsmod.WINDOW, len(infer.ENV_FEATURES))
