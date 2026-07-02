"""
가상 센서 — 특정 년도 토마토 온실 실데이터를 '라이브 스트림'처럼 재생(replay).

실제 IoT 센서가 없으므로, env_daily.csv 의 실측 시계열을 하루씩 전진시키며
'지금 들어오는 센서값'처럼 제공한다. 값이 실측이라 LSTM 예측이 realistic(random 아님).
커서가 가리키는 시점의 최근 WINDOW일이 forecast 입력이 된다.

app/phase3_llm.py 가 VirtualSensor 를 세션에 두고 tick()으로 시간을 진행 →
infer.set_live_window() 로 라이브 창을 주입 → 처방·코치·경보가 그 시점 기준으로 갱신.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dl import infer  # noqa: E402  (ENV_FEATURES, SEQ_KEYS, WINDOW, ENV_CSV)

WINDOW = infer.WINDOW


@lru_cache(maxsize=1)
def _tomato_df():
    import pandas as pd
    df = pd.read_csv(infer.ENV_CSV, encoding="utf-8-sig")
    return df[df["품목"].astype(str).str.contains("토마토", na=False)]


def available_years() -> list[int]:
    """토마토 데이터가 있는 연도 목록."""
    if not infer.ENV_CSV.exists():
        return []
    return sorted(int(y) for y in _tomato_df()["연도"].dropna().unique())


class VirtualSensor:
    """특정 년도 토마토 온실의 최장 시계열을 하루씩 재생하는 가상 센서."""

    def __init__(self, year: int | None = None):
        df = _tomato_df()
        if year is not None:
            df = df[df["연도"] == year]
        best = None
        for key, g in df.groupby(infer.SEQ_KEYS):
            g = g.sort_values("날짜")
            if len(g) >= WINDOW and (best is None or len(g) > len(best[0])):
                best = (g[infer.ENV_FEATURES].to_numpy(np.float32), list(g["날짜"]), key)
        if best is None:
            raise ValueError(f"{year}년 토마토 시계열(≥{WINDOW}일)이 없습니다.")
        self.series, self.dates, self.key = best
        self.year = year
        self.cursor = WINDOW - 1          # 예측 가능한 첫 시점(최근 7일 확보)
        self._injections: list[dict] = []   # 시뮬 주입(오프셋) — 원본 series는 불변

    def inject(self, feature: str, start, days: int, delta: float) -> None:
        """feature(예: 온도외부_평균)에 start(날짜 또는 인덱스)부터 days일간 delta를 더한다.

        원본 self.series는 변경하지 않고 reading()/window() 반환값에만 적용(read-time overlay)."""
        idx = self.dates.index(start) if isinstance(start, str) else int(start)
        self._injections.append({"feature": feature, "start": idx, "end": idx + days - 1, "delta": delta})

    def clear_injections(self) -> None:
        """주입 전부 해제 — '정상' 시나리오로 복귀."""
        self._injections = []

    def _apply_injections(self, arr: np.ndarray, abs_indices: list[int]) -> np.ndarray:
        if not self._injections:
            return arr
        out = arr.copy()
        feat_idx = {f: i for i, f in enumerate(infer.ENV_FEATURES)}
        for row, abs_i in enumerate(abs_indices):
            for inj in self._injections:
                if inj["start"] <= abs_i <= inj["end"]:
                    fi = feat_idx.get(inj["feature"])
                    if fi is not None:
                        out[row, fi] += inj["delta"]
        return out

    def window(self) -> np.ndarray:
        """현재 시점 기준 최근 WINDOW일 (7,8) — forecast 입력(주입 반영)."""
        lo = self.cursor - WINDOW + 1
        base = self.series[lo: self.cursor + 1]
        return self._apply_injections(base, list(range(lo, self.cursor + 1)))

    def reading(self) -> dict:
        """현재 시점의 센서 관측값(피처별, 주입 반영)."""
        row = self._apply_injections(self.series[self.cursor: self.cursor + 1], [self.cursor])[0]
        return {f: float(v) for f, v in zip(infer.ENV_FEATURES, row)}

    def date(self) -> str:
        return self.dates[self.cursor]

    def tick(self) -> str:
        """하루 전진. 끝에 도달하면 처음(예측 가능 시점)으로 순환."""
        self.cursor += 1
        if self.cursor >= len(self.series):
            self.cursor = WINDOW - 1
        return self.date()

    def seek(self, target: str | int) -> str:
        """날짜 문자열(또는 인덱스)로 커서를 직접 이동.

        범위를 벗어나면 WINDOW-1 ~ len(series)-1 로 클램프해 최근 7일 창 확보를 유지한다.
        """
        idx = self.dates.index(target) if isinstance(target, str) else int(target)
        self.cursor = min(max(idx, WINDOW - 1), len(self.series) - 1)
        return self.date()

    @property
    def farm(self) -> tuple:
        return self.key


# 이슈 #6 PR-2 데모 프리셋 — Streamlit 시나리오 selectbox에서 사용.
# "한파": 외기·내부 모두 급락(잔차는 작음) → cause=외기 요인.
# "히터고장": 외기는 그대로인데 내부만 급락(잔차 큼) → cause=설비 고장 의심.
SCENARIOS = ("정상", "한파", "히터고장")


def apply_scenario(vs: "VirtualSensor", name: str, days: int = 3) -> None:
    """vs 현재 커서 기준으로 프리셋 주입(먼저 기존 주입을 모두 해제)."""
    vs.clear_injections()
    if name == "한파":
        vs.inject("온도외부_평균", vs.cursor, days, -10.0)
        vs.inject("온도내부_평균", vs.cursor, days, -5.0)
        vs.inject("온도내부_최저", vs.cursor, days, -5.0)
    elif name == "히터고장":
        vs.inject("온도내부_평균", vs.cursor, days, -8.0)
        vs.inject("온도내부_최저", vs.cursor, days, -8.0)
    # "정상"은 clear_injections()만으로 충분
