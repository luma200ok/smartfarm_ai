"""
이슈 #6 PR-2 — 외기→실내 기대값(env_expect_reg.pkl) lazy-load + 예측 유틸.

train_expect.py(PR-1)가 저장한 payload를 읽어 "지금 외기 조건이면 정상적으로 기대되는
실내 온도(평균/최저)"를 계산한다. monitor.py가 실측과 이 기대값의 잔차(z-score)로
원인(외기 요인 vs 설비 고장 의심)을 구분하고, feedforward가 예보 기반 사전 경보에 쓴다.

컨벤션: pkl 부재(미배포 환경)는 예외 전파 없이 None으로 우아하게 저하 — 호출부는
expected() 가 None이면 기존 로직(원인 미표기)으로 계속 동작해야 한다.
"""
import sys
from datetime import date as _date, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ROOT = _SRC.parent
PKL_PATH = ROOT / "models" / "env_expect_reg.pkl"

_model = None
_load_attempted = False


def load_model(force: bool = False) -> dict | None:
    """pkl lazy-load(1회 캐싱). 없거나 로드 실패면 None(예외 전파 안 함)."""
    global _model, _load_attempted
    if force:
        _load_attempted = False
    if _load_attempted:
        return _model
    _load_attempted = True
    _model = None
    try:
        if PKL_PATH.exists():
            import joblib
            _model = joblib.load(PKL_PATH)
    except Exception:
        _model = None
    return _model


def _parse_date(d) -> _date:
    """문자열(YYYY-MM-DD 또는 YYYYMMDD)·date·datetime·None(→오늘) 모두 허용."""
    if d is None:
        return datetime.now().date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, _date):
        return d
    s = str(d)
    digits = s.replace("-", "")
    if len(digits) == 8 and digits.isdigit():
        return datetime.strptime(digits, "%Y%m%d").date()
    return datetime.fromisoformat(s[:10]).date()


def doy(d) -> int:
    return _parse_date(d).timetuple().tm_yday


def _doy_sin_cos(d) -> tuple[float, float]:
    import numpy as np
    n = doy(d)
    return float(np.sin(2 * np.pi * n / 365)), float(np.cos(2 * np.pi * n / 365))


def predict(model: dict, reading: dict, date=None) -> dict | None:
    """model(payload dict) + reading(온도외부_평균·일사량_평균 포함) + date → 타깃별 기대값.

    model이 None이면 None. reading에 필요한 피처가 없어도 None(우아한 저하).
    반환: {"평균": float, "최저": float, "resid_sigma": {"평균": float, "최저": float}}
    """
    if model is None:
        return None
    try:
        import numpy as np
        sin, cos = _doy_sin_cos(date)
        feat_map = {"온도외부_평균": reading.get("온도외부_평균"),
                    "일사량_평균": reading.get("일사량_평균"),
                    "doy_sin": sin, "doy_cos": cos}
        features = model["features"]
        row = [feat_map[f] for f in features]
        if any(v is None for v in row):
            return None
        X = np.array([row], dtype=float)
        out = {}
        for name, m in model["models"].items():
            out[name] = float(m.predict(X)[0])
        return {**out, "resid_sigma": dict(model.get("resid_sigma", {}))}
    except Exception:
        return None


def expected(reading: dict, date=None) -> dict | None:
    """monitor.py 등에서 쓰는 편의 함수 — 배포 pkl(load_model)을 lazy-load해 predict()."""
    model = load_model()
    return predict(model, reading, date)


def z_score(actual: float, expected_val: float, sigma: float) -> float:
    """(actual-expected)/sigma. sigma<=0(비정상)이면 0(분류 불가 취급)."""
    if not sigma or sigma <= 0:
        return 0.0
    return (actual - expected_val) / sigma
