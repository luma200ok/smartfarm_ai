"""train_expect.py 로직 검증 — 합성 데이터로 build_xy·트리밍·payload 스키마만 확인.
실제 RF/XGB 풀 학습(성능 검증)은 하지 않는다: 목표는 로직 검증."""
import numpy as np
import pandas as pd
import pytest

from ml.train_expect import (
    FEATURES,
    TARGETS,
    build_xy,
    doy_solar_climatology,
    fit_full,
    trim_outliers,
)


def _make_df(n_groups=6, n_per_group=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    years = [2021, 2022, 2023]
    farms = [1, 2]
    seasons = [1, 2]
    group_keys = [(y, f, s) for y in years for f in farms for s in seasons][:n_groups]
    for (year, farm, season) in group_keys:
        base_doy = rng.integers(1, 300)
        for i in range(n_per_group):
            doy = ((base_doy + i) % 365) + 1
            outer = 15 + 10 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1)
            solar = 200 + 100 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 5)
            inner_mean = outer + 5 + rng.normal(0, 0.5)
            inner_min = inner_mean - 3 + rng.normal(0, 0.3)
            rows.append({
                "연도": year, "농가명": farm, "작기": season,
                "품목": "방울토마토" if (farm + season) % 2 == 0 else "완숙토마토",
                "날짜": pd.Timestamp("2021-01-01") + pd.to_timedelta(int(doy) - 1, unit="D"),
                "온도내부_평균": inner_mean,
                "온도내부_최저": inner_min,
                "온도외부_평균": outer,
                "일사량_평균": max(solar, 0),
            })
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def synth_df():
    return _make_df()


def test_build_xy_shapes(synth_df):
    X, y_by_target, groups, features = build_xy(synth_df)
    assert features == FEATURES
    assert X.shape == (len(synth_df), len(FEATURES))
    assert isinstance(y_by_target, dict)
    assert set(y_by_target.keys()) == set(TARGETS.keys())
    for name, y in y_by_target.items():
        assert isinstance(y, np.ndarray)
        assert y.shape == (len(synth_df),)
    assert isinstance(groups, np.ndarray)
    assert groups.shape == (len(synth_df),)
    # 전일 내부온도(누수 위험 피처) 미포함 확인
    assert "온도내부_평균" not in features
    assert "온도내부_최저" not in features


def test_doy_encoding_boundaries():
    df = pd.DataFrame({
        "연도": [2022, 2022],
        "농가명": [1, 1],
        "작기": [1, 1],
        "품목": ["방울토마토", "방울토마토"],
        "날짜": ["2022-01-01", "2022-12-31"],  # doy=1, doy=365
        "온도내부_평균": [20.0, 20.0],
        "온도내부_최저": [18.0, 18.0],
        "온도외부_평균": [10.0, 10.0],
        "일사량_평균": [100.0, 100.0],
    })
    X, _, _, features = build_xy(df)
    sin_idx = features.index("doy_sin")
    cos_idx = features.index("doy_cos")

    # doy=1
    expected_sin_1 = np.sin(2 * np.pi * 1 / 365)
    expected_cos_1 = np.cos(2 * np.pi * 1 / 365)
    assert X[0, sin_idx] == pytest.approx(expected_sin_1, abs=1e-6)
    assert X[0, cos_idx] == pytest.approx(expected_cos_1, abs=1e-6)

    # doy=365 (연말 근처, 0에 가까워야 함)
    expected_sin_365 = np.sin(2 * np.pi * 365 / 365)
    expected_cos_365 = np.cos(2 * np.pi * 365 / 365)
    assert X[1, sin_idx] == pytest.approx(expected_sin_365, abs=1e-6)
    assert X[1, cos_idx] == pytest.approx(expected_cos_365, abs=1e-6)
    # 연초/연말 근접성 확인(주기적 인코딩 취지)
    assert abs(X[0, sin_idx] - X[1, sin_idx]) < 0.05
    assert abs(X[0, cos_idx] - X[1, cos_idx]) < 0.05


def test_trim_outliers_removes_injected_extremes(synth_df):
    df = synth_df.copy()
    # 극단 이상치 5행 주입 (내부온도를 비현실적으로 튀게)
    n_inject = 5
    idx = df.index[:n_inject]
    df.loc[idx, "온도내부_평균"] = df.loc[idx, "온도내부_평균"] + 100
    df.loc[idx, "온도내부_최저"] = df.loc[idx, "온도내부_최저"] + 100

    X, y_by_target, groups, _ = build_xy(df)
    X_trim, y_trim, groups_trim, keep_mask = trim_outliers(X, y_by_target, groups)

    # 주입한 이상치 행들이 제거되었는지 확인
    assert not keep_mask[:n_inject].any()
    assert X_trim.shape[0] < X.shape[0]
    assert X_trim.shape[0] == int(keep_mask.sum())
    for name in y_by_target:
        assert y_trim[name].shape[0] == X_trim.shape[0]
    assert groups_trim.shape[0] == X_trim.shape[0]


def test_doy_solar_climatology_covers_full_range(synth_df):
    from ml.train_expect import _doy_encode
    df_enc = _doy_encode(synth_df)
    clim = doy_solar_climatology(df_enc)
    assert isinstance(clim, dict)
    assert set(clim.keys()) == set(range(1, 367))
    assert all(isinstance(v, float) for v in clim.values())
    assert all(v >= 0 for v in clim.values())


def test_payload_schema_minimal(synth_df, tmp_path, monkeypatch):
    """실제 RF/XGB 풀 학습은 하지 않고, 얕은 RF로 실제 파이프라인
    (trim_outliers → second_pass_oof → fit_full)을 통과시켜
    OOF 기반 resid_sigma 산출과 payload 스키마(키 존재)를 검증한다."""
    import ml.train_expect as te

    X, y_by_target, groups, features = build_xy(synth_df)
    X_trim, y_trim, groups_trim, keep_mask = trim_outliers(X, y_by_target, groups)

    # 학습 비용을 줄이기 위해 build_models를 가벼운 RF로 임시 대체
    def _light_models():
        from sklearn.ensemble import RandomForestRegressor
        return {"RandomForest": RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)}

    monkeypatch.setattr(te, "build_models", _light_models)

    # 실제 second_pass_oof 호출 — OOF 기반 resid_sigma·MAE 산출 검증
    oof, resid_sigma, mae = te.second_pass_oof(X_trim, y_trim, groups_trim, "RandomForest")
    for name in y_trim:
        assert oof[name].shape == y_trim[name].shape
        assert resid_sigma[name] > 0
        assert mae[name] >= 0
        # OOF 잔차 std와 일치해야 함(train 잔차 아님)
        expected_sigma = float(np.std(y_trim[name] - oof[name]))
        assert resid_sigma[name] == pytest.approx(expected_sigma, rel=1e-9)

    models = {name: fit_full("RandomForest", X_trim, y) for name, y in y_trim.items()}

    from ml.train_expect import _doy_encode
    df_trim = _doy_encode(synth_df).loc[keep_mask].reset_index(drop=True)
    solar_clim = doy_solar_climatology(df_trim)

    payload = {
        "models": models,
        "features": features,
        "resid_sigma": resid_sigma,
        "doy_solar_climatology": solar_clim,
        "metrics": {"note": "test payload"},
    }

    for key in ("models", "features", "resid_sigma", "doy_solar_climatology", "metrics"):
        assert key in payload

    out = tmp_path / "test_env_expect_reg.pkl"
    import joblib
    joblib.dump(payload, out)
    loaded = joblib.load(out)
    for key in ("models", "features", "resid_sigma", "doy_solar_climatology", "metrics"):
        assert key in loaded
    assert set(loaded["models"].keys()) == set(TARGETS.keys())
