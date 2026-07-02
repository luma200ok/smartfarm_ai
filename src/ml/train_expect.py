"""
이슈 #6 PR-1 — 외기→실내 기대값 회귀 학습·평가 (날씨 feedforward 1단계)

토마토(방울+완숙) 데이터로 "외기 온도·일사량·계절(doy)" → "실내 기대 온도(평균/최저)"
회귀 모델을 학습한다. 배포 모델은 실시간 실내 온도가 이 기대값에서 벗어나면
사전 경보를 낼 때 기준선(baseline)으로 쓰인다.

핵심 설계:
  · 피처에 전일 내부온도 등 내부 실측값을 넣지 않는다(누수 방지) — 오직 외기+계절만 사용.
  · groups = 연도_농가명_작기 (기존 src/ml/train.py 패턴 재사용) → GroupKFold로 누수 없는 평가.
  · 2-pass 트리밍: 1차 GKF OOF 잔차의 3σ로 이상치 제거 → 2차 GKF OOF로 최종 지표 산출.
  · RF vs XGB 비교(GroupKFold neg_MAE) 후 베스트로 전체 재학습, pkl로 배포.
"""
import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from xgboost import XGBRegressor

ROOT = str(Path(__file__).resolve().parents[2])

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

TOMATO_CROPS = ["방울토마토", "완숙토마토"]
FEATURES = ["온도외부_평균", "일사량_평균", "doy_sin", "doy_cos"]
TARGETS = {"평균": "온도내부_평균", "최저": "온도내부_최저"}


RAW_FEATURES = ["온도외부_평균", "일사량_평균"]  # doy_sin/cos 는 날짜에서 파생


def load(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = df[df["품목"].isin(TOMATO_CROPS)].copy()
    # RF는 NaN에서 크래시 — 피처·타깃 컬럼 결측 행 제거 후 진행
    required = RAW_FEATURES + list(TARGETS.values()) + ["날짜"]
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped:
        print(f"  결측 제거 {dropped}행 (피처·타깃 NaN)")
    return df


def _doy_encode(df: pd.DataFrame) -> pd.DataFrame:
    doy = pd.to_datetime(df["날짜"]).dt.dayofyear.astype(float)
    df = df.copy()
    df["doy"] = doy
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return df


def build_xy(df: pd.DataFrame):
    """토마토(방울+완숙) 필터가 적용된(또는 그대로의) df에서 X/y/groups/features 구성.
    전일 내부온도 등 내부 실측값은 피처에서 제외(누수 방지) — 외기+계절만 사용."""
    df = _doy_encode(df)
    X = df[FEATURES].to_numpy(dtype=float)
    y_by_target = {name: df[col].to_numpy(dtype=float) for name, col in TARGETS.items()}
    groups = (df["연도"].astype(str) + "_" + df["농가명"].astype(str) + "_" + df["작기"].astype(str)).to_numpy()
    return X, y_by_target, groups, list(FEATURES)


def build_models():
    """RF vs XGB — 호출마다 fresh 인스턴스."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1),
    }


def _n_splits(groups, n_splits=5):
    """그룹 수가 5 미만인 소규모 데이터에서 GroupKFold 크래시 방지."""
    return min(n_splits, len(np.unique(groups)))


def _gkf_oof(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(_n_splits(groups, n_splits))
    return cross_val_predict(model, X, y, cv=gkf, groups=groups, n_jobs=-1)


def trim_outliers(X, y_by_target, groups, sigma_k=3.0):
    """2-pass GKF OOF 기반 트리밍.
    1차: 모델(RF)로 GKF OOF 예측 → 타깃별 잔차 std(σ) 계산.
    2차: |resid| > sigma_k*σ 인 행을 이상치로 마스킹(전 타깃 OR) 후 제거.
    반환: 트리밍된 X, y_by_target, groups, keep_mask(원본 기준)."""
    n = X.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    model_for_trim = build_models()["RandomForest"]
    for name, y in y_by_target.items():
        pred = _gkf_oof(model_for_trim, X, y, groups)
        resid = y - pred
        # 표준편차 대신 MAD 기반 강건 추정(1.4826*MAD ≈ 정규분포 std) 사용
        # — 표준 std는 이상치 자체에 의해 부풀려져(masking effect) 임계값이 무의미해질 수 있음.
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        sigma = 1.4826 * mad
        outlier = np.abs(resid - med) > (sigma_k * sigma if sigma > 0 else np.inf)
        keep_mask &= ~outlier

    X_trim = X[keep_mask]
    y_trim = {name: y[keep_mask] for name, y in y_by_target.items()}
    groups_trim = groups[keep_mask]
    return X_trim, y_trim, groups_trim, keep_mask


def second_pass_oof(X, y_by_target, groups, model_name):
    """트리밍된 데이터로 2차 GKF OOF 재평가 — 최종 MAE·resid_sigma 산출 기준."""
    oof = {}
    resid_sigma = {}
    mae = {}
    for name, y in y_by_target.items():
        model = build_models()[model_name]
        pred = _gkf_oof(model, X, y, groups)
        resid = y - pred
        oof[name] = pred
        resid_sigma[name] = float(resid.std())
        mae[name] = float(np.abs(resid).mean())
    return oof, resid_sigma, mae


def compare_models(X, y_by_target, groups):
    """RF vs XGB — GroupKFold(5) neg_MAE 비교(타깃 평균). 베스트 모델명 반환."""
    scores = {}
    for model_name in build_models():
        maes = []
        for name, y in y_by_target.items():
            model = build_models()[model_name]
            gkf = GroupKFold(_n_splits(groups))
            cv = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="neg_mean_absolute_error", n_jobs=-1)
            maes.append(-cv.mean())
        scores[model_name] = float(np.mean(maes))
        print(f"  {model_name:15} GKF MAE(평균)={scores[model_name]:.3f}")
    best = min(scores, key=scores.get)
    return best, scores


def compute_baselines(df_trim, y_by_target):
    """3종 baseline MAE — 전체(트리밍 후) 데이터 대비 계산(리포트용, 폴드 분리 없음).
    주의: doy기후평균은 같은 데이터로 만든 in-sample baseline이라 실제보다 유리하게 나온다
    (모델 GKF OOF MAE와 직접 비교 시 baseline이 과대평가됨을 감안할 것)."""
    baselines = {}
    outer_temp = df_trim["온도외부_평균"].to_numpy(dtype=float)
    for name, y in y_by_target.items():
        b = {}
        # (a) 외기 온도 그대로 사용
        b["외기그대로"] = float(np.abs(y - outer_temp).mean())
        # (b) 토마토 전체 평균값으로 항상 예측
        b["전체평균"] = float(np.abs(y - y.mean()).mean())
        # (c) doy별 기후평균(중앙값)으로 예측
        doy = df_trim["doy"].to_numpy(dtype=int)
        clim = pd.Series(y).groupby(doy).median()
        pred_c = pd.Series(doy).map(clim).to_numpy()
        b["doy기후평균"] = float(np.abs(y - pred_c).mean())
        baselines[name] = b
    return baselines


def doy_solar_climatology(df_trim: pd.DataFrame) -> dict:
    """트리밍 후 전체 데이터에서 doy(1~366)별 일사량_평균 중앙값 → 선형 보간으로 결측 채움."""
    doy = df_trim["doy"].to_numpy(dtype=int)
    solar = df_trim["일사량_평균"].to_numpy(dtype=float)
    med = pd.Series(solar).groupby(doy).median()
    full_index = pd.Index(range(1, 367))
    med = med.reindex(full_index)
    med = med.interpolate(limit_direction="both")
    return {int(k): float(v) for k, v in med.items()}


def fit_full(model_name, X, y):
    model = build_models()[model_name]
    model.fit(X, y)
    return model


def plot_model_compare(scores, path):
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    plt.figure(figsize=(6, 5))
    plt.bar(names, vals, color=["seagreen", "darkorange"][:len(names)])
    plt.ylabel("GKF MAE (°C, 평균)")
    plt.title("모델 비교 — 외기→실내 기대값 회귀")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_residual_hist(oof, y_by_target, path):
    plt.figure(figsize=(7, 5))
    for name, pred in oof.items():
        resid = y_by_target[name] - pred
        sns.histplot(resid, label=name, kde=True, stat="density", alpha=0.4)
    plt.axvline(0, color="black", linestyle="--")
    plt.title("잔차 분포 (2차 GKF OOF, 트리밍 후)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(models, features, path):
    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=features).sort_values()
            imp.plot(kind="barh", alpha=0.6, label=name)
    plt.title("피처 중요도")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="외기→실내 기대값 회귀 학습·평가")
    parser.add_argument("--data", default=f"{ROOT}/data/processed/env_daily.csv")
    parser.add_argument("--out", default=f"{ROOT}/models/env_expect_reg.pkl")
    parser.add_argument("--figs", default=f"{ROOT}/docs/figures/expect_regression")
    parser.add_argument("--mlflow-uri", default=f"sqlite:///{ROOT}/mlflow.db")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    print("[1] 데이터 로드 (토마토 방울+완숙)")
    df = load(args.data)
    print(f"  {df.shape[0]}행")

    print("[2] 피처 구성 (외기+계절만, 내부 실측값 제외)")
    X, y_by_target, groups, features = build_xy(df)
    df_enc = _doy_encode(df)

    print("[3] 1차 GKF OOF → 이상치 3σ 트리밍")
    X_trim, y_trim, groups_trim, keep_mask = trim_outliers(X, y_by_target, groups)
    n_removed = int((~keep_mask).sum())
    print(f"  제거 {n_removed}행 / {len(keep_mask)}행 ({n_removed / len(keep_mask):.2%})")
    df_trim = df_enc.loc[keep_mask].reset_index(drop=True)

    print("[4] 모델 비교 (RF vs XGB, GroupKFold neg_MAE)")
    best_name, cmp_scores = compare_models(X_trim, y_trim, groups_trim)
    print(f"  → 베스트: {best_name}")

    print("[5] 2차 GKF OOF (트리밍 후) — 최종 MAE·resid_sigma")
    oof, resid_sigma, gkf_mae = second_pass_oof(X_trim, y_trim, groups_trim, best_name)
    for name in y_trim:
        print(f"  {name:6} GKF MAE={gkf_mae[name]:.3f}  resid_sigma={resid_sigma[name]:.3f}")

    print("[6] Baseline 3종 MAE")
    baselines = compute_baselines(df_trim, y_trim)
    for name, b in baselines.items():
        print(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in b.items()))
    print("  ※ doy기후평균 baseline은 in-sample(폴드 분리 없음) — 모델 OOF MAE보다 유리하게 측정됨")

    print("[7] doy 일사량 기후평균 계산")
    solar_clim = doy_solar_climatology(df_trim)

    print("[8] 그림 저장")
    plot_model_compare(cmp_scores, f"{args.figs}/model_compare.png")
    plot_residual_hist(oof, y_trim, f"{args.figs}/residual_hist.png")

    print("[9] 최종 배포 모델 학습 (트리밍 후 전체 데이터, 타깃별 개별)")
    models = {name: fit_full(best_name, X_trim, y) for name, y in y_trim.items()}
    plot_feature_importance(models, features, f"{args.figs}/feature_importance.png")

    metrics = {
        "model_name": best_name,
        "compare_scores": cmp_scores,
        "gkf_mae": gkf_mae,
        "baselines": baselines,
        "baselines_note": "doy기후평균 baseline은 in-sample(폴드 분리 없음) — 모델 GKF OOF MAE 대비 유리하게 측정",
        "n_removed": n_removed,
        "n_total": int(len(keep_mask)),
        "removal_rate": n_removed / len(keep_mask),
    }

    payload = {
        "models": models,
        "features": features,
        "resid_sigma": resid_sigma,
        "doy_solar_climatology": solar_clim,
        "metrics": metrics,
    }
    joblib.dump(payload, args.out)
    print(f"\n[완료] 저장 → {args.out}")

    print("[10] MLflow 기록")
    import mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("phase_integration_expect")
    with mlflow.start_run(run_name=best_name):
        mlflow.log_params({"model": best_name, "features": ",".join(features)})
        for name in y_trim:
            mlflow.log_metric(f"gkf_mae_{name}", gkf_mae[name])
            mlflow.log_metric(f"resid_sigma_{name}", resid_sigma[name])
            for b_name, b_val in baselines[name].items():
                mlflow.log_metric(f"baseline_{b_name}_{name}", b_val)
        mlflow.log_metric("removal_rate", metrics["removal_rate"])
        mlflow.log_artifact(f"{args.figs}/model_compare.png")
        mlflow.log_artifact(f"{args.figs}/residual_hist.png")
        mlflow.log_artifact(f"{args.figs}/feature_importance.png")
    print("  MLflow 기록 완료")


if __name__ == "__main__":
    main()
