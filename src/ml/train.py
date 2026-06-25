"""
Phase 1 (ML) — 학습·평가: 환경 일별 통계 → 작물 8종 분류

모델 3종 비교(로지스틱·RandomForest·XGBoost) + 평가 3겹:
  ① test set Accuracy/F1
  ② StratifiedKFold 교차검증 (낙관적)
  ③ GroupKFold(농가+작기) 교차검증 (누수 없는 현실적 일반화) ★ Phase1 데이터누수 교훈 적용
베스트 모델 → models/ 저장, 그림 → docs/figures/phase1_ml/
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
DATA = f"{ROOT}/data/processed/env_daily.csv"
MODELS = f"{ROOT}/models"
FIGS = f"{ROOT}/docs/figures/phase1_ml"
os.makedirs(MODELS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FEATURES = ["온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
            "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균"]
TARGET = "품목"


def load():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    X = df[FEATURES]
    y = df[TARGET]
    groups = df["연도"].astype(str) + "_" + df["농가명"].astype(str) + "_" + df["작기"].astype(str)
    return df, X, y, groups


def evaluate_models(X_tr, X_te, y_tr, y_te):
    """3모델 학습 + test 평가. (스케일링은 로지스틱만 파이프라인으로)"""
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="mlogloss"),
    }
    results = {}
    for name, model in models.items():
        if name == "XGBoost":
            le = LabelEncoder()
            model.fit(X_tr, le.fit_transform(y_tr))
            pred = le.inverse_transform(model.predict(X_te))
        else:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        results[name] = {"model": model, "acc": acc, "f1": f1, "pred": pred}
        print(f"  {name:20} Acc={acc:.3f}  F1(macro)={f1:.3f}")
    return results


def plot_confusion(y_te, pred, labels, title, path):
    cm = confusion_matrix(y_te, pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("실제")
    plt.xlabel("예측")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(model, path):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color="seagreen")
    plt.title("피처 중요도 (RandomForest)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_model_compare(results, path):
    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 5))
    plt.bar(x - 0.2, accs, 0.4, label="Accuracy", color="seagreen")
    plt.bar(x + 0.2, f1s, 0.4, label="F1(macro)", color="darkorange")
    plt.xticks(x, names, rotation=10)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("모델 비교 — 환경→작물 분류")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    print("[1] 데이터 로드")
    df, X, y, groups = load()
    print(f"  {X.shape[0]}행 × {X.shape[1]}피처, 작물 {y.nunique()}종")

    print("[2] train/test 분리 (stratify)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("[3] 모델 3종 학습·평가 (test set)")
    results = evaluate_models(X_tr, X_te, y_tr, y_te)

    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    print(f"\n  → 베스트(F1 기준): {best_name}")

    print("[4] 평가 3겹 — 교차검증")
    labels = sorted(y.unique())
    # ② StratifiedKFold (낙관적)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    skf = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="f1_macro", n_jobs=-1)
    # ③ GroupKFold (농가+작기 누수 차단 — 현실적)
    gkf = cross_val_score(rf, X, y, cv=GroupKFold(5), groups=groups,
                          scoring="f1_macro", n_jobs=-1)
    print(f"  ② Stratified 5-Fold F1: {skf.mean():.3f} ± {skf.std():.3f} (낙관적)")
    print(f"  ③ Group(농가+작기) 5-Fold F1: {gkf.mean():.3f} ± {gkf.std():.3f} (현실적·누수 차단)")

    print("[5] 그림 저장")
    plot_confusion(y_te, best["pred"], labels, f"혼동행렬 — {best_name}", f"{FIGS}/confusion_matrix.png")
    rf_full = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_full.fit(X_tr, y_tr)
    plot_feature_importance(rf_full, f"{FIGS}/feature_importance.png")
    plot_model_compare(results, f"{FIGS}/model_compare.png")

    print("[6] 베스트 모델 저장 (+데모용 통계 동봉 → 배포 앱이 csv 없이 자립 동작)")
    ranges = {f: (float(df[f].min()), float(df[f].max()), float(df[f].median())) for f in FEATURES}
    payload = {
        "model": best["model"], "features": FEATURES, "labels": labels, "model_name": best_name,
        "ranges": ranges,                                  # 슬라이더 (min, max, median)
        "crop_mean": df.groupby(TARGET)[FEATURES].mean(),  # 작물별 평균
        "crop_min": df.groupby(TARGET)[FEATURES].min(),
        "crop_max": df.groupby(TARGET)[FEATURES].max(),
    }
    joblib.dump(payload, f"{MODELS}/phase1_crop_env_clf.pkl")
    print(f"  저장 → models/phase1_crop_env_clf.pkl")

    # 결과 요약 저장 (md 작성용)
    with open(f"{FIGS}/_metrics.txt", "w") as f:
        for n in results:
            f.write(f"{n}\tAcc={results[n]['acc']:.3f}\tF1={results[n]['f1']:.3f}\n")
        f.write(f"BEST\t{best_name}\n")
        f.write(f"StratifiedKFold_F1\t{skf.mean():.3f}+-{skf.std():.3f}\n")
        f.write(f"GroupKFold_F1\t{gkf.mean():.3f}+-{gkf.std():.3f}\n")
        f.write("\nclassification_report(best):\n")
        f.write(classification_report(y_te, best["pred"]))
    print("\n[완료] 결과 → docs/figures/phase1_ml/_metrics.txt")


if __name__ == "__main__":
    main()
