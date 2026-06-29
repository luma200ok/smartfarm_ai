"""
Phase 2 (DL) 보조 시각화 — 발표용 EDA · 학습곡선 그림 생성.

산출:
  docs/figures/phase2_dl/11_eda.png            토마토 클래스 분포 + 샘플 잎 3종 + 환경 시계열
  docs/figures/phase2_dl/12_learning_curve.png 전이학습(ResNet18 head) epoch별 loss·val acc

실행:
  python src/dl/eda_viz.py            # 둘 다
  python src/dl/eda_viz.py --only eda
  python src/dl/eda_viz.py --only curve
"""
import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
TOMATO = f"{ROOT}/data/tomato"
ENV_CSV = f"{ROOT}/data/processed/env_daily.csv"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

device = "mps" if torch.backends.mps.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
KOR = {"normal": "정상", "leaf_mold": "잎곰팡이병", "tylcv": "황화잎말이"}


# ════════════════════════════════════════════════════════════════════
# ① EDA — 클래스 분포 + 샘플 잎 + 환경 시계열
# ════════════════════════════════════════════════════════════════════
def make_eda():
    print("[EDA] 그림 생성 중 …")
    # 확보 가능한 원천 이미지 수(불균형 근거) — prepare_tomato 의 인덱싱 재사용
    try:
        from prepare_tomato import _collect
        buckets = _collect()
        avail = {c: len(buckets[c]) for c in ("normal", "leaf_mold", "tylcv")}
    except Exception as e:
        print(f"  (원천 카운트 스킵: {e}) → 폴더 기준으로 대체")
        avail = {c: len(glob.glob(f"{TOMATO}/train/{c}/*.jpg")) +
                    len(glob.glob(f"{TOMATO}/val/{c}/*.jpg"))
                 for c in ("normal", "leaf_mold", "tylcv")}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # [0,0] 클래스 분포 (로그 스케일 — 정상 ≫ 질병)
    ax = axes[0, 0]
    cs = ["normal", "leaf_mold", "tylcv"]
    vals = [avail[c] for c in cs]
    bars = ax.bar([KOR[c] for c in cs], vals, color=["#4c78a8", "#e45756", "#f58518"])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("확보 이미지 수 (log)")
    ratio = avail["normal"] / max(avail["leaf_mold"] + avail["tylcv"], 1)
    ax.set_title(f"클래스 분포 — 정상 {avail['normal']} vs 질병 {avail['leaf_mold']+avail['tylcv']}"
                 f"  (≈{ratio:.0f}:1 불균형)")

    # [0,1][0,2][1,0] 클래스별 샘플 잎
    panel = [axes[0, 1], axes[0, 2], axes[1, 0]]
    for ax, c in zip(panel, cs):
        fs = sorted(glob.glob(f"{TOMATO}/train/{c}/*.jpg"))
        if fs:
            ax.imshow(Image.open(fs[0]).convert("RGB"))
        ax.set_title(f"샘플 — {KOR[c]}")
        ax.axis("off")

    # [1,1] 대표 환경 시계열 (내부·외부온도)
    ax = axes[1, 1]
    if os.path.exists(ENV_CSV):
        import pandas as pd
        df = pd.read_csv(ENV_CSV)
        g = df.groupby(["도", "시군", "농가명", "작기", "품목"])
        key = max(g.groups, key=lambda k: len(g.get_group(k)))
        sdf = g.get_group(key).sort_values("날짜")
        ax.plot(sdf["온도내부_평균"].to_numpy(), label="내부온도", color="#e45756", lw=1.5)
        ax.plot(sdf["온도외부_평균"].to_numpy(), label="외부온도", color="#4c78a8", lw=1.2)
        ax.set_title("대표 농가 환경 시계열"); ax.set_xlabel("일자"); ax.set_ylabel("℃")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    # [1,2] 내부온도 분포
    ax = axes[1, 2]
    if os.path.exists(ENV_CSV):
        ax.hist(df["온도내부_평균"].dropna(), bins=40, color="#54a24b", alpha=0.85)
        ax.set_title("내부온도 분포(전체 농가)"); ax.set_xlabel("내부온도 평균(℃)"); ax.set_ylabel("빈도")
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    fig.suptitle("데이터 탐색(EDA) — 토마토 잎 3종 + 환경 시계열", fontsize=14)
    fig.tight_layout()
    path = f"{FIGS}/11_eda.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  저장 → {path}")


# ════════════════════════════════════════════════════════════════════
# ② 학습곡선 — 전이학습 epoch별 train loss · val acc
# ════════════════════════════════════════════════════════════════════
def make_learning_curve(epochs=10):
    print("[학습곡선] ResNet18 head 학습 중 …")
    from torchvision import datasets, transforms, models
    tf_tr = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tf_va = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_ds = datasets.ImageFolder(f"{TOMATO}/train", tf_tr)
    val_ds = datasets.ImageFolder(f"{TOMATO}/val", tf_va)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, len(train_ds.classes))
    m.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)

    losses, accs = [], []
    for e in range(epochs):
        m.train(); tot = 0.0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        m.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = m(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item(); total += len(yb)
        losses.append(tot / n); accs.append(correct / total)
        print(f"  epoch {e+1:2d} | train loss {losses[-1]:.3f} | val acc {accs[-1]:.3f}")

    xs = range(1, epochs + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(xs, losses, "-o", color="#e45756", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#e45756")
    ax1.tick_params(axis="y", labelcolor="#e45756"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, accs, "-s", color="#4c78a8", label="val acc")
    ax2.set_ylabel("val accuracy", color="#4c78a8"); ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#4c78a8")
    ax1.set_title(f"전이학습 학습곡선 — loss↓·val acc↑ (best {max(accs):.3f})")
    fig.tight_layout()
    path = f"{FIGS}/12_learning_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  저장 → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 보조 시각화 — EDA · 학습곡선")
    parser.add_argument("--only", choices=["eda", "curve"], default=None)
    args = parser.parse_args()
    if args.only in (None, "eda"):
        make_eda()
    if args.only in (None, "curve"):
        make_learning_curve()
    print("✅ 보조 시각화 완료")
