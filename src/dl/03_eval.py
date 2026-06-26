"""
Phase 2 (DL) · STEP 3 · 평가 — 강건화 · 검증 (토마토 모델)

이 파일 하나 = STEP 3 전체. 청크 2개가 블록으로 들어있다:
  · 청크 2-7  과적합·불균형·학습안정 대응   — 클래스 가중치 효과를 소수클래스 recall 로 실증
  · 청크 2-9  평가 심화                       — 혼동행렬 · ROC/AUC · 오분류(FN) 사례

STEP 2의 전이학습(2-5) 위에서 동작한다:
  · 2-7 = 토마토 데이터 필요 → 먼저 `python src/dl/prepare_tomato.py`
  · 2-9 = 2-5 가 저장한 models/tomato_resnet18.pt 필요 → 먼저 `02_core.py --chunk 2-5`

인과 흐름(2-7 → 2-9): 2-7에서 불균형을 가중치로 잡고 → 2-9에서 'FN(놓친 질병)이 실제로 줄었나'를 검증.

실행:
  python src/dl/03_eval.py --chunk 2-7   # 불균형 대응(클래스 가중치)
  python src/dl/03_eval.py --chunk 2-9   # 평가심화(혼동행렬·ROC·FN)
출력 그림 → docs/figures/phase2_dl/
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
TOMATO = f"{ROOT}/data/tomato"
MODELS = f"{ROOT}/models"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_resnet18(n_classes=2):
    """2-5와 동일한 방식: ImageNet 백본 freeze + head 교체."""
    from torchvision import models
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def _transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _train_head(model, loader, weight=None, epochs=2):
    crit = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ════════════════════════════════════════════════════════════════════
# 청크 2-7 · 과적합·불균형·학습안정 대응
#   ① 토마토 train 을 '인위적 불균형'(정상 다수 · 질병 소수)으로 만들고
#   ② 가중치 없이 vs CrossEntropyLoss(weight=클래스역빈도) 두 모델 학습
#   ③ 소수클래스(질병) recall 비교 → 가중치가 '놓친 질병'을 줄이는지 실증
# ════════════════════════════════════════════════════════════════════
def _per_class_recall(model, loader, n_classes=2):
    model.eval()
    hit = np.zeros(n_classes); tot = np.zeros(n_classes)
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            yb = yb.numpy()
            for c in range(n_classes):
                m = yb == c
                tot[c] += m.sum(); hit[c] += (pred[m] == c).sum()
    return hit / np.maximum(tot, 1)


def run_chunk_2_7():
    print("\n" + "═" * 64 + "\n청크 2-7 · 과적합·불균형·학습안정 대응\n" + "═" * 64)
    if not os.path.isdir(f"{TOMATO}/train"):
        print("⚠️ 토마토 데이터 없음 → python src/dl/prepare_tomato.py")
        return
    from torchvision import datasets

    tf = _transforms()
    train_full = datasets.ImageFolder(f"{TOMATO}/train", tf)
    val_ds = datasets.ImageFolder(f"{TOMATO}/val", tf)
    classes = train_full.classes
    dis = classes.index("disease")
    targets = np.array(train_full.targets)

    # ① 인위적 불균형: 정상 전부 + 질병 1/3 만 (소수클래스 상황 연출)
    rng = np.random.default_rng(0)
    normal_idx = np.where(targets != dis)[0]
    disease_idx = np.where(targets == dis)[0]
    dis_few = rng.choice(disease_idx, max(len(disease_idx) // 3, 5), replace=False)
    imb_idx = np.concatenate([normal_idx, dis_few])
    imb_ds = Subset(train_full, imb_idx)
    n_normal, n_dis = len(normal_idx), len(dis_few)
    print(f"\n[①] 불균형 train: 정상 {n_normal} : 질병 {n_dis}  (비율 {n_normal/max(n_dis,1):.1f}:1)")

    loader = DataLoader(imb_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 클래스 역빈도 가중치 (적은 클래스에 큰 가중치)
    counts = np.array([n_dis if c == dis else n_normal for c in range(len(classes))], dtype=float)
    weight = torch.tensor((counts.sum() / counts) / (counts.sum() / counts).sum() * len(classes),
                          dtype=torch.float32)

    # ② 두 모델 학습
    print("\n[②] 가중치 없이 vs 클래스 가중치")
    results = {}
    for tag, w in [("가중치 X", None), ("가중치 O", weight)]:
        torch.manual_seed(0)
        model = _build_resnet18(len(classes))
        model = _train_head(model, loader, weight=w, epochs=2)
        rec = _per_class_recall(model, val_loader, len(classes))
        results[tag] = rec
        print(f"  {tag}: recall(정상)={rec[1-dis]:.3f}  recall(질병)={rec[dis]:.3f}")

    # ③ 질병 recall 비교 그림
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tags = list(results.keys())
    dis_rec = [results[t][dis] for t in tags]
    nor_rec = [results[t][1 - dis] for t in tags]
    x = np.arange(len(tags)); w_ = 0.35
    ax.bar(x - w_/2, dis_rec, w_, label="질병 recall (소수)", color="#e45756")
    ax.bar(x + w_/2, nor_rec, w_, label="정상 recall (다수)", color="#4c78a8")
    ax.set_xticks(x); ax.set_xticklabels(tags); ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall"); ax.set_title("클래스 가중치 효과 — 소수클래스(질병) 놓침 줄이기")
    ax.legend()
    for i, v in enumerate(dis_rec):
        ax.text(i - w_/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = f"{FIGS}/07_imbalance.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n[③] 그림 저장 → {path}")
    print("  가중치를 주면 소수클래스(질병)에 손실을 더 크게 매겨 → 질병 recall(놓친 병 ↓) 개선.")


# ════════════════════════════════════════════════════════════════════
# 청크 2-9 · 평가 심화 — 혼동행렬 · ROC/AUC · 오분류(FN)
#   ① 2-5 저장 모델로 val 전체 예측 → 혼동행렬(어디서 헷갈리나)
#   ② ROC 곡선·AUC (이진분류 임계값 전반의 성능)
#   ③ 오분류, 특히 FN(질병을 정상으로 놓침) 사례를 눈으로
# ════════════════════════════════════════════════════════════════════
def run_chunk_2_9():
    print("\n" + "═" * 64 + "\n청크 2-9 · 평가 심화 (혼동행렬·ROC/AUC·FN)\n" + "═" * 64)
    ckpt = f"{MODELS}/tomato_resnet18.pt"
    if not os.path.exists(ckpt):
        print("⚠️ 2-5 학습 모델 없음 → python src/dl/02_core.py --chunk 2-5")
        return
    from torchvision import datasets
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

    tf = _transforms()
    val_ds = datasets.ImageFolder(f"{TOMATO}/val", tf)
    classes = val_ds.classes
    dis = classes.index("disease")
    loader = DataLoader(val_ds, batch_size=32)

    model = _build_resnet18(len(classes))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)

    # 전체 예측 + 질병 확률 수집
    y_true, y_pred, p_dis = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            prob = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            y_pred.extend(prob.argmax(1)); p_dis.extend(prob[:, dis]); y_true.extend(yb.numpy())
    y_true, y_pred, p_dis = np.array(y_true), np.array(y_pred), np.array(p_dis)

    print("\n[①] classification report")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    # ROC (질병=양성)
    fpr, tpr, _ = roc_curve(y_true == dis, p_dis)
    roc_auc = auc(fpr, tpr)
    print(f"[②] ROC-AUC(질병 양성) = {roc_auc:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(len(classes))); ax1.set_xticklabels(classes)
    ax1.set_yticks(range(len(classes))); ax1.set_yticklabels(classes)
    ax1.set_xlabel("예측"); ax1.set_ylabel("실제"); ax1.set_title("혼동행렬")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax1.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=13)
    ax2.plot(fpr, tpr, lw=2, color="#e45756", label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], ls="--", color="gray")
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR(질병 검출률)")
    ax2.set_title("ROC 곡선 (질병=양성)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/09_eval.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  혼동행렬·ROC 그림 저장 → {path}")

    # ③ FN(질병을 정상으로 놓침) 사례 — 농업에선 가장 위험한 오류
    fn_idx = np.where((y_true == dis) & (y_pred != dis))[0]
    print(f"\n[③] FN(놓친 질병) {len(fn_idx)}건", end="")
    if len(fn_idx):
        show = fn_idx[:3]
        fig, axes = plt.subplots(1, len(show), figsize=(4*len(show), 4))
        if len(show) == 1:
            axes = [axes]
        for ax, idx in zip(axes, show):
            img = val_ds[idx][0].numpy().transpose(1, 2, 0)
            img = (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
            ax.imshow(img); ax.set_title(f"실제 질병 → 정상 오판\n(질병확률 {p_dis[idx]:.2f})")
            ax.axis("off")
        fig.suptitle("FN 사례 — '병든 잎을 건강하다' 놓친 케이스(가장 위험)", fontsize=12)
        fig.tight_layout()
        path2 = f"{FIGS}/09_false_negatives.png"
        fig.savefig(path2, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f" → 사례 그림 저장 {path2}")
    else:
        print(" — 놓친 질병 없음(이 val 셋 기준).")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 STEP 3 평가 — 청크 2-7·2-9")
    parser.add_argument("--chunk", choices=["2-7", "2-9", "all"], default="all")
    args = parser.parse_args()

    runners = {"2-7": run_chunk_2_7, "2-9": run_chunk_2_9}
    if args.chunk == "all":
        for run in runners.values():
            run()
        print("\n✅ STEP 3(평가) 완료 — 다음 STEP 4(데모): 모델 저장 + Streamlit")
    else:
        runners[args.chunk]()
        print(f"\n✅ 청크 {args.chunk} 완료")
