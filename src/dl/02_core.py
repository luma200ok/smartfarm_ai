"""
Phase 2 (DL) · STEP 2 · 핵심 — 모델 구축 (비전 + 시계열)

이 파일 하나 = STEP 2 전체. 청크 4개가 블록으로 들어있다:
  · 청크 2-4  CNN 기초 (Conv·Pooling)        — FashionMNIST, 데이터 자동 다운(불필요 입력 0)
  · 청크 2-5  전이학습 (여러 백본 비교) ⭐     — 토마토 잎 정상/질병, ImageNet 사전학습 재사용
  · 청크 2-6  Grad-CAM (설명가능 AI) ⭐        — 2-5 모델이 '어느 병반'을 보고 판단했나
  · 청크 2-8  LSTM (환경 시계열)               — env_daily.csv 온도 추세 예측

STEP 1과 다른 점: 여기선 손계산 대조가 없다. nn.Conv2d·사전학습 백본을 '믿고' 실전 코드를 쓴다.
  · 2-4·2-8 = 데이터 준비 불필요(내장/기존 CSV) → 바로 실행됨
  · 2-5·2-6 = 토마토 실데이터 필요 → 먼저 `python src/dl/prepare_tomato.py`

실행:
  python src/dl/02_core.py --chunk 2-4   # CNN 기초 (자동 다운로드)
  python src/dl/02_core.py --chunk 2-8   # LSTM 시계열
  python src/dl/02_core.py --chunk 2-5   # 전이학습 (토마토 데이터 준비 후)
  python src/dl/02_core.py --chunk 2-6   # Grad-CAM (2-5 학습 후)
출력 그림 → docs/figures/phase2_dl/ · 모델 → models/
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
DATA_TV = f"{ROOT}/data/torchvision"      # FashionMNIST 캐시
TOMATO = f"{ROOT}/data/tomato"            # prepare_tomato.py 산출물(ImageFolder)
MODELS = f"{ROOT}/models"
ENV_CSV = f"{ROOT}/data/processed/env_daily.csv"
for d in (FIGS, DATA_TV, MODELS):
    os.makedirs(d, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════
# 청크 2-4 · CNN 기초 — Conv · Pooling   (FashionMNIST, 데이터 자동 다운)
#   ① Conv = 작은 필터가 이미지를 훑어 '특징맵'을 만든다 (FC 와 달리 위치 공유)
#   ② Pooling = 특징맵을 절반으로 줄여 계산↓·위치 약간 무시
#   ③ Conv→Pool 쌓고 마지막에 FC → 10종 분류 + 특징맵 눈으로 확인
# ════════════════════════════════════════════════════════════════════
# ── ① CNN 모델 정의: Conv·Pool 2겹 + FC ──
class SmallCNN(nn.Module):
    """28×28 흑백 → 10클래스. Conv 2겹 + Pool 2번 + FC 1개."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 1채널→16개 필터
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16→32
        self.pool = nn.MaxPool2d(2)                                # 크기 절반
        self.fc = nn.Linear(32 * 7 * 7, n_classes)                # 28→14→7 이므로 7×7

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # (B,16,28,28) → (B,16,14,14)
        x = self.pool(torch.relu(self.conv2(x)))   # (B,32,14,14) → (B,32,7,7)
        x = x.flatten(1)                            # (B, 32*7*7) 펼치기
        return self.fc(x)


# ── ② 학습·특징맵 시각화 실행 ──
def run_chunk_2_4():
    print("\n" + "═" * 64 + "\n청크 2-4 · CNN 기초 (Conv·Pooling)\n" + "═" * 64)
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor()])
    print("\n[준비] FashionMNIST 다운로드/로드 (최초 1회 ~30MB)")
    train = datasets.FashionMNIST(DATA_TV, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA_TV, train=False, download=True, transform=tf)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    test_loader = DataLoader(test, batch_size=256)

    model = SmallCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ① 학습 (2 epoch 면 개념 확인엔 충분)
    print("\n[①] CNN 학습 (2 epoch)")
    EPOCHS = 2
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # 매 epoch 테스트 정확도
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item(); total += len(yb)
        print(f"  epoch {epoch+1} | test acc = {correct/total:.3f}")

    # ② Conv 특징맵 시각화 — 첫 conv 층이 한 이미지에서 뽑은 16채널 중 8개
    print("\n[②] conv1 특징맵 시각화")
    model.eval()
    img, label = test[0]
    with torch.no_grad():
        feat = torch.relu(model.conv1(img.unsqueeze(0).to(device)))[0].cpu()  # (16,28,28)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes[0, 0].imshow(img.squeeze(), cmap="gray"); axes[0, 0].set_title("입력 이미지")
    axes[0, 0].axis("off")
    # 나머지 9칸에 특징맵 9개
    for i, ax in enumerate(axes.ravel()[1:]):
        ax.imshow(feat[i], cmap="viridis"); ax.set_title(f"필터 {i}")
        ax.axis("off")
    fig.suptitle("conv1 특징맵 — 필터마다 다른 부분(가장자리·질감)에 반응", fontsize=13)
    fig.tight_layout()
    path = f"{FIGS}/04_feature_maps.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  특징맵 그림 저장 → {path}")
    print("  → Conv 는 한 필터(3×3)를 이미지 전체에 공유해 '어디에 있든' 같은 특징을 잡는다(FC 와 차이).")


# ════════════════════════════════════════════════════════════════════
# 청크 2-5 · 전이학습 — 여러 백본 비교 (토마토 잎 정상/질병) ⭐
#   ① ImageNet 으로 미리 학습된 백본을 가져와 '특징 추출기'로 재사용(freeze)
#   ② 마지막 분류기(head)만 우리 2클래스로 교체해 학습 → 적은 데이터로 고성능
#   ③ resnet18 vs mobilenet_v2 비교 → 같은 데이터에 어느 백본이 유리한가
# ════════════════════════════════════════════════════════════════════
# ── ① 백본 준비: 사전학습 모델 호출 → 몸통 freeze → 머리(분류기) 교체 ──
def _build_backbone(name, n_classes=2):
    """사전학습 백본을 가져와 backbone 은 얼리고(head 만 학습) 분류기를 교체."""
    from torchvision import models
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False                       # backbone freeze
        m.fc = nn.Linear(m.fc.in_features, n_classes)     # head 만 새로(학습 대상)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(name)
    return m


# ── ② 데이터 준비: 토마토 사진을 224 크기·정규화로 가공해 공급(교재 준비) ──
def _tomato_loaders(batch=32):
    from torchvision import datasets, transforms
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    train_ds = datasets.ImageFolder(f"{TOMATO}/train", tf_train)
    val_ds = datasets.ImageFolder(f"{TOMATO}/val", tf_val)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True),
            DataLoader(val_ds, batch_size=batch),
            train_ds.classes)


# ── ③ 학습: freeze 된 몸통 위에서 머리(head)만 학습 ──
def _train_head(model, train_loader, val_loader, epochs=3, epoch_cb=None):
    """freeze 된 backbone 위에서 head 만 학습. (val 정확도 반환)
    epoch_cb(epoch, val_acc, train_loss) 가 주어지면 epoch 마다 호출(MLflow 로깅용)."""
    crit = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]   # head 파라미터만
    opt = torch.optim.Adam(params, lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += float(loss) * xb.size(0); seen += xb.size(0)
        acc = _eval_acc(model, val_loader)
        print(f"    epoch {epoch+1} | val acc = {acc:.3f}")
        if epoch_cb:
            epoch_cb(epoch + 1, acc, run_loss / max(seen, 1))
    return acc


# ── (보조) 정확도 측정: 시험모드로 val/test 정확도 계산 ──
def _eval_acc(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += (pred == yb).sum().item(); total += len(yb)
    return correct / max(total, 1)


# ── ④ 실행: 두 백본(resnet18·mobilenet_v2)을 같은 데이터로 학습→비교 ──
def run_chunk_2_5():
    print("\n" + "═" * 64 + "\n청크 2-5 · 전이학습 (백본 비교) ⭐\n" + "═" * 64)
    if not os.path.isdir(f"{TOMATO}/train"):
        print("⚠️ 토마토 데이터가 없습니다.")
        print("   먼저:  python src/dl/prepare_tomato.py")
        return

    train_loader, val_loader, classes = _tomato_loaders()
    print(f"\n클래스 = {classes}  (ImageFolder 가 폴더명으로 자동 라벨링)")

    # MLflow 실험 추적 — 백본별 run 으로 하이퍼파라미터·정확도·가중치를 로컬 mlruns/ 에 기록
    # (서버·레지스트리 없이 파일 스토어. 데모 서빙엔 영향 없음 — 학습 단계 전용)
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")   # MLflow 3.x 권장 DB 백엔드(파일스토어 폐기)
    mlflow.set_experiment("phase2_tomato_backbone")

    results = {}
    for name in ("resnet18", "mobilenet_v2"):
        print(f"\n[백본] {name} — backbone freeze, head 만 학습")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"backbone": name, "strategy": "freeze_backbone+train_head",
                               "epochs": 3, "lr": 1e-3, "batch": 32,
                               "optimizer": "Adam", "n_classes": len(classes)})
            model = _build_backbone(name, n_classes=len(classes))
            acc = _train_head(
                model, train_loader, val_loader, epochs=3,
                epoch_cb=lambda e, a, l: (mlflow.log_metric("val_acc", a, step=e),
                                          mlflow.log_metric("train_loss", l, step=e)))
            results[name] = acc
            mlflow.log_metric("val_acc_final", acc)
            ckpt = f"{MODELS}/tomato_{name}.pt"
            torch.save(model.state_dict(), ckpt)   # 2-6·2-10 에서 재사용
            mlflow.log_artifact(ckpt)
            print(f"    저장 → {ckpt}  (MLflow run='{name}' 기록)")

    # 백본 비교 막대그림
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(list(results.keys()), list(results.values()), color=["#4c78a8", "#e45756"])
    for i, (k, v) in enumerate(results.items()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel("val 정확도")
    ax.set_title("전이학습 백본 비교 — 토마토 정상/질병")
    fig.tight_layout()
    path = f"{FIGS}/05_backbone_compare.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n[비교] 그림 저장 → {path}")
    best = max(results, key=results.get)
    print(f"  best = {best} ({results[best]:.3f}) — 적은 데이터로도 사전학습 특징 재사용 덕에 고성능.")


# ════════════════════════════════════════════════════════════════════
# 청크 2-6 · Grad-CAM — 설명가능 AI ⭐
#   ① 모델이 '어느 픽셀'을 보고 그 클래스로 판단했는지 히트맵으로
#   ② 마지막 conv 층의 활성화 × (그 클래스에 대한) 기울기 → 가중합 = CAM
#   ③ 잎 사진 위에 겹쳐 '병반을 봤는지' 눈으로 검증 (오분류 디버깅에 강력)
# ════════════════════════════════════════════════════════════════════
# ── ① Grad-CAM 히트맵 생성·저장 ──
def run_chunk_2_6():
    print("\n" + "═" * 64 + "\n청크 2-6 · Grad-CAM (설명가능 AI) ⭐\n" + "═" * 64)
    ckpt = f"{MODELS}/tomato_resnet18.pt"
    if not os.path.exists(ckpt):
        print("⚠️ 2-5 에서 학습한 resnet18 가중치가 없습니다.")
        print("   먼저:  python src/dl/02_core.py --chunk 2-5")
        return
    if not os.path.isdir(f"{TOMATO}/val"):
        print("⚠️ 토마토 데이터 없음 → python src/dl/prepare_tomato.py")
        return

    from torchvision import datasets, transforms

    # val 데이터(클래스 수 동적 — 2분류든 3분류든 자동)
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_ds = datasets.ImageFolder(f"{TOMATO}/val", tf)

    model = _build_backbone("resnet18", n_classes=len(val_ds.classes))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)

    # 마지막 conv 블록에 forward hook 으로 활성화를 저장하고 그 텐서의 grad 를 보존(retain_grad).
    # ※ backbone 이 freeze(requires_grad=False)라 입력에 requires_grad 를 켜야 grad 가 layer4 까지 흐른다.
    target_layer = model.layer4[-1]
    store = {}

    def _save_act(m, i, o):
        o.retain_grad()
        store["act"] = o
    target_layer.register_forward_hook(_save_act)
    x, y = val_ds[0]
    xb = x.unsqueeze(0).to(device).requires_grad_(True)   # grad 가 layer4 까지 흐르도록

    logits = model(xb)                       # forward (hook 이 act 저장)
    cls = logits.argmax(1).item()
    model.zero_grad()
    logits[0, cls].backward()                # 예측 클래스 점수로 역전파

    act = store["act"].detach()[0]            # (C,h,w)
    grad = store["act"].grad[0]               # retain_grad 로 보존된 dL/d(act)
    weights = grad.mean(dim=(1, 2))           # 채널별 중요도 = 기울기 공간평균
    cam = torch.relu((weights[:, None, None] * act).sum(0))   # 가중합 후 ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 0~1 정규화
    cam = cam.cpu().numpy()

    # 원본 이미지 복원(정규화 역연산) + 히트맵 오버레이
    import torch.nn.functional as F
    cam_up = F.interpolate(torch.tensor(cam)[None, None], size=(224, 224),
                           mode="bilinear", align_corners=False)[0, 0].numpy()
    img = x.cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img); axes[0].set_title(f"입력 (정답 {val_ds.classes[y]})"); axes[0].axis("off")
    axes[1].imshow(cam_up, cmap="jet"); axes[1].set_title("Grad-CAM 히트맵"); axes[1].axis("off")
    axes[2].imshow(img); axes[2].imshow(cam_up, cmap="jet", alpha=0.45)
    axes[2].set_title(f"오버레이 (예측 {val_ds.classes[cls]})"); axes[2].axis("off")
    fig.suptitle("Grad-CAM — 모델이 '어디를 보고' 그 클래스로 판단했나", fontsize=13)
    fig.tight_layout()
    path = f"{FIGS}/06_gradcam.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n[Grad-CAM] 그림 저장 → {path}")
    print(f"  예측={val_ds.classes[cls]} / 정답={val_ds.classes[y]} — 붉은 영역이 판단 근거(병반이면 OK).")


# ════════════════════════════════════════════════════════════════════
# 청크 2-8 · LSTM — 환경 시계열 (다변량 · 다중 시계열)
#   ① 단변량(온도 1개)→ '다변량'(환경 8변수)로 확장: 외부온도·일사량·CO2·습도 흐름까지 본다
#   ② 시계열 1개 → 농가·작기·품목별 485개 시계열 전부 학습(2022~24 다년, 시간순 분할, 농가 누수 차단)
#   ③ baseline(어제값=오늘)과 MAE 비교 → '단변량이라 baseline 못 이김'을 정면 검증
# ════════════════════════════════════════════════════════════════════
# 환경 8변수(입력) — 첫 번째(온도내부_평균)가 예측 타깃
ENV_FEATURES = ["온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
                "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균"]
SEQ_KEYS = ["도", "시군", "농가명", "작기", "품목"]
WINDOW = 7


# ── ① LSTM 모델 정의: 입력 F개 변수 → hidden → 다음날 내부온도 1개 ──
class TempLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (B, seq, F)
        out, _ = self.lstm(x)             # out: (B, seq, hidden)
        return self.fc(out[:, -1, :])     # 마지막 타임스텝만 → 다음날 1개


# ── ② 다변량·다중 시계열 전처리·학습·예측 실행 ──
def run_chunk_2_8():
    print("\n" + "═" * 64 + "\n청크 2-8 · LSTM (다변량 · 다중 시계열)\n" + "═" * 64)
    if not os.path.exists(ENV_CSV):
        print(f"⚠️ {ENV_CSV} 없음 (Phase1 전처리 산출물 필요)")
        return
    import pandas as pd
    from torch.utils.data import TensorDataset

    df = pd.read_csv(ENV_CSV)
    F = len(ENV_FEATURES)
    # 농가·작기·품목별 시퀀스를 만들고, 각 시퀀스를 '시간순' 80/20 분할(셔플 금지, 농가 누수 차단)
    Xtr_l, Ytr_l, Xva_l, Yva_l, persist_l = [], [], [], [], []
    rep = None                                  # 대표 시각화용(가장 긴 시퀀스)
    n_groups = 0
    for _, gdf in df.groupby(SEQ_KEYS):
        arr = gdf.sort_values("날짜")[ENV_FEATURES].to_numpy(np.float32)
        if len(arr) < WINDOW + 5:
            continue
        n_groups += 1
        split = int(len(arr) * 0.8)
        for i in range(len(arr) - WINDOW):
            t = i + WINDOW                       # 예측 대상 시점
            x = arr[i:t]                         # 과거 WINDOW 일 × F변수
            y = arr[t, 0]                        # 다음날 내부온도(raw ℃)
            if t < split:
                Xtr_l.append(x); Ytr_l.append(y)
            else:
                Xva_l.append(x); Yva_l.append(y)
                persist_l.append(arr[t - 1, 0])  # baseline: 어제값=오늘
        if rep is None or len(arr) > len(rep[0]):
            rep = (arr, split)

    Xtr, Xva = np.array(Xtr_l), np.array(Xva_l)
    Ytr, Yva = np.array(Ytr_l), np.array(Yva_l)
    print(f"\n시계열 그룹 {n_groups}개 · 입력 변수 {F}개 → train {len(Xtr)} / val {len(Xva)} window")

    # 정규화: train 통계로만(시간 누수 방지). 타깃(feature0)은 그 통계로 역변환.
    mu = Xtr.reshape(-1, F).mean(0); sd = Xtr.reshape(-1, F).std(0) + 1e-8
    mu0, sd0 = mu[0], sd[0]
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ytr = (Ytr - mu0) / sd0

    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr).unsqueeze(1)),
                        batch_size=256, shuffle=True)
    model = TempLSTM(n_feat=F).to(device)
    crit = nn.MSELoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\n[학습] LSTM 20 epoch (다변량 · 다중 시계열)")
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d} | train loss = {loss.item():.4f}")

    # 전체 val MAE + persistence baseline 비교
    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xva).to(device)).cpu().numpy().ravel() * sd0 + mu0
    mae = np.abs(pv - Yva).mean()
    base_mae = np.abs(np.array(persist_l) - Yva).mean()
    print(f"\n[검증] LSTM MAE = {mae:.2f}℃  ·  persistence(어제값) baseline = {base_mae:.2f}℃")

    # 대표 시퀀스(가장 긴 농가)의 val 구간을 그림으로
    arr, split = rep
    xs, ts, bs = [], [], []
    for i in range(len(arr) - WINDOW):
        t = i + WINDOW
        if t >= split:
            xs.append((arr[i:t] - mu) / sd); ts.append(arr[t, 0]); bs.append(arr[t - 1, 0])
    with torch.no_grad():
        pr = model(torch.tensor(np.array(xs, dtype=np.float32)).to(device)).cpu().numpy().ravel() * sd0 + mu0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ts, label="실제 온도", lw=2, color="#333")
    ax.plot(pr, label="LSTM 예측(다변량)", lw=2, ls="--", color="#e45756")
    ax.plot(bs, label="baseline(어제값)", lw=1.3, ls=":", color="#999")
    ax.set_title(f"LSTM 다변량·다중 시계열 — 다음날 내부온도 예측\n"
                 f"전체 val MAE {mae:.2f}℃  vs  baseline {base_mae:.2f}℃")
    ax.set_xlabel("대표 농가 검증 구간 일자"); ax.set_ylabel("내부온도 평균(℃)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/08_lstm_forecast.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  그림 저장 → {path}")
    print("  다변량(외부온도·일사량·CO2…) + 201개 시계열 통합 학습 — baseline 대비가 핵심 평가.")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 STEP 2 핵심 — 청크 2-4·2-5·2-6·2-8")
    parser.add_argument("--chunk", choices=["2-4", "2-5", "2-6", "2-8", "all"], default="all")
    args = parser.parse_args()

    runners = {"2-4": run_chunk_2_4, "2-5": run_chunk_2_5,
               "2-6": run_chunk_2_6, "2-8": run_chunk_2_8}
    if args.chunk == "all":
        for run in runners.values():
            run()
        print("\n✅ STEP 2(핵심) 완료 — 다음 STEP 3(평가): 과적합·불균형 대응 + 평가 심화")
    else:
        runners[args.chunk]()
        print(f"\n✅ 청크 {args.chunk} 완료")
