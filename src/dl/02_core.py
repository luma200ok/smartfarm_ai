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


def _train_head(model, train_loader, val_loader, epochs=3):
    """freeze 된 backbone 위에서 head 만 학습. (val 정확도 반환)"""
    crit = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]   # head 파라미터만
    opt = torch.optim.Adam(params, lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = _eval_acc(model, val_loader)
        print(f"    epoch {epoch+1} | val acc = {acc:.3f}")
    return acc


def _eval_acc(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += (pred == yb).sum().item(); total += len(yb)
    return correct / max(total, 1)


def run_chunk_2_5():
    print("\n" + "═" * 64 + "\n청크 2-5 · 전이학습 (백본 비교) ⭐\n" + "═" * 64)
    if not os.path.isdir(f"{TOMATO}/train"):
        print("⚠️ 토마토 데이터가 없습니다.")
        print("   먼저:  python src/dl/prepare_tomato.py")
        return

    train_loader, val_loader, classes = _tomato_loaders()
    print(f"\n클래스 = {classes}  (ImageFolder 가 폴더명으로 자동 라벨링)")

    results = {}
    for name in ("resnet18", "mobilenet_v2"):
        print(f"\n[백본] {name} — backbone freeze, head 만 학습")
        model = _build_backbone(name, n_classes=len(classes))
        acc = _train_head(model, train_loader, val_loader, epochs=3)
        results[name] = acc
        torch.save(model.state_dict(), f"{MODELS}/tomato_{name}.pt")   # 2-6·2-10 에서 재사용
        print(f"    저장 → {MODELS}/tomato_{name}.pt")

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
    model = _build_backbone("resnet18", n_classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)

    # 마지막 conv 블록에 hook 을 걸어 활성화·기울기를 가로챈다
    target_layer = model.layer4[-1]
    store = {}
    target_layer.register_forward_hook(lambda m, i, o: store.update(act=o.detach()))
    target_layer.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0].detach()))

    # val 첫 이미지 1장으로 시연
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    val_ds = datasets.ImageFolder(f"{TOMATO}/val", tf)
    x, y = val_ds[0]
    xb = x.unsqueeze(0).to(device)

    logits = model(xb)                       # forward (hook 이 act 저장)
    cls = logits.argmax(1).item()
    model.zero_grad()
    logits[0, cls].backward()                # 예측 클래스 점수로 역전파 (hook 이 grad 저장)

    act = store["act"][0]                     # (C,h,w)
    grad = store["grad"][0]                   # (C,h,w)
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
# 청크 2-8 · LSTM — 환경 시계열 (온도 추세 예측)
#   ① 시계열 = '순서가 의미 있는' 데이터 → LSTM 이 과거를 기억해 다음을 예측
#   ② sliding window: 과거 7일 온도 → 다음날 온도 (지도학습으로 변환)
#   ③ 예측 vs 실제 그림으로 추세를 따라가는지 확인
# ════════════════════════════════════════════════════════════════════
class TempLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                 # x: (B, seq, 1)
        out, _ = self.lstm(x)             # out: (B, seq, hidden)
        return self.fc(out[:, -1, :])     # 마지막 타임스텝만 → 다음값 1개


def run_chunk_2_8():
    print("\n" + "═" * 64 + "\n청크 2-8 · LSTM (환경 시계열)\n" + "═" * 64)
    if not os.path.exists(ENV_CSV):
        print(f"⚠️ {ENV_CSV} 없음 (Phase1 전처리 산출물 필요)")
        return
    import pandas as pd

    df = pd.read_csv(ENV_CSV)
    # 한 농가·작기·품목의 가장 긴 연속 시계열을 고른다
    keys = ["도", "시군", "농가명", "작기", "품목"]
    g = df.groupby(keys)
    key = max(g.groups, key=lambda k: len(g.get_group(k)))
    seq_df = g.get_group(key).sort_values("날짜")
    series = seq_df["온도내부_평균"].to_numpy(dtype=np.float32)
    print(f"\n선택 시계열: {dict(zip(keys, key))}  (길이 {len(series)}일)")

    # 정규화 + sliding window (과거 WINDOW 일 → 다음날)
    mu, sd = series.mean(), series.std()
    norm = (series - mu) / sd
    WINDOW = 7
    X, Y = [], []
    for i in range(len(norm) - WINDOW):
        X.append(norm[i:i + WINDOW])
        Y.append(norm[i + WINDOW])
    X = torch.tensor(np.array(X)).unsqueeze(-1)   # (N, 7, 1)
    Y = torch.tensor(np.array(Y)).unsqueeze(-1)   # (N, 1)

    # 앞 80% 학습 / 뒤 20% 검증 (시계열이라 시간순 분할 — 셔플 금지)
    n_train = int(len(X) * 0.8)
    Xtr, Ytr = X[:n_train].to(device), Y[:n_train].to(device)
    Xte, Yte = X[n_train:].to(device), Y[n_train:].to(device)

    model = TempLSTM().to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    print("\n[학습] LSTM 200 epoch")
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xtr), Ytr)
        loss.backward(); opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d} | train loss = {loss.item():.4f}")

    # 검증 구간 예측 (정규화 역연산해 실제 온도 단위로)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).cpu().numpy().ravel() * sd + mu
    true = Yte.cpu().numpy().ravel() * sd + mu

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(true, label="실제 온도", lw=2, color="#333")
    ax.plot(pred, label="LSTM 예측", lw=2, ls="--", color="#e45756")
    ax.set_title("LSTM 환경 시계열 — 다음날 내부온도 예측(검증 구간)")
    ax.set_xlabel("검증 구간 일자"); ax.set_ylabel("내부온도 평균(℃)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/08_lstm_forecast.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    mae = np.abs(pred - true).mean()
    print(f"\n[검증] MAE = {mae:.2f}℃  → 그림 저장 {path}")
    print("  LSTM 은 과거 7일의 흐름을 hidden state 에 기억해 다음날을 추정 — 순서가 핵심.")


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
