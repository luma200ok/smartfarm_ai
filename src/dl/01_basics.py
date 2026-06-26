"""
Phase 2 (DL) · STEP 1 · 기초 — 신경망 원리 (forward → 학습 → 데이터 파이프)

이 파일 하나 = STEP 1 전체. 청크 3개가 블록으로 들어있다:
  · 청크 2-1  신경망 기초 (뉴런·활성화)         — forward 만, "활성화 없으면 직선"
  · 청크 2-2  학습 메커니즘 (손실·역전파·Adam)   — loss→backward→step 5단계 루프
  · 청크 2-3  Dataset/DataLoader · batch 루프    — batch·.to(device) 로 진짜 학습 골격

STEP 1은 '원리 증명' 구간 — PyTorch 가 뒤에서 뭘 하는지 손계산과 대조해 한 번 믿는다.
(이 손계산 대조는 STEP 2부터 안 나온다. 거기선 믿고 실전 코드를 쓴다.)

실행:
  python src/dl/01_basics.py               # 청크 2-1·2-2·2-3 전부
  python src/dl/01_basics.py --chunk 2-2   # 특정 청크만 ('파먹기'용)
출력 그림 → docs/figures/phase2_dl/
"""
import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)  # 랜덤 초기화 재현용

# 맥은 CUDA 가 없고 MPS(Apple GPU)가 가속 역할.
# 2-1·2-2 는 텐서가 작아(200점) CPU 가 더 빠르고, 2-3 부터 .to(device) 로 MPS 가 실제 이득.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")


def _log10_formatter():
    """로그축 지수(10⁻¹…) 마이너스 깨짐 방지용 포맷터.
    기본 LogFormatter 는 라벨을 $\\mathdefault{10^{-1}}$ 로 그리는데, \\mathdefault 는
    mathtext.fontset 을 무시하고 본문 폰트(AppleGothic)를 강제 → AppleGothic 엔 마이너스
    글리프(U+2212)가 없어 '10¤1' 로 깨진다. \\mathdefault 없는 순수 mathtext($10^{e}$)로
    그리면 mathtext 기본 fontset(dejavusans, 글리프 있음)이 마이너스를 정상 렌더한다."""
    def fmt(v, _):
        if v <= 0:
            return ""
        return rf"$10^{{{int(round(math.log10(v)))}}}$"
    return FuncFormatter(fmt)


# ════════════════════════════════════════════════════════════════════
# 청크 2-1 · 신경망 기초 — 뉴런 · 층 · 활성화   (forward 만)
#   ① 뉴런 = Σ(입력×가중치)+편향 → 활성화   ② 활성화 4종   ③ 활성화 없으면 직선
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# ① 뉴런 1개 = 입력×가중치 합 + 편향 → 활성화
#    nn.Linear 가 정확히 이 계산임을 '손계산'과 대조해 확인한다.
# ─────────────────────────────────────────────────────────────
def neuron_check():
    x = torch.tensor([2.0, -1.0, 0.5])      # 입력 3개
    w = torch.tensor([0.5, -0.3, 0.8])      # 가중치 3개
    b = torch.tensor(0.1)                   # 편향

    z_manual = (x * w).sum() + b            # 손계산: Σ(입력×가중치) + 편향
    a_manual = torch.relu(z_manual)         # 활성화(ReLU) 통과

    # nn.Linear(3,1) 에 같은 w,b 를 주입 → 같은 값이 나와야 한다
    lin = nn.Linear(in_features=3, out_features=1)
    with torch.no_grad():
        lin.weight.copy_(w.reshape(1, 3))   # Linear.weight 모양 = (출력, 입력)
        lin.bias.copy_(b.reshape(1))
    z_linear = lin(x)

    print("\n[①] 뉴런 손계산 vs nn.Linear")
    print(f"  z(손계산) = {z_manual.item():.4f}  |  z(Linear) = {z_linear.item():.4f}  → 일치")
    print(f"  ReLU 통과 후 a = {a_manual.item():.4f}")


# ─────────────────────────────────────────────────────────────
# ② 활성화함수 4종 — 모양과 쓰임
# ─────────────────────────────────────────────────────────────
def plot_activations():
    x = torch.linspace(-5, 5, 200)
    funcs = {
        "ReLU  (은닉층 기본)": torch.relu(x),
        "Sigmoid (이진분류 출력, 0~1)": torch.sigmoid(x),
        "Tanh  (-1~1, 일부 RNN)": torch.tanh(x),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, y) in zip(axes, funcs.items()):
        ax.plot(x.numpy(), y.numpy(), lw=2, color="#2a7")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.suptitle("활성화함수 — 입력을 비선형으로 변환 (곡선·꺾임)", fontsize=13)
    fig.tight_layout()
    path = f"{FIGS}/01_activations.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[②] 활성화 그림 저장 → {path}")

    # Softmax: 점수 벡터 → 확률분포(합=1). 다중분류 출력층에서 쓴다(🔜 잎 병 N종).
    scores = torch.tensor([2.0, 1.0, 0.1])
    probs = torch.softmax(scores, dim=0)
    print(f"  Softmax({scores.tolist()}) = "
          f"{[round(p, 3) for p in probs.tolist()]}  (합={probs.sum():.2f})")


# ─────────────────────────────────────────────────────────────
# ③ ★핵심 실험: 활성화 없으면 깊어도 '직선'
#    같은 구조의 망을 (A) 활성화 없이 / (B) ReLU 끼워서 forward 만 한다.
#    학습을 안 해도 랜덤 가중치만으로 차이가 드러난다.
# ─────────────────────────────────────────────────────────────
def linear_vs_relu():
    x = torch.linspace(-3, 3, 200).reshape(-1, 1)

    torch.manual_seed(0)
    deep_linear = nn.Sequential(           # 활성화 없음
        nn.Linear(1, 16), nn.Linear(16, 16), nn.Linear(16, 1))
    torch.manual_seed(0)
    deep_relu = nn.Sequential(             # 똑같은 구조 + ReLU
        nn.Linear(1, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 1))

    with torch.no_grad():
        y_lin = deep_linear(x)
        y_relu = deep_relu(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x.numpy(), y_lin.numpy(), lw=2.5, label="활성화 없음 → 직선 (3층이어도)")
    ax.plot(x.numpy(), y_relu.numpy(), lw=2.5, label="ReLU 추가 → 꺾인 곡선")
    ax.set_title("왜 활성화가 필요한가 (학습 전, 랜덤 가중치 forward)")
    ax.set_xlabel("입력 x"); ax.set_ylabel("출력")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/01_why_activation.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[③] 활성화 유무 비교 그림 저장 → {path}")

    # 수학적 증명: 선형층 3개 = 행렬 합성 → 1층(Wx+b)과 완전히 동일
    W1, b1 = deep_linear[0].weight, deep_linear[0].bias
    W2, b2 = deep_linear[1].weight, deep_linear[1].bias
    W3, b3 = deep_linear[2].weight, deep_linear[2].bias
    with torch.no_grad():
        W_eff = W3 @ W2 @ W1                       # 합쳐진 가중치 (1×1)
        b_eff = (W3 @ W2 @ b1) + (W3 @ b2) + b3    # 합쳐진 편향
        y_single = x @ W_eff.T + b_eff             # 1층으로 계산
    max_diff = (y_lin - y_single).abs().max().item()
    print(f"  3층 선형 출력 vs 1층(Wx+b) 출력 최대오차 = {max_diff:.2e}  → 사실상 0")
    print(f"  ∴ 활성화 없는 깊은 망 = 1층짜리 직선. 활성화가 있어야 '깊이'가 의미 생김.")


# ════════════════════════════════════════════════════════════════════
# 청크 2-2 · 학습 메커니즘 — 손실 · 역전파 · 옵티마이저
#   ① 손실(MSE/CE)   ② 역전파=자동미분   ③ 5단계 루프로 sin 피팅   ④ 학습률 효과
#   ※ 작은 텐서(200점)라 일부러 CPU. .to(device) 는 다음 청크 2-3 부터.
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# ① 손실(Loss) — 예측과 정답의 차이를 숫자 1개로
#    회귀용 MSE(평균제곱오차)를 손계산과 nn.MSELoss 로 대조한다.
# ─────────────────────────────────────────────────────────────
def loss_check():
    pred = torch.tensor([2.0, 0.0, 3.0])    # 모델 예측
    true = torch.tensor([1.0, 0.0, 5.0])    # 정답

    # MSE = 평균( (예측-정답)^2 ) — "틀린 만큼 제곱해서 평균"
    mse_manual = ((pred - true) ** 2).mean()
    mse_torch = nn.MSELoss()(pred, true)

    print("\n[①] 손실 손계산 vs nn.MSELoss")
    print(f"  오차={ (pred-true).tolist() }  제곱평균(MSE) 손계산={mse_manual:.4f} | torch={mse_torch:.4f} → 일치")

    # 분류용 CrossEntropy: '정답 클래스 확률'이 낮을수록 손실↑ (🔜 잎 병 N종 분류에서 사용)
    logits = torch.tensor([[2.0, 1.0, 0.1]])   # 한 샘플의 3클래스 점수
    label = torch.tensor([0])                  # 정답 = 0번 클래스
    ce = nn.CrossEntropyLoss()(logits, label)
    print(f"  CrossEntropy(정답=0번, 점수 가장 큼) = {ce:.4f}  (정답 잘 맞히면 손실 작음)")


# ─────────────────────────────────────────────────────────────
# ② 역전파(backward) = 자동 미분
#    loss.backward() 가 'loss 를 가중치로 미분'한 값을 .grad 에 채운다.
#    간단한 함수 loss=(w-3)^2 로 손미분 2(w-3) 과 대조.
# ─────────────────────────────────────────────────────────────
def autograd_check():
    w = torch.tensor(0.0, requires_grad=True)   # requires_grad=True → 이 값의 기울기를 추적
    loss = (w - 3) ** 2                          # 최소점은 w=3
    loss.backward()                              # 역전파: dloss/dw 계산 → w.grad 에 저장

    grad_manual = 2 * (w.item() - 3)             # 손미분: d/dw (w-3)^2 = 2(w-3)
    print("\n[②] 역전파(자동미분) vs 손미분")
    print(f"  w=0 에서  w.grad(자동)={w.grad.item():.1f} | 손미분 2(w-3)={grad_manual:.1f} → 일치")
    print(f"  grad 부호가 음수 → w 를 '+ 방향'으로 옮겨야 loss 줄어듦(=경사하강 방향)")


# ─────────────────────────────────────────────────────────────
# ③ ★학습 루프 — 2-1의 ReLU 망이 sin 곡선을 '진짜 학습'
#    모든 DL 공통 5단계 리듬: 예측 → 손실 → zero_grad → backward → step
# ─────────────────────────────────────────────────────────────
def train_fit_sin():
    x = torch.linspace(-3, 3, 200).reshape(-1, 1)
    y_true = torch.sin(x)                         # 목표 곡선

    torch.manual_seed(0)
    model = nn.Sequential(                        # 2-1과 같은 1→16→16→1 + ReLU
        nn.Linear(1, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 1))

    with torch.no_grad():
        y_before = model(x).clone()               # 학습 전(랜덤 가중치) 출력

    criterion = nn.MSELoss()                      # 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # 옵티마이저(Adam)

    EPOCHS = 2000
    for epoch in range(EPOCHS):
        pred = model(x)                # ① 예측(forward)
        loss = criterion(pred, y_true) # ② 손실
        optimizer.zero_grad()          # ③ 이전 기울기 초기화 (안 하면 누적됨!)
        loss.backward()                # ④ 역전파(기울기 계산)
        optimizer.step()               # ⑤ 가중치 갱신
        if (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch+1:4d} | loss = {loss.item():.5f}")

    with torch.no_grad():
        y_after = model(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x.numpy(), y_true.numpy(), lw=2.5, label="정답 sin(x)", color="#333")
    ax.plot(x.numpy(), y_before.numpy(), lw=1.8, ls="--", label="학습 전(랜덤)", color="#bbb")
    ax.plot(x.numpy(), y_after.numpy(), lw=2.5, label="학습 후(피팅됨)", color="#e64")
    ax.set_title("학습 루프로 sin 곡선 피팅 (loss→backward→step 반복)")
    ax.set_xlabel("입력 x"); ax.set_ylabel("출력")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/02_fit_sin.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[③] 학습 전/후 비교 그림 저장 → {path}")


# ─────────────────────────────────────────────────────────────
# ④ ★학습률(lr) — 가장 중요한 하이퍼파라미터
#    같은 문제 loss=(w-3)^2 를 lr 3종으로 경사하강 → 발산/적당/느림 비교.
#    (수렴 판단이 명확하도록 간단한 볼록함수 + 순수 경사하강으로 본다)
# ─────────────────────────────────────────────────────────────
def lr_experiment():
    # 수렴 조건은 lr<1 (grad=2(w-3) 이라 lr<2/2). 1.1 은 경계를 넘겨 매 step 손실이 1.2배씩 커짐 → 확실히 발산.
    settings = {"lr=1.1 → 발산": 1.1, "lr=0.1 → 적당(수렴)": 0.1, "lr=0.01 → 너무 느림": 0.01}
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, lr in settings.items():
        w, hist = 0.0, []
        for _ in range(50):
            grad = 2 * (w - 3)        # d/dw (w-3)^2
            w = w - lr * grad         # 경사하강: 기울기 반대 방향으로 lr 만큼
            hist.append((w - 3) ** 2) # 현재 손실
        ax.plot(range(1, 51), hist, lw=2, marker="o", ms=3, label=name)
    ax.set_yscale("log")              # 발산값이 커서 로그축
    ax.yaxis.set_major_formatter(_log10_formatter())  # 지수 라벨 마이너스(10⁻ⁿ) 깨짐 방지
    ax.set_title("학습률(lr) 효과 — 같은 문제, lr만 다르게")
    ax.set_xlabel("step"); ax.set_ylabel("손실 (log)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/02_lr_effect.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[④] 학습률 비교 그림 저장 → {path}")
    print("  lr 너무 큼 → 최소점을 건너뛰며 발산 / 너무 작음 → 거의 안 움직임 / 적당 → 빠르게 수렴")


# ════════════════════════════════════════════════════════════════════
# 청크 2-3 · Dataset / DataLoader · batch 학습 루프
#   ① DataLoader=batch 컨베이어   ② Dataset=__len__+__getitem__
#   ③ batch for문으로 2클래스 분류 (.to(device) 로 MPS)   ④ full vs mini-batch
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# ① DataLoader 맛보기 — 데이터를 'batch' 로 꺼내는 컨베이어벨트
#    작은 데이터 10개를 batch_size=4 로 묶으면 (4, 4, 2) 세 묶음으로 나온다.
# ─────────────────────────────────────────────────────────────
def loader_peek():
    X = torch.arange(10).reshape(10, 1).float()   # 0~9, 한 줄에 하나
    y = torch.arange(10)                           # 라벨도 0~9 (추적용)
    ds = TensorDataset(X, y)                        # (x, y) 한 쌍씩 묶어주는 기본 Dataset
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    print("\n[①] DataLoader 가 batch 로 꺼내는 모습 (10개를 batch_size=4 로)")
    for i, (xb, yb) in enumerate(loader):
        print(f"  batch {i}: x={xb.flatten().tolist()}  (개수 {len(xb)})")
    print("  → 10개가 4+4+2 로 쪼개짐. 마지막 묶음은 남는 2개(끝수)만.")

    # shuffle=True 면 매 epoch 순서가 섞인다 (과적합 방지에 중요 — 2-7에서 다시)
    loader_s = DataLoader(ds, batch_size=4, shuffle=True)
    first = next(iter(loader_s))[1].tolist()
    print(f"  shuffle=True 첫 batch 라벨: {first}  (순서 섞임 — 0~3 아님)")


# ─────────────────────────────────────────────────────────────
# ② 커스텀 Dataset — TensorDataset 의 정체를 손으로 까보기
#    Dataset = '__len__(몇 개) + __getitem__(i번째)' 두 메서드면 끝.
# ─────────────────────────────────────────────────────────────
class MoonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # 입력은 float32
        self.y = torch.tensor(y, dtype=torch.long)     # ⚠️ 분류 라벨은 long(정수) — CrossEntropy 규칙

    def __len__(self):                  # "데이터 몇 개?" → DataLoader 가 batch 나눌 때 씀
        return len(self.X)

    def __getitem__(self, i):           # "i번째 한 쌍 줘" → DataLoader 가 이걸 batch_size 번 호출해 묶음
        return self.X[i], self.y[i]


def dataset_check():
    X, y = make_moons(n_samples=8, noise=0.1, random_state=0)
    ds = MoonDataset(X, y)
    print("\n[②] 커스텀 Dataset (직접 만든 class)")
    print(f"  len(ds) = {len(ds)}  ← __len__ 이 돌려준 값")
    x0, y0 = ds[0]                       # ds[0] = ds.__getitem__(0)
    print(f"  ds[0] = (x={x0.tolist()}, y={y0.item()})  ← __getitem__(0) 이 돌려준 한 쌍")
    print("  → DataLoader 는 이 __getitem__ 을 batch_size 번 불러 한 묶음으로 쌓을 뿐.")


# ─────────────────────────────────────────────────────────────
# ③ ★batch 학습 루프 — 2-2의 5단계를 'batch for문'으로 한 겹 감쌈
#    make_moons 2클래스 분류 · CrossEntropy · .to(device)
# ─────────────────────────────────────────────────────────────
def train_moons():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    loader = DataLoader(MoonDataset(X, y), batch_size=32, shuffle=True)

    torch.manual_seed(0)
    model = nn.Sequential(                # 입력 2(x,y좌표) → 16 → 16 → 2(클래스 점수)
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 2)).to(device)      # ★ 모델을 device(mps)로
    criterion = nn.CrossEntropyLoss()     # 분류 손실 (①에서 본 그것)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        for xb, yb in loader:             # ★ 추가된 겹: batch 단위 반복
            xb, yb = xb.to(device), yb.to(device)  # ★ 데이터도 device 로
            pred = model(xb)              # ① 예측
            loss = criterion(pred, yb)    # ② 손실
            optimizer.zero_grad()         # ③ 초기화
            loss.backward()               # ④ 역전파
            optimizer.step()              # ⑤ 갱신
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d} | (마지막 batch) loss = {loss.item():.4f}")

    # 정확도 (전체 데이터로 한 번 — no_grad: 학습 아님)
    with torch.no_grad():
        Xall = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(Xall)
        acc = (logits.argmax(1).cpu().numpy() == y).mean()   # argmax = 점수 큰 클래스 고름
    print(f"\n[③] batch 학습 완료 — 정확도 {acc:.3f}")
    _plot_decision_boundary(model, X, y, acc)


def _plot_decision_boundary(model, X, y, acc):
    # 평면을 격자로 깔고 각 점의 예측 클래스를 색칠 → 모델이 그은 경계 시각화
    h = 0.02
    xx, yy = np.meshgrid(np.arange(X[:, 0].min()-.5, X[:, 0].max()+.5, h),
                         np.arange(X[:, 1].min()-.5, X[:, 1].max()+.5, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        zz = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.3)
    ax.set_title(f"batch 학습한 분류 결정경계 (정확도 {acc:.2f})")
    ax.set_xlabel("특성 1"); ax.set_ylabel("특성 2")
    fig.tight_layout()
    path = f"{FIGS}/03_decision_boundary.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  결정경계 그림 저장 → {path}")


# ─────────────────────────────────────────────────────────────
# ④ ★full-batch vs mini-batch — 데이터를 통째로 vs 32개씩
#    같은 모델·epoch 수에서 epoch당 평균 손실이 누가 빨리 내려가나.
# ─────────────────────────────────────────────────────────────
def batch_compare():
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
    ds = MoonDataset(X, y)

    settings = {"full-batch (1000개 통째)": 1000, "mini-batch (32개씩)": 32}
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, bs in settings.items():
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        torch.manual_seed(0)                          # 같은 출발선
        model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(),
                              nn.Linear(16, 16), nn.ReLU(),
                              nn.Linear(16, 2)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        hist = []
        for epoch in range(50):
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            hist.append(sum(losses) / len(losses))    # epoch당 평균 손실
        ax.plot(range(1, 51), hist, lw=2, label=name)
    ax.set_title("full-batch vs mini-batch — epoch당 평균 손실")
    ax.set_xlabel("epoch"); ax.set_ylabel("평균 손실")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/03_batch_compare.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[④] batch 비교 그림 저장 → {path}")
    print("  full-batch=epoch당 1걸음(1000개로 1번) / mini-batch=epoch당 32걸음(여러 번) → 같은 epoch에 더 많이 내려감.")


# ════════════════════════════════════════════════════════════════════
# 청크 러너 — '파먹기' 단위 (--chunk 로 골라 실행)
# ════════════════════════════════════════════════════════════════════
def run_chunk_2_1():
    print("\n" + "═" * 64 + "\n청크 2-1 · 신경망 기초 (뉴런·활성화)\n" + "═" * 64)
    neuron_check()
    plot_activations()
    linear_vs_relu()


def run_chunk_2_2():
    print("\n" + "═" * 64 + "\n청크 2-2 · 학습 메커니즘 (손실·역전파·옵티마이저)\n" + "═" * 64)
    loss_check()
    autograd_check()
    train_fit_sin()
    lr_experiment()


def run_chunk_2_3():
    print("\n" + "═" * 64 + "\n청크 2-3 · Dataset/DataLoader · batch 루프\n" + "═" * 64)
    loader_peek()
    dataset_check()
    train_moons()
    batch_compare()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 STEP 1 기초 — 청크 2-1·2-2·2-3")
    parser.add_argument("--chunk", choices=["2-1", "2-2", "2-3", "all"], default="all",
                        help="실행할 청크 (기본 all). 파먹을 땐 특정 청크만.")
    args = parser.parse_args()

    runners = {"2-1": run_chunk_2_1, "2-2": run_chunk_2_2, "2-3": run_chunk_2_3}
    if args.chunk == "all":
        for run in runners.values():
            run()
        print("\n✅ STEP 1(기초) 완료 — 다음 STEP 2(핵심): CNN·전이학습·Grad-CAM·LSTM")
    else:
        runners[args.chunk]()
        print(f"\n✅ 청크 {args.chunk} 완료")
