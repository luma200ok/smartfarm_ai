"""
Phase 2 (DL) · 청크 2-2 — 학습 메커니즘: 손실 · 역전파 · 옵티마이저

학습 목표 (코드로 직접 확인):
  ① 손실(Loss) = "예측이 정답에서 얼마나 틀렸나"를 숫자 1개로  (MSE 손계산 vs nn.MSELoss 대조)
  ② 역전파(backward) = loss 를 가중치로 '자동 미분' → `.grad` 가 손미분과 일치
  ③ ★핵심: 5단계 학습 루프로 2-1의 ReLU 망이 sin 곡선을 '진짜 학습'
     (학습 전 랜덤 → 학습 후 곡선에 달라붙는 것 그림으로)
  ④ ★학습률(lr): 너무 크면 발산 / 너무 작으면 느림 / 적당하면 수렴  (직접 비교)

※ 2-1은 forward(순전파)만 봤다. 여기서 비로소 '학습'(loss→backward→step)을 붙인다.
   이게 ML(`model.fit()` 한 줄)과 DL의 가장 큰 차이 — 루프를 직접 쓴다.
출력 그림 → docs/figures/phase2_dl/
"""
import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def log10_formatter():
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

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")
# ※ 이 청크는 텐서가 작아(200포인트) GPU(mps) 전송 오버헤드가 더 크다 → 일부러 CPU로 돈다.
#    device 를 실제로 .to(device) 로 붙이는 건 본격 학습인 2-3(DataLoader)부터.


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
    ax.yaxis.set_major_formatter(log10_formatter())  # 지수 라벨 마이너스(10⁻ⁿ) 깨짐 방지
    ax.set_title("학습률(lr) 효과 — 같은 문제, lr만 다르게")
    ax.set_xlabel("step"); ax.set_ylabel("손실 (log)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"{FIGS}/02_lr_effect.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[④] 학습률 비교 그림 저장 → {path}")
    print("  lr 너무 큼 → 최소점을 건너뛰며 발산 / 너무 작음 → 거의 안 움직임 / 적당 → 빠르게 수렴")


if __name__ == "__main__":
    loss_check()
    autograd_check()
    train_fit_sin()
    lr_experiment()
    print("\n청크 2-2 완료 — 손실·역전파·옵티마이저·학습률 확인. 다음 → 2-3 Dataset/DataLoader.")
