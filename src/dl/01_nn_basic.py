"""
Phase 2 (DL) · 청크 2-1 — 신경망 기초: 뉴런 · 층 · 활성화

학습 목표 (코드로 직접 확인):
  ① 뉴런 = 입력×가중치 합 + 편향 → 활성화함수   (nn.Linear 가 이 계산임을 손계산과 대조)
  ② 활성화함수 4종(ReLU/Sigmoid/Tanh/Softmax)의 모양·쓰임
  ③ ★핵심: "활성화 없으면 층을 아무리 깊게 쌓아도 결국 직선"
     → 학습(2-2) 없이 forward + 행렬 합성만으로 증명

※ 학습 루프(손실·역전파·Adam)는 다음 청크 2-2. 여기선 순전파(forward)만 본다.
출력 그림 → docs/figures/phase2_dl/
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(42)  # 랜덤 초기화 재현용

# 맥은 CUDA가 없고 MPS(Apple GPU)가 가속 역할. (이번 청크는 가벼워 CPU여도 동일)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")


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


if __name__ == "__main__":
    neuron_check()
    plot_activations()
    linear_vs_relu()
    print("\n청크 2-1 완료 — 뉴런·활성화·'활성화 없으면 직선' 확인. 다음 → 2-2 학습 메커니즘.")
