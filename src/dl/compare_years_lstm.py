"""
Phase 2 (DL) 보강 — 단년(2022) vs 다년(2022~24) LSTM 내부온도 예측 비교

02_core.py 청크 2-8과 동일 설정: WINDOW=7, hidden=64, 20 epoch, lr=1e-3, batch=256.
핵심 지표 = 'persistence(어제값) baseline 대비 개선폭'
  → 평가셋이 달라도 데이터셋 내부 상대지표라 단년/다년 공정 비교 가능.
입력: data/processed/env_daily_2022.csv (단년) · env_daily.csv (다년)
출력: docs/figures/phase2_dl/08b_lstm_year_compare.png · _lstm_year_compare.txt
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figures" / "phase2_dl"
SINGLE = ROOT / "data" / "processed" / "env_daily_2022.csv"
MULTI = ROOT / "data" / "processed" / "env_daily.csv"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 02_core.py 청크 2-8과 동일
ENV_FEATURES = ["온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
                "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균"]
SEQ_KEYS = ["도", "시군", "농가명", "작기", "품목"]
WINDOW = 7

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


class TempLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def run(path):
    """시계열 구성 → LSTM 20 epoch 학습 → val MAE & baseline MAE. (02_core 청크 2-8 동일 로직)"""
    torch.manual_seed(42)
    np.random.seed(42)
    df = pd.read_csv(path)
    F = len(ENV_FEATURES)
    Xtr_l, Ytr_l, Xva_l, Yva_l, persist_l = [], [], [], [], []
    n_groups = 0
    for _, gdf in df.groupby(SEQ_KEYS):
        arr = gdf.sort_values("날짜")[ENV_FEATURES].to_numpy(np.float32)
        if len(arr) < WINDOW + 5:
            continue
        n_groups += 1
        split = int(len(arr) * 0.8)
        for i in range(len(arr) - WINDOW):
            t = i + WINDOW
            x, y = arr[i:t], arr[t, 0]
            if t < split:
                Xtr_l.append(x); Ytr_l.append(y)
            else:
                Xva_l.append(x); Yva_l.append(y); persist_l.append(arr[t - 1, 0])

    Xtr, Xva = np.array(Xtr_l), np.array(Xva_l)
    Ytr, Yva = np.array(Ytr_l), np.array(Yva_l)
    mu = Xtr.reshape(-1, F).mean(0); sd = Xtr.reshape(-1, F).std(0) + 1e-8
    mu0, sd0 = mu[0], sd[0]
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ytr = (Ytr - mu0) / sd0

    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr).unsqueeze(1)),
                        batch_size=256, shuffle=True)
    model = TempLSTM(F).to(device)
    crit = nn.MSELoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xva).to(device)).cpu().numpy().ravel() * sd0 + mu0
    mae = float(np.abs(pv - Yva).mean())
    base = float(np.abs(np.array(persist_l) - Yva).mean())
    return {"groups": n_groups, "train": len(Xtr), "val": len(Xva), "mae": mae, "base": base}


def main():
    print(f"device = {device}")
    print("[1] 단년(2022) LSTM 학습...")
    s = run(SINGLE)
    print(f"  시계열 {s['groups']}개 · LSTM MAE={s['mae']:.2f}℃ · baseline={s['base']:.2f}℃")
    print("[2] 다년(2022~24) LSTM 학습...")
    m = run(MULTI)
    print(f"  시계열 {m['groups']}개 · LSTM MAE={m['mae']:.2f}℃ · baseline={m['base']:.2f}℃")

    print(f"\n{'':14}{'시계열':>6}{'train':>8}{'val':>8}{'LSTM':>8}{'baseline':>10}{'개선':>8}")
    rows = [("단년 2022", s), ("다년 2022~24", m)]
    for lab, r in rows:
        print(f"{lab:14}{r['groups']:>6}{r['train']:>8}{r['val']:>8}"
              f"{r['mae']:>7.2f}℃{r['base']:>9.2f}℃{r['base'] - r['mae']:>+7.2f}")

    # 그림: 단년/다년 × (LSTM, baseline) MAE
    x = np.arange(2)
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.2, [s["mae"], m["mae"]], 0.4, label="LSTM(다변량)", color="#e45756")
    plt.bar(x + 0.2, [s["base"], m["base"]], 0.4, label="baseline(어제값)", color="#999")
    for i, r in enumerate([s, m]):
        plt.text(i - 0.2, r["mae"] + 0.02, f"{r['mae']:.2f}", ha="center", fontsize=9)
        plt.text(i + 0.2, r["base"] + 0.02, f"{r['base']:.2f}", ha="center", fontsize=9)
    plt.xticks(x, [f"단년 2022\n({s['groups']}시계열)", f"다년 2022~24\n({m['groups']}시계열)"])
    plt.ylabel("내부온도 예측 MAE (℃, 낮을수록 좋음)")
    plt.title("데이터 양 효과 — LSTM 온도예측: 단년 vs 다년")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "08b_lstm_year_compare.png", dpi=120)
    plt.close()
    print(f"\n그림 저장 → {FIGS}/08b_lstm_year_compare.png")

    with open(FIGS / "_lstm_year_compare.txt", "w") as f:
        for lab, r in rows:
            f.write(f"{lab}: 시계열={r['groups']}, train={r['train']}, val={r['val']}, "
                    f"LSTM MAE={r['mae']:.2f}, baseline={r['base']:.2f}, "
                    f"개선={r['base'] - r['mae']:+.2f}\n")
    print("요약 → _lstm_year_compare.txt")


if __name__ == "__main__":
    main()
