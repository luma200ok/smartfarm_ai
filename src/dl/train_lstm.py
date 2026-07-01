"""
Phase 3-3 — 환경 LSTM 학습·저장 (get_forecast 실연동용).

02_core.py run_chunk_2_8 의 전처리·학습 로직을 이식하고, 추론에 필요한
가중치(env_lstm.pt)와 정규화 통계·피처 메타(env_lstm_meta.json)를 저장한다.
(02_core.py 는 숫자 파일명이라 import 불가 → 로직 복제. figure 대신 아티팩트 저장이 목적.)

실행:  python src/dl/train_lstm.py    (로컬 1회, env_daily.csv 필요)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
ENV_CSV = ROOT / "data" / "processed" / "env_daily.csv"
MODELS = ROOT / "models"
CKPT = MODELS / "env_lstm.pt"
META = MODELS / "env_lstm_meta.json"

ENV_FEATURES = ["온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
                "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균"]
SEQ_KEYS = ["도", "시군", "농가명", "작기", "품목"]
WINDOW = 7
device = "mps" if torch.backends.mps.is_available() else "cpu"


class TempLSTM(nn.Module):
    """(B,7,8) → 다음날 내부온도 1개. 02_core.py 와 동일 구조(서빙 호환)."""
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def main():
    if not ENV_CSV.exists():
        raise SystemExit(f"⚠️ {ENV_CSV} 없음 (Phase1 전처리 산출물 필요)")
    df = pd.read_csv(ENV_CSV, encoding="utf-8-sig")
    # env_daily.csv 는 9개 작물 혼합 → 토마토 온실만 사용(이 앱은 토마토 전용).
    df = df[df["품목"].astype(str).str.contains("토마토", na=False)]
    print(f"토마토 품목: {sorted(df['품목'].unique())} · {len(df)}행")
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
    print(f"시계열 그룹 {n_groups} · 피처 {F} → train {len(Xtr)} / val {len(Xva)}")

    mu = Xtr.reshape(-1, F).mean(0); sd = Xtr.reshape(-1, F).std(0) + 1e-8
    mu0, sd0 = float(mu[0]), float(sd[0])
    Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd
    ytr = (Ytr - mu0) / sd0

    loader = DataLoader(TensorDataset(torch.tensor(Xtr_n), torch.tensor(ytr).unsqueeze(1)),
                        batch_size=256, shuffle=True)
    model = TempLSTM(n_feat=F).to(device)
    crit = nn.MSELoss(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("[학습] LSTM 20 epoch")
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d} | train loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pv = model(torch.tensor(Xva_n).to(device)).cpu().numpy().ravel() * sd0 + mu0
    mae = float(np.abs(pv - Yva).mean())
    base_mae = float(np.abs(np.array(persist_l) - Yva).mean())
    print(f"[검증] LSTM MAE = {mae:.2f}℃ · persistence baseline = {base_mae:.2f}℃")

    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    META.write_text(json.dumps({
        "features": ENV_FEATURES, "window": WINDOW,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "val_mae": round(mae, 3), "baseline_mae": round(base_mae, 3),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 → {CKPT.name} · {META.name}")


if __name__ == "__main__":
    main()
