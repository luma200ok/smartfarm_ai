"""
Phase 3 (LLM) 재사용 추론 계층 — streamlit 비의존 순수 함수.

app/phase2_dl.py 의 진단·게이트·검출 로직을 src 계층으로 이식.
Phase 2 는 그대로 streamlit(@st.cache_resource)에 묶여 있어 src/llm 에서 못 쓴다 →
LLM tool(src/llm/tools.py)이 import 할 수 있도록 동일 로직을 여기 모은다.
모델 로드는 @lru_cache 로 프로세스 1회만(streamlit 캐시 대체).

모델: models/tomato_resnet18.pt(진단) · tomato_part.pt(부위 게이트) · tomato_yolov8n.pt(검출)
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]          # 배포 안전(절대경로 하드코딩 지양)
MODELS = ROOT / "models"
CKPT = MODELS / "tomato_resnet18.pt"
PART_CKPT = MODELS / "tomato_part.pt"
YOLO_CKPT = MODELS / "tomato_yolov8n.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 진단 3클래스(ImageFolder 알파벳순, 학습과 동일)
CLASSES = ["leaf_mold", "normal", "tylcv"]
LABEL_KR = {"leaf_mold": "잎곰팡이병", "normal": "정상", "tylcv": "황화잎말이바이러스"}
# 부위 게이트 4클래스
PART_CLASSES = ["flower", "fruit", "leaf", "stem"]
PART_KR = {"flower": "꽃", "fruit": "과실", "leaf": "잎", "stem": "줄기"}

# OOD 게이트: 닫힌 분류기는 잎 아닌 이미지도 한 클래스로 찍음 → ImageNet 1000클래스로 "식물인가" 판별.
PLANT_THRESHOLD = 0.04                               # 식물 클래스 확률 합 < 이 값이면 "잎 아님"
PLANT_KEYWORDS = ["leaf", "cardoon", "nettle", "cabbage", "broccoli", "cauliflower", "cucumber",
                  "artichoke", "zucchini", "corn", "pot ", "plant", "acorn", "fig", "pineapple",
                  "buckeye", "ear", "hay", "daisy", "mushroom", "bell pepper", "granny smith",
                  "custard apple", "hip", "head cabbage", "spaghetti squash", "butternut"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _tf():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# ── 진단 (resnet18 전이학습) ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_diagnosis_model():
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    return m.eval().to(device)


def diagnose(pil):
    """잎 사진 1장 → {label, prob, probs{클래스:확률}}. Grad-CAM 없음(처방 텍스트엔 불필요)."""
    model = load_diagnosis_model()
    with torch.no_grad():
        x = _tf()(pil.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x)[0], 0)
    cls = int(probs.argmax())
    return {
        "label": CLASSES[cls],
        "prob": float(probs[cls]),
        "probs": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
    }


# ── 부위 게이트 (과실/꽃/잎/줄기) ─────────────────────────────────────────
@lru_cache(maxsize=1)
def load_part_model():
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(PART_CLASSES))
    m.load_state_dict(torch.load(PART_CKPT, map_location=device, weights_only=True))
    return m.eval().to(device)


def part_of(pil):
    """부위 추론 → (부위코드, 확률). leaf 아니면 잎 진단 차단.

    부위 모델(PART_CKPT) 없으면 게이트 스킵('leaf', 0.0) — 파일 미배포 시 죽지 않게.
    """
    if not PART_CKPT.exists():
        _log.warning("부위 게이트 모델 없음(%s) — 잎 게이트를 건너뜀. 비잎 사진이 진단될 수 있으니 "
                     "재배포 시 tomato_part.pt 를 반드시 배치할 것.", PART_CKPT)
        return "leaf", 0.0
    model = load_part_model()
    with torch.no_grad():
        x = _tf()(pil.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x)[0], 0)
    idx = int(probs.argmax())
    return PART_CLASSES[idx], float(probs[idx])


# ── OOD 게이트 (ImageNet 사전학습으로 "식물인가") ──────────────────────────
@lru_cache(maxsize=1)
def load_leaf_gate():
    from torchvision import models
    w = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=w).eval().to(device)
    cats = w.meta["categories"]
    idx = sorted({i for i, c in enumerate(cats)
                  if any(k in c.lower() for k in PLANT_KEYWORDS)})
    return net, w.transforms(), idx


def ood_plant_score(pil):
    """입력이 식물·잎일 정도(ImageNet 식물 클래스 softmax 합, 0~1). 낮을수록 OOD(잎 아님)."""
    net, tf, idx = load_leaf_gate()
    with torch.no_grad():
        x = tf(pil.convert("RGB")).unsqueeze(0).to(device)
        return float(torch.softmax(net(x)[0], 0)[idx].sum())


# ── 병변 위치 검출 (YOLOv8) ───────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_CKPT))


def _region(y_center, height):
    """박스 y중심 → 잎 위치(상/중/하). 처방에 '아래쪽 잎 집중' 같은 근거로 쓰임."""
    r = y_center / max(height, 1)
    return "상단" if r < 1 / 3 else ("중단" if r < 2 / 3 else "하단")


def detect(pil, conf=0.25):
    """YOLO 검출 → [{cls, conf, region}]. region 은 박스 y중심으로 파생한 잎 위치."""
    yolo = load_yolo()
    res = yolo.predict(pil.convert("RGB"), device=device, conf=conf, verbose=False)[0]
    height = res.orig_shape[0]
    names = res.names
    out = []
    for b in res.boxes:
        y1, y2 = float(b.xyxy[0][1]), float(b.xyxy[0][3])
        out.append({
            "cls": names[int(b.cls)],
            "conf": float(b.conf),
            "region": _region((y1 + y2) / 2, height),
        })
    return out


# ── 환경 예측 (LSTM 다변량 시계열) ────────────────────────────────────────
LSTM_CKPT = MODELS / "env_lstm.pt"
LSTM_META = MODELS / "env_lstm_meta.json"
ENV_CSV = ROOT / "data" / "processed" / "env_daily.csv"
ENV_FEATURES = ["온도내부_평균", "온도내부_최저", "온도내부_최고", "온도내부_표준편차",
                "습도내부_평균", "co2_평균", "온도외부_평균", "일사량_평균"]
SEQ_KEYS = ["도", "시군", "농가명", "작기", "품목"]
WINDOW = 7
_HUM_IDX = ENV_FEATURES.index("습도내부_평균")   # 피처 4


class TempLSTM(nn.Module):
    """(B,7,8) → 다음날 내부온도 1개. train_lstm.py·02_core.py 와 동일 구조."""
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


@lru_cache(maxsize=1)
def load_forecast_model():
    """(model, meta) 또는 None(가중치·메타 없으면 graceful — train_lstm.py 미실행 시)."""
    if not (LSTM_CKPT.exists() and LSTM_META.exists()):
        _log.warning("LSTM 가중치 없음(%s) — forecast 비활성. train_lstm.py 실행 필요.", LSTM_CKPT)
        return None
    meta = json.loads(LSTM_META.read_text(encoding="utf-8"))
    m = TempLSTM(n_feat=len(meta["features"]))
    m.load_state_dict(torch.load(LSTM_CKPT, map_location=device, weights_only=True))
    return m.eval().to(device), meta


@lru_cache(maxsize=1)
def latest_window():
    """토마토 온실 대표(최장) 시계열의 최근 WINDOW일 (7,8). csv 없으면 None.

    env_daily.csv 는 9개 작물 혼합 → '토마토' 품목만 필터(이 앱은 토마토 전용).
    lru_cache — 매 호출 116k행 재로드 방지.
    """
    if not ENV_CSV.exists():
        return None
    import pandas as pd
    df = pd.read_csv(ENV_CSV, encoding="utf-8-sig")
    df = df[df["품목"].astype(str).str.contains("토마토", na=False)]
    best = None
    for _, gdf in df.groupby(SEQ_KEYS):
        arr = gdf.sort_values("날짜")[ENV_FEATURES].to_numpy(np.float32)
        if len(arr) >= WINDOW and (best is None or len(arr) > len(best)):
            best = arr
    return None if best is None else best[-WINDOW:]


def forecast(window):
    """최근 7일 (7,8) → {next_temp, recent_temp, trend, humidity_risk, humidity_mean}. 모델 없으면 None.

    next_temp 만 LSTM 예측. trend·humidity_risk 는 파생값(습도위험=최근 7일 평균 기준).
    """
    loaded = load_forecast_model()
    if loaded is None:
        return None
    model, meta = loaded
    # 학습·추론 피처 순서 불일치 시 '조용히 틀린 정규화'를 막는다(드리프트 가드).
    assert meta["features"] == ENV_FEATURES, "LSTM 메타 피처 순서 불일치 — train_lstm.py 재학습 필요"
    mu = np.asarray(meta["mu"], dtype="float32")
    sd = np.asarray(meta["sd"], dtype="float32")
    x = (np.asarray(window, dtype="float32") - mu) / sd
    with torch.no_grad():
        pred = model(torch.tensor(x[None], dtype=torch.float32).to(device)).item()
    next_temp = pred * float(sd[0]) + float(mu[0])
    recent_temp = float(window[-1, 0])
    diff = next_temp - recent_temp
    trend = "상승" if diff > 0.5 else ("하강" if diff < -0.5 else "유지")   # ±0.5℃ 이내는 유지
    hum = float(np.asarray(window)[:, _HUM_IDX].mean())
    humidity_risk = "높음" if hum >= 80 else ("보통" if hum >= 60 else "낮음")   # 곰팡이 위험 습도 밴드
    return {"next_temp": round(next_temp, 1), "recent_temp": round(recent_temp, 1),
            "trend": trend, "humidity_risk": humidity_risk, "humidity_mean": round(hum, 1)}
