"""
Phase 2 (DL) · STEP 5 · 검출 — YOLO 병해 잎 '위치 검출' (고급/선택)

이 파일 하나 = 청크 2-11. 비전 파트의 마지막 두께:
  · 2-5 분류(CNN)  = "이 잎이 병들었나?"        (사진 1장 → 정상/질병)
  · 2-11 검출(YOLO) = "병든 잎이 '어디' 있나?"   (장면 속 잎 위치 + 정상/질병 라벨)

분류는 잎이 화면을 꽉 채운 사진을 가정하지만, 검출은 **장면에서 잎을 찾아 박스 + 진단**한다.
사전학습 YOLOv8n 을 토마토 3클래스(normal/leaf_mold/tylcv)로 전이학습한다(2-5 전이학습과 같은 태도).

선행:
  python src/dl/prepare_tomato_yolo.py     # 라벨 JSON → YOLO 데이터셋 + data.yaml

실행:
  python src/dl/05_detect.py               # 학습(기본 30 epoch) + 검증(mAP) + 검출 시연
  python src/dl/05_detect.py --epochs 50   # 더 길게
출력: 그림 docs/figures/phase2_dl/10_yolo_detect.png · 모델 models/tomato_yolov8n.pt
"""
import os
import glob
import shutil
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
FIGS = f"{ROOT}/docs/figures/phase2_dl"
DATA = f"{ROOT}/data/tomato_yolo"
MODELS = f"{ROOT}/models"
RUNS = f"{ROOT}/runs"
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")


# ════════════════════════════════════════════════════════════════════
# 청크 2-11 · YOLO 병해 잎 위치 검출
#   ① 사전학습 YOLOv8n → 토마토 2클래스로 전이학습(박스 + 분류 동시 학습)
#   ② 검증 mAP(검출 표준 지표: 박스가 얼마나 정확히 겹치며 맞췄나)
#   ③ val 이미지에 '박스 + 라벨 + 신뢰도'를 그려 위치 검출 시연
# ════════════════════════════════════════════════════════════════════
def run_chunk_2_11(epochs=30):
    print("\n" + "═" * 64 + "\n청크 2-11 · YOLO 병해 잎 위치 검출 (고급/선택)\n" + "═" * 64)
    if not os.path.exists(f"{DATA}/data.yaml"):
        print("⚠️ YOLO 데이터 없음 → python src/dl/prepare_tomato_yolo.py")
        return
    from ultralytics import YOLO

    # ① 사전학습 YOLOv8n 전이학습 (분류의 freeze 와 달리 검출은 보통 전체 미세조정)
    print(f"\n[①] YOLOv8n 전이학습 — {epochs} epoch (data=tomato_yolo)")
    model = YOLO("yolov8n.pt")            # 최초 1회 가중치 자동 다운(~6MB)
    model.train(data=f"{DATA}/data.yaml", epochs=epochs, imgsz=640, batch=16,
                device=device, project=RUNS, name="tomato_yolo", exist_ok=True,
                verbose=False, plots=True)

    # ② 검증 — mAP (박스 IoU 기반 검출 정확도)
    print("\n[②] 검증 (mAP)")
    metrics = model.val(data=f"{DATA}/data.yaml", device=device, verbose=False)
    map50 = float(metrics.box.map50)      # IoU 0.5 기준
    map5095 = float(metrics.box.map)      # IoU 0.5~0.95 평균(더 엄격)
    print(f"  mAP@50 = {map50:.3f}   mAP@50-95 = {map5095:.3f}")

    # 학습 가중치를 models/ 로 복사(다른 청크·데모와 일관)
    best = f"{RUNS}/tomato_yolo/weights/best.pt"
    if os.path.exists(best):
        shutil.copy(best, f"{MODELS}/tomato_yolov8n.pt")
        print(f"  best.pt → {MODELS}/tomato_yolov8n.pt")

    # ③ 검출 시연 — val 이미지 4장(질병 2 + 정상 2)에 박스 그려 저장
    print("\n[③] 검출 시연 (박스 + 라벨 + 신뢰도)")
    mold = sorted(glob.glob(f"{DATA}/images/val/leaf_mold_*.jpg"))[:1]
    tylcv = sorted(glob.glob(f"{DATA}/images/val/tylcv_*.jpg"))[:1]
    nor = sorted(glob.glob(f"{DATA}/images/val/normal_*.jpg"))[:2]
    shots = mold + tylcv + nor
    if not shots:
        print("  ⚠️ val 이미지 없음")
        return
    results = model.predict(shots, device=device, conf=0.25, verbose=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, r, path in zip(axes.ravel(), results, shots):
        ax.imshow(r.plot()[:, :, ::-1])   # plot()=BGR → RGB
        ax.set_title(os.path.basename(path)[:22], fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(shots):]:
        ax.axis("off")
    fig.suptitle(f"YOLO 토마토 병해 잎 검출 — 박스 위치 + 진단  "
                 f"(mAP@50={map50:.2f})", fontsize=13)
    fig.tight_layout()
    path = f"{FIGS}/10_yolo_detect.png"
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  검출 그림 저장 → {path}")
    print("  → 분류(2-5)는 '병들었나'만, YOLO 는 '어디에 있는 무엇'까지 — 비전 파트의 마지막 두께.")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 · 청크 2-11 YOLO 검출")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run_chunk_2_11(epochs=args.epochs)
    print("\n✅ 청크 2-11 완료 — 비전 파트(분류+설명+검출) 마무리")
