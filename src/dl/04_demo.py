"""
Phase 2 (DL) · STEP 4 · 데모 — 모델 저장 · 추론 파이프라인 · 회고

이 파일 하나 = STEP 4 전체. 청크 2개가 블록으로 들어있다:
  · 청크 2-10  모델 저장·추론 + Streamlit 데모(app/phase2_dl.py) — 사진 → 진단 + Grad-CAM
  · 청크 2-12  회고·참고문헌 → 포트폴리오 phase2_dl.md (체크리스트 출력)

2-5 가 저장한 models/tomato_resnet18.pt 를 추론에 재사용한다(Grad-CAM 위해 resnet18).
실행:
  python src/dl/04_demo.py --chunk 2-10       # 저장 모델로 1장 추론 시연
  streamlit run app/phase2_dl.py              # 웹 데모(사진 업로드 → 진단 + 히트맵)
"""
import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
MODELS = f"{ROOT}/models"
TOMATO = f"{ROOT}/data/tomato"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}  (torch {torch.__version__})")

CLASSES = ["leaf_mold", "normal", "tylcv"]   # ImageFolder 알파벳순(2-5와 동일, 3분류)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── (공용①) 모델 로드 ──
def load_resnet18(ckpt, n_classes=3):
    """추론용 resnet18 — 사전학습 weight 불필요(우리 학습 가중치를 덮어쓰므로)."""
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.eval().to(device)


# ── (공용②) 전처리 ──
def preprocess(pil):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return tf(pil.convert("RGB")).unsqueeze(0)


# ── (공용③) 예측+CAM ──
def predict(model, pil, want_cam=True):
    """한 장 추론 → {label, prob, probs, cam}. cam 은 2-6과 같은 Grad-CAM(224×224, 0~1)."""
    x = preprocess(pil).to(device)
    store = {}
    handle = None
    if want_cam:
        x = x.requires_grad_(True)
        layer = model.layer4[-1]

        def _save(m, i, o):
            o.retain_grad()
            store["act"] = o
        handle = layer.register_forward_hook(_save)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    cls = int(probs.argmax())

    cam = None
    if want_cam:
        model.zero_grad()
        logits[0, cls].backward()
        act = store["act"].detach()[0]
        grad = store["act"].grad[0]
        w = grad.mean(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * act).sum(0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(224, 224),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        handle.remove()
    return {"label": CLASSES[cls], "prob": float(probs[cls]),
            "probs": probs.detach().cpu().numpy(), "cam": cam}


# ════════════════════════════════════════════════════════════════════
# 청크 2-10 · 모델 저장 · 추론 파이프라인 + Streamlit 데모
#   ① 2-5 가 저장한 .pt 를 로드해 '한 장 → 진단' 추론 함수로 정리
#   ② 실제 val 이미지 1장으로 추론 시연(label·확률)
#   ③ 웹 데모는 app/phase2_dl.py (streamlit run)
# ════════════════════════════════════════════════════════════════════
# ── (실행) 추론 파이프라인 시연 ──
def run_chunk_2_10():
    print("\n" + "═" * 64 + "\n청크 2-10 · 모델 저장 · 추론 + Streamlit\n" + "═" * 64)
    ckpt = f"{MODELS}/tomato_resnet18.pt"
    if not os.path.exists(ckpt):
        print("⚠️ 2-5 학습 모델 없음 → python src/dl/02_core.py --chunk 2-5")
        return

    model = load_resnet18(ckpt)
    print(f"\n[①] 모델 로드 → {ckpt}")

    # ② val 이미지 1장으로 추론 시연
    samples = sorted(glob.glob(f"{TOMATO}/val/leaf_mold/*.jpg"))
    if not samples:
        print("⚠️ 추론할 val 이미지 없음 → python src/dl/prepare_tomato.py")
        return
    pil = Image.open(samples[0])
    r = predict(model, pil)
    print(f"\n[②] 추론 시연: {os.path.basename(samples[0])}")
    print(f"  진단 = {r['label']}  (확률 {r['prob']:.3f})")
    print(f"  클래스별 확률: " + ", ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, r['probs'])))
    print(f"  Grad-CAM 히트맵 shape = {r['cam'].shape}")

    print("\n[③] 웹 데모 실행:")
    print("  streamlit run app/phase2_dl.py")
    print("  → 잎 사진 업로드 → 정상/질병 진단 + 어디를 보고 판단했는지 히트맵")


# ════════════════════════════════════════════════════════════════════
# 청크 2-12 · 회고 · 참고문헌 (포트폴리오 phase2_dl.md 로)
#   학습 로그용 짧은 체크리스트 — 정식 수행내역서는 docs/phase2_dl.md 에 작성
# ════════════════════════════════════════════════════════════════════
# ── (실행) 회고·참고문헌 출력 ──
def run_chunk_2_12():
    print("\n" + "═" * 64 + "\n청크 2-12 · 회고 · 참고문헌\n" + "═" * 64)
    items = [
        "STEP 1 기초: 뉴런·활성화·역전파·DataLoader 를 손계산 대조로 검증",
        "STEP 2 핵심: CNN → 전이학습(백본 비교) → Grad-CAM → LSTM",
        "STEP 3 평가: 클래스 가중치로 소수클래스(질병) recall 개선 → ROC/AUC·FN 분석",
        "STEP 4 데모: .pt 저장 → Streamlit(사진 업로드 → 진단 + 히트맵)",
        "차별화: ML 로 불가능한 '사진 진단' + 설명가능 AI(Grad-CAM) + 시계열(LSTM)",
        "한계/다음: 토마토 1작물 → 다작물 확장, 질병 종류 다중분류, YOLO 병반 위치(2-11)",
    ]
    print("\n[회고 체크리스트] → 정식 정리는 docs/phase2_dl.md")
    for it in items:
        print(f"  · {it}")
    print("\n  참고: PlantVillage / AI Hub 시설작물 질병진단 · torchvision 사전학습 백본 · Grad-CAM(Selvaraju 2017)")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 STEP 4 데모 — 청크 2-10·2-12")
    parser.add_argument("--chunk", choices=["2-10", "2-12", "all"], default="all")
    args = parser.parse_args()

    runners = {"2-10": run_chunk_2_10, "2-12": run_chunk_2_12}
    if args.chunk == "all":
        for run in runners.values():
            run()
        print("\n✅ STEP 4(데모) 완료 — Phase 2 DL 코드 끝! 정식 정리 → docs/phase2_dl.md")
    else:
        runners[args.chunk]()
        print(f"\n✅ 청크 {args.chunk} 완료")
