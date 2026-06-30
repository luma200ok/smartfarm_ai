"""
Phase 2 (DL) · 도메인 교차 검증 — 진단 모델이 두 도메인에서 모두 동작하는지 점검

재학습(PlantVillage 혼합) 전후로 normal 정답률이 어떻게 바뀌는지 본다.
  · AI Hub val(data/tomato/val, pv_ 제외)  = 학습 원도메인(회귀가 없어야 함)
  · PlantVillage val(pv_ 만)               = 공개 예시 도메인(여기가 개선 타깃)
  · app/samples 3장                        = 데모에 실제 노출되는 사진

실행:  python src/dl/eval_domain.py
"""
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

ROOT = Path("/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai")
CKPT = ROOT / "models" / "tomato_resnet18.pt"
CLASSES = ["leaf_mold", "normal", "tylcv"]            # ImageFolder 알파벳순
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
device = "mps" if torch.backends.mps.is_available() else "cpu"
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])


def load():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    m.load_state_dict(torch.load(CKPT, map_location=device))
    return m.eval().to(device)


def predict(m, path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(m(x)[0], 0)
    return CLASSES[int(p.argmax())], {c: round(float(v), 3) for c, v in zip(CLASSES, p)}


def eval_folder(m, folder, true_cls, pv):
    """folder 안에서 출처(pv 여부) 필터 후 정답률. pv=True→pv_*, False→pv_ 제외."""
    paths = [p for p in Path(folder).glob("*.jpg")
             if (p.name.startswith("pv_") == pv)]
    if not paths:
        return None
    hit = sum(predict(m, p)[0] == true_cls for p in paths)
    return hit, len(paths)


def main():
    if not CKPT.exists():
        print(f"⛔ 모델 없음: {CKPT}")
        return
    m = load()

    for title, pv in [("AI Hub val (학습 원도메인)", False),
                      ("PlantVillage val (공개 예시 도메인)", True)]:
        print(f"\n[{title}]")
        for cls in CLASSES:
            r = eval_folder(m, ROOT / "data/tomato/val" / cls, cls, pv)
            if r:
                hit, n = r
                print(f"   {cls:10s} 정답률 {hit/n*100:5.1f}%  ({hit}/{n})")

    print("\n[app/samples 데모 예시 3장]")
    for name, true_cls in [("normal", "normal"), ("leaf_mold", "leaf_mold"), ("tylcv", "tylcv")]:
        p = ROOT / "app/samples" / f"{name}.jpg"
        if p.exists():
            pred, probs = predict(m, p)
            ok = "✅" if pred == true_cls else "❌"
            print(f"   {ok} {name}.jpg → 예측 {pred}  {probs}")


if __name__ == "__main__":
    main()
