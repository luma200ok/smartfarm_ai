"""
Phase 2 (DL) · STEP 2 데이터 준비 — 토마토 잎 병해 (AI Hub '시설작물 질병진단')

2-5 전이학습 · 2-6 Grad-CAM 이 쓸 데이터를 만든다.
원본(zip) → torchvision ImageFolder 구조로 해제·정리:
  data/tomato/{train,val}/{normal,disease}/*.jpg

원본 위치(zip, 14GB):
  ~/Downloads/071.시설 작물 질병 진단/01.데이터/
    {1.Training, 2.Validation}/원천데이터/11.토마토/11.토마토_{0.정상,1.질병,9.증강}.zip
  → 0.정상 = normal · 1.질병 = disease   (9.증강은 원본 아님 → 제외)

왜 이렇게 가공하나:
  · 클래스 불균형(정상 7GB ≫ 질병 0.36GB) → 정상을 클래스 상한으로 다운샘플(--per-class)
  · 원본이 4032×3024 로 너무 큼 → 256px 로 줄여 저장(디스크·로딩 ↓)
  · ImageFolder = '클래스명 폴더에 이미지만 넣으면 끝' → 라벨 JSON 파싱 불필요

실행(한 번만, 수 분 소요):
  python src/dl/prepare_tomato.py                  # train 클래스당 1500, val 400
  python src/dl/prepare_tomato.py --per-class 800  # 더 작게(빠른 실험)
"""
import os
import io
import zipfile
import argparse
import random

from PIL import Image

HOME = os.path.expanduser("~")
SRC = f"{HOME}/Downloads/071.시설 작물 질병 진단/01.데이터"
ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
OUT = f"{ROOT}/data/tomato"
RESIZE = 256

# 원본 폴더명 → (split 폴더, 클래스 폴더)
SPLITS = {"1.Training": "train", "2.Validation": "val"}
CLASSES = {"정상": "normal", "질병": "disease"}   # 9.증강은 키에 없으므로 자동 제외

random.seed(42)


def find_zip(split_kr, cls_kr):
    """원천데이터/11.토마토 안에서 해당 클래스 zip 1개를 찾는다."""
    base = os.path.join(SRC, split_kr, "원천데이터", "11.토마토")
    if not os.path.isdir(base):
        return None
    for fn in os.listdir(base):
        # 파일명 예: '11.토마토_0.정상.zip' / '11.토마토_1.질병.zip'
        if fn.endswith(".zip") and cls_kr in fn and "증강" not in fn:
            return os.path.join(base, fn)
    return None


def extract_resized(zip_path, out_dir, limit, resize):
    """zip 안 이미지를 limit 장까지 256px 로 줄여 out_dir 에 저장."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(names)                      # 한쪽에 치우치지 않게 섞어서 limit 추출

    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        for n in names:
            if saved >= limit:
                break
            try:
                img = Image.open(io.BytesIO(zf.read(n))).convert("RGB")
                img.thumbnail((resize, resize))            # 비율 유지하며 축소
                img.save(os.path.join(out_dir, f"{saved:05d}.jpg"), quality=88)
                saved += 1
                if saved % 200 == 0:
                    print(f"    {saved}/{limit} …")
            except Exception:
                continue                                   # 깨진 이미지 건너뜀
    return saved


def main():
    parser = argparse.ArgumentParser(description="토마토 zip → ImageFolder 정리")
    parser.add_argument("--per-class", type=int, default=1500,
                        help="train 클래스당 최대 장수(기본 1500). val 은 이 값의 약 1/4.")
    parser.add_argument("--resize", type=int, default=RESIZE)
    args = parser.parse_args()

    if not os.path.isdir(SRC):
        print(f"⛔ 원본 폴더가 없습니다: {SRC}")
        print("   AI Hub '시설작물 질병진단' 압축을 위 경로에 두고 다시 실행하세요.")
        return

    print(f"원본: {SRC}")
    print(f"출력: {OUT}  (256px, train≤{args.per_class}/클래스)\n")

    total = 0
    for split_kr, split_en in SPLITS.items():
        limit = args.per_class if split_en == "train" else max(args.per_class // 4, 100)
        for cls_kr, cls_en in CLASSES.items():
            zp = find_zip(split_kr, cls_kr)
            out_dir = os.path.join(OUT, split_en, cls_en)
            if zp is None:
                print(f"  ⚠️ {split_en}/{cls_en}: zip 못 찾음(건너뜀)")
                continue
            print(f"  {split_en}/{cls_en} ← {os.path.basename(zp)} (최대 {limit}장)")
            n = extract_resized(zp, out_dir, limit, args.resize)
            print(f"    → {n}장 저장")
            total += n

    print(f"\n✅ 완료 — 총 {total}장 → {OUT}")
    print("   이제: python src/dl/02_core.py --chunk 2-5   (전이학습)")


if __name__ == "__main__":
    main()
