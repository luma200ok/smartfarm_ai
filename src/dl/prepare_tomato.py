"""
Phase 2 (DL) · STEP 2 데이터 준비 — 토마토 잎 병해 3분류 (AI Hub '시설작물 질병진단')

2-5 전이학습 · 2-6 Grad-CAM · 2-7/2-9 평가가 쓸 데이터를 만든다.
원본(zip) → torchvision ImageFolder 구조로 해제·정리:
  data/tomato/{train,val}/{normal,leaf_mold,tylcv}/*.jpg

★ 3분류로 확장한 이유:
  토마토 '질병'은 실제로 **두 종류**다(라벨 JSON의 disease 코드).
    · disease=18 → leaf_mold  (토마토 잎곰팡이병)
    · disease=19 → tylcv      (토마토 황화잎말이바이러스)
  이전엔 둘을 'disease' 하나로 뭉갰지만, 라벨이 구분돼 있으므로 정상/잎곰팡이/황화잎말이 3분류로 쓴다.
  (코드→병명은 AI Hub 071 코드 체계. 폴더명에 코드 의미를 남겨 추적 가능.)

원본 위치:
  ~/Downloads/071.시설 작물 질병 진단/01.데이터/{1.Training,2.Validation}/
    원천데이터/11.토마토/11.토마토_{0.정상,1.질병}.zip          ← 이미지
    라벨링데이터/11.토마토/[라벨]11.토마토_{0.정상,1.질병}.zip   ← disease 코드 JSON

가공 방식:
  · normal = 정상 원천 zip 이미지 그대로(라벨 불필요).
  · 질병 = 원천 이미지를 라벨 JSON과 파일명으로 짝지어 disease 코드로 leaf_mold/tylcv 라우팅.
  · train/val 을 겹치지 않게 분할(같은 이미지 양쪽 유입=누수 방지), 256px 로 축소 저장.

실행(한 번만, 수 분):
  python src/dl/prepare_tomato.py                  # train 클래스당 ≤120, val 그 1/4
  python src/dl/prepare_tomato.py --per-class 200  # 더 크게(데이터 있으면)
"""
import os
import io
import json
import zipfile
import argparse
import random

from PIL import Image

HOME = os.path.expanduser("~")
SRC = f"{HOME}/Downloads/071.시설 작물 질병 진단/01.데이터"
ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
OUT = f"{ROOT}/data/tomato"
RESIZE = 256
SPLIT_DIRS = ["2.Validation", "1.Training"]

# disease 코드 → ImageFolder 클래스 폴더명 (토마토; AI Hub 071 코드 체계)
DISEASE_MAP = {18: "leaf_mold", 19: "tylcv"}     # 잎곰팡이병 · 황화잎말이바이러스
CLASSES = ["normal", "leaf_mold", "tylcv"]

random.seed(42)


# ── 원천(이미지) zip: basename → (zip, member) 인덱스 ──
def _index_images(cls_kr):
    idx = {}
    for sp in SPLIT_DIRS:
        zp = f"{SRC}/{sp}/원천데이터/11.토마토/11.토마토_{cls_kr}.zip"
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if n.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx.setdefault(os.path.basename(n), (zp, n))
    return idx


# ── 라벨(JSON) zip: basename(.json제거) → (zip, member) ──
def _index_labels(cls_kr):
    idx = {}
    for sp in SPLIT_DIRS:
        zp = f"{SRC}/{sp}/라벨링데이터/11.토마토/[라벨]11.토마토_{cls_kr}.zip"
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".json"):
                    idx.setdefault(os.path.basename(n)[:-5], (zp, n))
    return idx


# ── 이미지 1장 추출·리사이즈·저장 ──
def _save_image(zip_path, member, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        img = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
    img.thumbnail((RESIZE, RESIZE))
    img.save(os.path.join(out_dir, f"{stem}.jpg"), quality=88)


# ── 클래스별 (zip,member,stem) 목록 수집 ──
def _collect():
    """{클래스명: [(img_zip, img_member, stem), ...]} 반환."""
    buckets = {c: [] for c in CLASSES}

    # 정상: 라벨 불필요
    nimg = _index_images("0.정상")
    for base, (zp, m) in nimg.items():
        buckets["normal"].append((zp, m, base))

    # 질병: 이미지 ∩ 라벨, disease 코드로 라우팅
    dimg = _index_images("1.질병")
    dlbl = _index_labels("1.질병")
    for base in set(dimg) & set(dlbl):
        lzip, lmem = dlbl[base]
        with zipfile.ZipFile(lzip) as zf:
            code = json.loads(zf.read(lmem))["annotations"].get("disease")
        cls = DISEASE_MAP.get(code)
        if cls is None:            # 코드 18·19 외 소수 잡음은 제외
            continue
        izip, imem = dimg[base]
        buckets[cls].append((izip, imem, base))
    return buckets


# ── (실행) 전체 파이프라인 ──
def main():
    parser = argparse.ArgumentParser(description="토마토 → ImageFolder 3분류(normal/leaf_mold/tylcv)")
    parser.add_argument("--per-class", type=int, default=120,
                        help="train 클래스당 최대 장수(기본 120). val 은 이 값의 1/4.")
    args = parser.parse_args()

    if not os.path.isdir(SRC):
        print(f"⛔ 원본 폴더가 없습니다: {SRC}")
        return

    n_train = args.per_class
    n_val = max(args.per_class // 4, 20)
    print(f"출력: {OUT}  (256px, train≤{n_train}/클래스, val≤{n_val}/클래스)\n")

    buckets = _collect()
    total = 0
    for cls in CLASSES:
        items = buckets[cls]
        random.shuffle(items)
        n_tr = min(n_train, int(len(items) * 0.8))
        n_va = min(n_val, len(items) - n_tr)
        print(f"  [{cls}] 확보 {len(items)}장 → train {n_tr} / val {n_va}")
        for split, names in (("train", items[:n_tr]), ("val", items[n_tr:n_tr + n_va])):
            for zp, m, stem in names:
                _save_image(zp, m, os.path.join(OUT, split, cls), f"{cls}_{stem}")
                total += 1

    print(f"\n✅ 완료 — 총 {total}장 → {OUT}")
    print(f"   클래스 = {CLASSES}  (normal=정상 · leaf_mold=잎곰팡이병(18) · tylcv=황화잎말이(19))")
    print("   이제: python src/dl/02_core.py --chunk 2-5   (전이학습 3분류)")


if __name__ == "__main__":
    main()
