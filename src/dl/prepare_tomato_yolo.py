"""
Phase 2 (DL) · STEP 5 데이터 준비 — 토마토 YOLO 검출용 (AI Hub '시설작물 질병진단')

2-11 YOLO(병해 잎 '위치 검출')가 쓸 데이터를 만든다.
2-5 분류(prepare_tomato.py)와 다른 점: 분류는 '폴더=클래스' 라벨만 필요했지만,
검출은 **바운딩박스 좌표**가 필요하다 → 원본의 '라벨링데이터'(JSON)를 파싱해 YOLO 포맷으로 변환.

원본 구조:
  ~/Downloads/071.시설 작물 질병 진단/01.데이터/{1.Training,2.Validation}/
    원천데이터/11.토마토/11.토마토_{0.정상,1.질병}.zip          ← 이미지(jpg)
    라벨링데이터/11.토마토/[라벨]11.토마토_{0.정상,1.질병}.zip   ← 박스 JSON

라벨 JSON 구조(이미지 1장 = 박스 1개):
  description.width/height                       원본 해상도(4032×3024)
  annotations.points[0] = {xtl,ytl,xbr,ybr}      잎 영역 박스(픽셀)
  → 클래스는 폴더(정상=0 / 질병=1)로 확정(disease 코드보다 안전)

출력(ultralytics 표준 구조):
  data/tomato_yolo/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt        # 한 줄: "cls cx cy w h" (0~1 정규화)
    data.yaml

※ 정규화 좌표는 리사이즈와 무관(원본 w/h로 나눔) → 이미지는 640px 로 줄여 저장해도 박스 유효.
※ 질병 원천이 246장뿐 → 정상도 같은 규모로 맞춰 균형 잡음.

실행(한 번만):
  python src/dl/prepare_tomato_yolo.py                  # train 150/클래스, val 40/클래스
  python src/dl/prepare_tomato_yolo.py --per-class 200  # 더 크게
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
OUT = f"{ROOT}/data/tomato_yolo"
RESIZE = 640
SPLITS = ["1.Training", "2.Validation"]
# YOLO 클래스 id: 0 정상 · 1 잎곰팡이병(disease=18) · 2 황화잎말이(disease=19)
CLASS_NAMES = ["normal", "leaf_mold", "tylcv"]
DISEASE_MAP = {18: 1, 19: 2}                          # 라벨 disease 코드 → 클래스 id

random.seed(42)


# ── ① 원천(이미지) zip 인덱스: basename → (zip경로, 멤버명) ──
def index_source_images(cls_kr):
    idx = {}
    for sp in SPLITS:
        zp = f"{SRC}/{sp}/원천데이터/11.토마토/11.토마토_{_tag(cls_kr)}.zip"
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if n.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx.setdefault(os.path.basename(n), (zp, n))
    return idx


# ── ② 라벨(JSON) 목록: basename(.json제거) → (zip경로, 멤버명) ──
def index_labels(cls_kr):
    idx = {}
    for sp in SPLITS:
        zp = f"{SRC}/{sp}/라벨링데이터/11.토마토/[라벨]11.토마토_{_tag(cls_kr)}.zip"
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".json"):
                    idx.setdefault(os.path.basename(n)[:-5], (zp, n))   # x.jpg.json → x.jpg
    return idx


def _tag(cls_kr):
    return "0.정상" if cls_kr == "정상" else "1.질병"


# ── ③ JSON → (YOLO 정규화 박스 "cx cy w h", disease 코드) ──
def to_yolo_box(label_bytes):
    d = json.loads(label_bytes.decode("utf-8"))
    W = d["description"]["width"]; H = d["description"]["height"]
    a = d["annotations"]; p = a["points"][0]
    cx = (p["xtl"] + p["xbr"]) / 2 / W
    cy = (p["ytl"] + p["ybr"]) / 2 / H
    w = (p["xbr"] - p["xtl"]) / W
    h = (p["ybr"] - p["ytl"]) / H
    clip = lambda v: max(0.0, min(1.0, v))
    return (clip(cx), clip(cy), clip(w), clip(h)), a.get("disease")


# ── ④ 한 쌍(이미지+라벨) 저장 ──
def save_pair(img_zip, img_member, box, cls_id, split, stem):
    img_dir = f"{OUT}/images/{split}"; lbl_dir = f"{OUT}/labels/{split}"
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    with zipfile.ZipFile(img_zip) as zf:
        img = Image.open(io.BytesIO(zf.read(img_member))).convert("RGB")
    img.thumbnail((RESIZE, RESIZE))                       # 정규화 박스라 리사이즈 무관
    img.save(f"{img_dir}/{stem}.jpg", quality=88)
    with open(f"{lbl_dir}/{stem}.txt", "w") as f:
        f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


# ── (실행) 전체 변환 파이프라인 ──
def main():
    parser = argparse.ArgumentParser(description="토마토 라벨 JSON → YOLO 검출 데이터셋")
    parser.add_argument("--per-class", type=int, default=150,
                        help="train 클래스당 최대 장수(기본 150). val 은 이 값의 ~1/4.")
    args = parser.parse_args()

    if not os.path.isdir(SRC):
        print(f"⛔ 원본 폴더가 없습니다: {SRC}")
        return

    n_train = args.per_class
    n_val = max(args.per_class // 4, 20)
    print(f"출력: {OUT}  (640px, train≤{n_train}/클래스, val≤{n_val}/클래스)\n")

    # 클래스별 (이미지위치, 라벨위치, stem) 수집
    buckets = {0: [], 1: [], 2: []}
    nimg, nlbl = index_source_images("정상"), index_labels("정상")
    for stem in sorted(set(nimg) & set(nlbl)):
        buckets[0].append((nimg[stem], nlbl[stem], stem))      # 정상 → 클래스 0
    dimg, dlbl = index_source_images("질병"), index_labels("질병")
    for stem in sorted(set(dimg) & set(dlbl)):
        lzip, lmem = dlbl[stem]
        with zipfile.ZipFile(lzip) as zf:
            _, code = to_yolo_box(zf.read(lmem))
        cid = DISEASE_MAP.get(code)                            # 18→1, 19→2, 그 외 제외
        if cid is None:
            continue
        buckets[cid].append((dimg[stem], dlbl[stem], stem))

    total = 0
    for cid in (0, 1, 2):
        items = buckets[cid]; random.shuffle(items)
        n_tr = min(n_train, int(len(items) * 0.8))
        n_va = min(n_val, len(items) - n_tr)
        cname = CLASS_NAMES[cid]
        print(f"  [{cname}] 매칭 {len(items)}쌍 → train {n_tr} / val {n_va}")
        for split, group in (("train", items[:n_tr]), ("val", items[n_tr:n_tr + n_va])):
            for (izip, imem), (lzip, lmem), stem in group:
                with zipfile.ZipFile(lzip) as zf:
                    box, _ = to_yolo_box(zf.read(lmem))
                save_pair(izip, imem, box, cid, split, f"{cname}_{stem}")
                total += 1

    # data.yaml (ultralytics 학습이 읽는 설정)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml = f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    with open(f"{OUT}/data.yaml", "w") as f:
        f.write(yaml)

    print(f"\n✅ 완료 — 총 {total}장 + data.yaml → {OUT}")
    print(f"   클래스 = {CLASS_NAMES}  (normal · leaf_mold(18) · tylcv(19))")
    print("   이제: python src/dl/05_detect.py   (YOLO 학습·검출)")


if __name__ == "__main__":
    main()
