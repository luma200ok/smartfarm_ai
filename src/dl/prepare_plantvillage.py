"""
Phase 2 (DL) · 도메인 보강 — PlantVillage(CC0) 토마토 잎을 학습셋에 혼합

배경: 진단 모델은 AI Hub 071(현장 촬영, 복잡한 배경)로만 학습됐다. 그런데 라이선스
정리(2026-06-30) 때 공개 예시를 PlantVillage(균일한 회색 배경)로 교체하면서, 모델이
한 번도 본 적 없는 분포가 데모에 입력됐다 → 정상 잎이 tylcv 로 오분류(PV normal 정답률 10%).

대응: PlantVillage(CC0 1.0, spMohanty/PlantVillage-Dataset)의 토마토 3클래스를 받아
AI Hub data/tomato 에 섞는다. head 만 학습하는 freeze 전략이라도 PV 분포를 함께 보면
공개 예시 도메인에서도 normal 을 normal 로 잡게 된다.

클래스 매핑(PV 폴더 → 진단 클래스):
  Tomato___healthy                        → normal
  Tomato___Leaf_Mold                      → leaf_mold
  Tomato___Tomato_Yellow_Leaf_Curl_Virus  → tylcv

저장: 256px(AI Hub 가공과 동일) · 파일 prefix `pv_`(출처 구분·예시 holdout 추적)
  data/tomato/train/{cls}/pv_*.jpg   (클래스당 --per-class, 기본 300)
  data/tomato/val/{cls}/pv_*.jpg     (클래스당 --val, 기본 30 — 데모 예시는 여기서 고름)

데이터 누수 방지: train ∩ val 파일명 겹침 없음(셔플 후 분할). 데모 예시(app/samples)는
재학습 뒤 val(holdout) 이미지에서 새로 고른다(별도 스크립트) → 학습에 안 쓴 사진만 노출.

실행(한 번, 네트워크 필요):
  python src/dl/prepare_plantvillage.py                 # 클래스당 train 300 / val 30
  python src/dl/prepare_plantvillage.py --per-class 200 # 더 작게
이후 재학습:
  python src/dl/02_core.py --chunk 2-5
"""
import io
import os
import json
import time
import random
import argparse
import urllib.request

from PIL import Image

ROOT = "/Users/jeongjaebong/IntelliJ/mycode/toy_project/solo/smartfarm_ai"
OUT = f"{ROOT}/data/tomato"
RESIZE = 256

API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{folder}?ref=master"
# PV 폴더명 → 진단 클래스
PV_MAP = {
    "Tomato___healthy": "normal",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tylcv",
}
random.seed(42)


def _list_files(folder):
    """GitHub API 로 폴더 내 이미지의 (name, download_url) 목록."""
    req = urllib.request.Request(API.format(folder=folder),
                                 headers={"User-Agent": "smartfarm-prepare"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return [(x["name"], x["download_url"]) for x in data
            if x["name"].lower().endswith((".jpg", ".jpeg", ".png"))]


def _fetch_resave(url, out_path):
    """원본 1장 다운로드 → 256px 축소 저장. 실패 시 False."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smartfarm-prepare"})
        with urllib.request.urlopen(req, timeout=30) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        img.thumbnail((RESIZE, RESIZE))
        img.save(out_path, quality=88)
        return True
    except Exception as e:
        print(f"      ⚠️ 실패 {os.path.basename(out_path)}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="PlantVillage(CC0) 토마토 3클래스 → data/tomato 혼합")
    ap.add_argument("--per-class", type=int, default=300, help="클래스당 train 장수(기본 300)")
    ap.add_argument("--val", type=int, default=30, help="클래스당 val(holdout) 장수(기본 30)")
    args = ap.parse_args()

    grand = 0
    for folder, cls in PV_MAP.items():
        need = args.per_class + args.val
        print(f"\n[{cls}] ← PlantVillage/{folder}  (목표 train {args.per_class} / val {args.val})")
        files = _list_files(folder)
        random.shuffle(files)
        files = files[:need]
        print(f"   원본 {len(files)}장 확보 → 다운로드·256px 저장")

        n_tr = min(args.per_class, int(len(files) * 0.9))
        splits = [("train", files[:n_tr]), ("val", files[n_tr:need])]
        saved = 0
        for split, items in splits:
            d = f"{OUT}/{split}/{cls}"
            os.makedirs(d, exist_ok=True)
            for name, url in items:
                stem = os.path.splitext(name)[0]
                if _fetch_resave(url, f"{d}/pv_{stem}.jpg"):
                    saved += 1
                time.sleep(0.02)
            print(f"   [{split}] {len(items)}장")
        grand += saved
        print(f"   ✅ {cls} {saved}장 저장")

    print(f"\n✅ PlantVillage 혼합 완료 — 총 {grand}장 추가")
    print("   다음: python src/dl/02_core.py --chunk 2-5   (진단 재학습)")


if __name__ == "__main__":
    main()
