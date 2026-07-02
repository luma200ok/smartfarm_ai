"""
이슈 #6 1단계 — 기상청(KMA) 단기예보 API 클라이언트.

목적: 실내 센서만으로는 이상의 "원인"(외부 날씨)을 알 수 없어 외부 기상 데이터를 결합.
- get_current(): 초단기실황(getUltraSrtNcst) — 기온·습도·강수
- get_forecast_3d(): 단기예보(getVilageFcst) — 3일 최저/최고·시간별 기온/습도/강수확률/하늘상태

컨벤션(다른 llm 모듈과 동일):
- KMA_SERVICE_KEY는 호출 시점 os.getenv()로 읽는다(모듈 상수 캐싱 금지) — rag/__init__.py:19-23 참고.
- 키 미설정/호출 실패는 예외를 밖으로 전파하지 않고 {"unavailable": True, "reason": "..."} 반환
  (tools.get_forecast·notify.send_discord 와 동일한 best-effort 원칙).
- requests 외 새 의존성 추가 금지.
"""
import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# override=True: 쉘 전역 env보다 프로젝트 .env가 우선 (notify.py와 동일 이유)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

_log = logging.getLogger(__name__)
_TIMEOUT = 10

_BASE_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"

# 서울 기본값(농장 좌표 미설정 시)
_SEOUL_LAT, _SEOUL_LON = 37.5665, 126.9780

# ── TTL 캐시: (endpoint, nx, ny) -> (ts, data) ──────────────────────────
_CACHE: dict[tuple[str, int, int], tuple[float, dict]] = {}
_TTL = {"getUltraSrtNcst": 600, "getVilageFcst": 1800}


def _farm_location() -> tuple[float, float]:
    """농장 위치. env FARM_LAT/FARM_LON 없으면 서울 기본값."""
    try:
        lat = float(os.getenv("FARM_LAT") or _SEOUL_LAT)
        lon = float(os.getenv("FARM_LON") or _SEOUL_LON)
    except ValueError:
        lat, lon = _SEOUL_LAT, _SEOUL_LON
    return lat, lon


def _to_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도 → 기상청 격자(nx, ny). Lambert Conformal Conic 변환(기상청 공개 공식).

    서울(37.5665, 126.9780) → (nx=60, ny=127) 로 검증됨.
    """
    import math

    RE = 6371.00877       # 지구 반경(km)
    GRID = 5.0            # 격자 간격(km)
    SLAT1 = 30.0          # 투영 위도1
    SLAT2 = 60.0          # 투영 위도2
    OLON = 126.0          # 기준점 경도
    OLAT = 38.0           # 기준점 위도
    XO = 43               # 기준점 X좌표(GRID)
    YO = 136              # 기준점 Y좌표(GRID)

    DEGRAD = math.pi / 180.0

    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)


def _cache_get(endpoint: str, nx: int, ny: int) -> dict | None:
    key = (endpoint, nx, ny)
    hit = _CACHE.get(key)
    if hit is None:
        return None
    ts, data = hit
    if time.time() - ts > _TTL[endpoint]:
        return None
    return data


def _cache_put(endpoint: str, nx: int, ny: int, data: dict) -> None:
    _CACHE[(endpoint, nx, ny)] = (time.time(), data)


def clear_cache() -> None:
    """TTL 캐시 비우기 — UI '새로고침' 버튼 등 강제 재조회용(공개 API)."""
    _CACHE.clear()


def _ultra_srt_base(now=None) -> tuple[str, str]:
    """초단기실황 base_date/base_time — 매시 40분 이후 제공, 그 전이면 직전 정시-1h.

    now 는 테스트 주입용(기본: 현재 시각). 자정 경계(00:00~00:39)면 전날 23시.
    """
    from datetime import datetime, timedelta

    if now is None:
        now = datetime.now()
    if now.minute < 40:
        now = now - timedelta(hours=1)
    return now.strftime("%Y%m%d"), now.strftime("%H00")


_VILAGE_FCST_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]


def _vilage_fcst_base(now=None) -> tuple[str, str]:
    """단기예보 base_date/base_time — 발표시각(02,05,08,11,14,17,20,23) +10분 제공.

    현재 시각 이전 가장 최근 발표시각 선택. 당일 02:10 이전이면 전날 23시.
    now 는 테스트 주입용(기본: 현재 시각).
    """
    from datetime import datetime, timedelta

    if now is None:
        now = datetime.now()
    candidates = [now.replace(hour=h, minute=10, second=0, microsecond=0) for h in _VILAGE_FCST_HOURS]
    past = [c for c in candidates if c <= now]
    if past:
        chosen = max(past)
    else:
        chosen = (now - timedelta(days=1)).replace(hour=23, minute=10, second=0, microsecond=0)
    return chosen.strftime("%Y%m%d"), chosen.strftime("%H00")


def _normalize_items(raw) -> list:
    """items.item 정규화 — 공공데이터포털은 결과 1건이면 dict로 내려올 수 있음."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    return []


def _request(endpoint: str, params: dict) -> dict | None:
    """공통 HTTP 호출. 실패/오류 시 None(예외 절대 전파 안 함)."""
    key = os.getenv("KMA_SERVICE_KEY")
    if not key:
        return None
    try:
        q = {"serviceKey": key, "pageNo": 1, "dataType": "JSON", **params}
        r = requests.get(f"{_BASE_URL}/{endpoint}", params=q, timeout=_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        result_code = body["response"]["header"]["resultCode"]
        if result_code != "00":
            _log.warning("KMA %s resultCode=%s", endpoint, result_code)
            return None
        items = _normalize_items(body["response"]["body"]["items"]["item"])
        return {"items": items}
    except Exception as e:
        # 예외 문자열에 요청 URL(serviceKey 포함)이 섞일 수 있어 타입만 기록(notify.py와 동일 원칙)
        _log.warning("KMA %s 호출 실패: %s", endpoint, type(e).__name__)
        return None


def get_current(lat: float | None = None, lon: float | None = None) -> dict:
    """초단기실황 — 기온(℃)·습도(%)·강수(mm). 실패 시 unavailable."""
    if not os.getenv("KMA_SERVICE_KEY"):                  # 미설정·빈 문자열 모두 미설정 취급
        return {"unavailable": True, "reason": "KMA_SERVICE_KEY 미설정"}

    if lat is None or lon is None:
        lat, lon = _farm_location()
    nx, ny = _to_grid(lat, lon)

    cached = _cache_get("getUltraSrtNcst", nx, ny)
    if cached is not None:
        return cached

    base_date, base_time = _ultra_srt_base()
    resp = _request("getUltraSrtNcst", {
        "numOfRows": 10, "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny,
    })
    if resp is None:
        return {"unavailable": True, "reason": "KMA API 호출 실패 또는 응답 오류"}

    values = {}
    for item in resp["items"]:
        values[item.get("category")] = item.get("obsrValue")

    try:
        result = {
            "unavailable": False,
            "temp": float(values["T1H"]) if "T1H" in values else None,
            "humidity": float(values["REH"]) if "REH" in values else None,
            "rain": float(values["RN1"]) if "RN1" in values else None,
            "base_date": base_date, "base_time": base_time,
        }
    except (TypeError, ValueError) as e:
        return {"unavailable": True, "reason": f"응답 파싱 실패: {e}"}

    _cache_put("getUltraSrtNcst", nx, ny, result)
    return result


def get_forecast_3d(lat: float | None = None, lon: float | None = None) -> dict:
    """단기예보 3일 — 날짜별 최저/최고기온 + 시간별 기온·습도·강수확률·하늘상태. 실패 시 unavailable."""
    if not os.getenv("KMA_SERVICE_KEY"):                  # 미설정·빈 문자열 모두 미설정 취급
        return {"unavailable": True, "reason": "KMA_SERVICE_KEY 미설정"}

    if lat is None or lon is None:
        lat, lon = _farm_location()
    nx, ny = _to_grid(lat, lon)

    cached = _cache_get("getVilageFcst", nx, ny)
    if cached is not None:
        return cached

    base_date, base_time = _vilage_fcst_base()
    resp = _request("getVilageFcst", {
        "numOfRows": 1000, "base_date": base_date, "base_time": base_time,
        "nx": nx, "ny": ny,
    })
    if resp is None:
        return {"unavailable": True, "reason": "KMA API 호출 실패 또는 응답 오류"}

    daily: dict[str, dict] = {}
    hourly: list[dict] = []
    hourly_by_key: dict[tuple[str, str], dict] = {}

    for item in resp["items"]:
        category = item.get("category")
        fdate = item.get("fcstDate")
        ftime = item.get("fcstTime")
        fval = item.get("fcstValue")
        if fdate is None or ftime is None:
            continue

        if category in ("TMN", "TMX"):
            d = daily.setdefault(fdate, {})
            d[category] = fval

        if category in ("TMP", "REH", "POP", "SKY"):
            k = (fdate, ftime)
            h = hourly_by_key.get(k)
            if h is None:
                h = {"date": fdate, "time": ftime}
                hourly_by_key[k] = h
                hourly.append(h)
            h[category] = fval

    try:
        daily_out = []
        for d in sorted(daily.keys()):
            v = daily[d]
            daily_out.append({
                "date": d,
                "tmn": float(v["TMN"]) if "TMN" in v else None,
                "tmx": float(v["TMX"]) if "TMX" in v else None,
            })

        hourly.sort(key=lambda h: (h["date"], h["time"]))
        hourly_out = [{
            "date": h["date"], "time": h["time"],
            "temp": float(h["TMP"]) if "TMP" in h else None,
            "humidity": float(h["REH"]) if "REH" in h else None,
            "pop": float(h["POP"]) if "POP" in h else None,
            "sky": h.get("SKY"),
        } for h in hourly]
    except (TypeError, ValueError) as e:
        return {"unavailable": True, "reason": f"응답 파싱 실패: {e}"}

    result = {
        "unavailable": False,
        "daily": daily_out,
        "hourly": hourly_out,
        "base_date": base_date, "base_time": base_time,
    }
    _cache_put("getVilageFcst", nx, ny, result)
    return result
