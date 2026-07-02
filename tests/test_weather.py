"""src/llm/weather.py — 기상청(KMA) API 클라이언트 (requests 모킹, 예외 전파 없음)."""
from unittest.mock import MagicMock, patch

from llm import weather


def _ncst_response():
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {"items": {"item": [
                {"category": "T1H", "obsrValue": "23.5"},
                {"category": "REH", "obsrValue": "60"},
                {"category": "RN1", "obsrValue": "0"},
            ]}},
        }
    }


def _fcst_response():
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {"items": {"item": [
                {"category": "TMN", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "15"},
                {"category": "TMX", "fcstDate": "20260703", "fcstTime": "1500", "fcstValue": "27"},
                {"category": "TMP", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "16"},
                {"category": "REH", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "70"},
                {"category": "POP", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "20"},
                {"category": "SKY", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "1"},
            ]}},
        }
    }


def _error_response(code="03"):
    return {"response": {"header": {"resultCode": code, "resultMsg": "SERVICE KEY IS NOT REGISTERED"},
                          "body": {}}}


def setup_function(_):
    weather._CACHE.clear()
    weather._FAIL_CACHE.clear()


def test_to_grid_seoul():
    assert weather._to_grid(37.5665, 126.9780) == (60, 127)


def test_get_current_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
    r = weather.get_current()
    assert r["unavailable"] is True


def test_get_forecast_3d_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
    r = weather.get_forecast_3d()
    assert r["unavailable"] is True


def test_get_current_parses_response(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _ncst_response())) as m:
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["temp"] == 23.5
    assert r["humidity"] == 60.0
    assert r["rain"] == 0.0
    m.assert_called_once()
    assert m.call_args.args[0].startswith("https://")     # 평문 HTTP 금지(P2)


def test_single_item_dict_is_normalized(monkeypatch):
    """공공데이터포털은 결과 1건이면 item이 list 아닌 dict — 예외 없이 파싱돼야 함(P1)."""
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    body = _ncst_response()
    body["response"]["body"]["items"]["item"] = {"category": "T1H", "obsrValue": "9.9"}
    with patch("requests.get", return_value=MagicMock(status_code=200, json=lambda: body)):
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["temp"] == 9.9
    assert r["humidity"] is None                          # 나머지 카테고리는 누락 → None


def test_forecast_single_item_dict_is_normalized(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    body = _fcst_response()
    body["response"]["body"]["items"]["item"] = {
        "category": "TMP", "fcstDate": "20260703", "fcstTime": "0600", "fcstValue": "16"}
    with patch("requests.get", return_value=MagicMock(status_code=200, json=lambda: body)):
        r = weather.get_forecast_3d(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["hourly"][0]["temp"] == 16.0


def test_empty_string_key_is_unavailable(monkeypatch):
    """빈 문자열 키도 미설정 취급(P3) — API 호출 없이 unavailable."""
    monkeypatch.setenv("KMA_SERVICE_KEY", "")
    with patch("requests.get") as m:
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is True and "미설정" in r["reason"]
    m.assert_not_called()


def test_request_failure_log_hides_service_key(monkeypatch, caplog):
    """예외 로그에 serviceKey가 노출되면 안 됨(P1) — 예외 타입만 기록."""
    import logging
    monkeypatch.setenv("KMA_SERVICE_KEY", "SECRET_KEY_abcdefg")
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    err = Exception("Max retries: https://apis.data.go.kr/...?serviceKey=SECRET_KEY_abcdefg")
    with caplog.at_level(logging.WARNING, logger="llm.weather"):
        with patch("requests.get", side_effect=err):
            r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is True
    assert "SECRET_KEY_abcdefg" not in caplog.text


def test_ultra_srt_base_boundaries():
    """실황 base_time 경계 — 매시 40분 전후·자정 경계(P2)."""
    from datetime import datetime
    # 40분 직전 → 직전 정시-1h
    assert weather._ultra_srt_base(datetime(2026, 7, 2, 14, 39)) == ("20260702", "1300")
    # 40분 직후 → 그 시각 정시
    assert weather._ultra_srt_base(datetime(2026, 7, 2, 14, 40)) == ("20260702", "1400")
    # 자정 직후(00:05) → 전날 23시
    assert weather._ultra_srt_base(datetime(2026, 7, 2, 0, 5)) == ("20260701", "2300")


def test_vilage_fcst_base_boundaries():
    """단기예보 base_time 경계 — 발표시각+10분 전후·02:10 이전(P2)."""
    from datetime import datetime
    # 02:10 직전(당일 첫 발표 전) → 전날 23시
    assert weather._vilage_fcst_base(datetime(2026, 7, 2, 2, 9)) == ("20260701", "2300")
    # 02:10 정각 → 당일 02시
    assert weather._vilage_fcst_base(datetime(2026, 7, 2, 2, 10)) == ("20260702", "0200")
    # 05:10 직전 → 여전히 02시
    assert weather._vilage_fcst_base(datetime(2026, 7, 2, 5, 9)) == ("20260702", "0200")
    # 23:10 이후 → 당일 23시
    assert weather._vilage_fcst_base(datetime(2026, 7, 2, 23, 30)) == ("20260702", "2300")


def test_clear_cache_forces_refetch(monkeypatch):
    """clear_cache() 공개 API — 캐시 비운 뒤 재호출 시 API 재요청(P2)."""
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _ncst_response())) as m:
        weather.get_current(37.5665, 126.9780)
        weather.clear_cache()
        weather.get_current(37.5665, 126.9780)
    assert m.call_count == 2


def test_get_forecast_3d_parses_response(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _fcst_response())):
        r = weather.get_forecast_3d(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["daily"][0]["tmn"] == 15.0
    assert r["daily"][0]["tmx"] == 27.0
    assert r["hourly"][0]["temp"] == 16.0
    assert r["hourly"][0]["humidity"] == 70.0
    assert r["hourly"][0]["pop"] == 20.0
    assert r["hourly"][0]["sky"] == "1"


def test_result_code_error_is_unavailable(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _error_response())):
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is True


def test_http_exception_is_graceful(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    with patch("requests.get", side_effect=Exception("network down")):
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is True


def test_cache_avoids_second_call(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _ncst_response())) as m:
        weather.get_current(37.5665, 126.9780)
        weather.get_current(37.5665, 126.9780)
    assert m.call_count == 1


def test_forecast_cache_avoids_second_call(monkeypatch):
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    with patch("requests.get", return_value=MagicMock(
            status_code=200, json=lambda: _fcst_response())) as m:
        weather.get_forecast_3d(37.5665, 126.9780)
        weather.get_forecast_3d(37.5665, 126.9780)
    assert m.call_count == 1


# ── 이슈 #6 후속 — 재시도 1회 + 실패 단기 캐시(negative cache) ──────────────

def test_retry_once_then_success(monkeypatch):
    """1차 호출 실패(예외) → 1.5초 대기 후 재시도해 성공하면 정상 반환, requests.get 2회 호출."""
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    responses = [Exception("timeout"), MagicMock(status_code=200, json=lambda: _ncst_response())]
    with patch("requests.get", side_effect=responses) as m:
        r = weather.get_current(37.5665, 126.9780)
    assert r["unavailable"] is False
    assert r["temp"] == 23.5
    assert m.call_count == 2


def test_retry_fails_then_negative_cache_short_circuits(monkeypatch):
    """재시도도 실패하면 unavailable. 60초 내 재요청은 requests.get 미호출로 즉시 unavailable."""
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    with patch("requests.get", side_effect=Exception("timeout")) as m:
        r1 = weather.get_current(37.5665, 126.9780)
        assert r1["unavailable"] is True
        assert m.call_count == 2                          # 최초 시도 + 재시도 1회

        r2 = weather.get_current(37.5665, 126.9780)        # 실패 캐시 TTL(60s) 내 재요청
        assert r2["unavailable"] is True
        assert m.call_count == 2                           # 추가 호출 없음(negative cache hit)


def test_clear_cache_also_clears_negative_cache(monkeypatch):
    """clear_cache()는 실패 캐시도 비워 다음 요청은 즉시 재시도되어야 함."""
    monkeypatch.setenv("KMA_SERVICE_KEY", "dummy-key")
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    with patch("requests.get", side_effect=Exception("timeout")) as m:
        r1 = weather.get_current(37.5665, 126.9780)
        assert r1["unavailable"] is True
        assert m.call_count == 2

        weather.clear_cache()

        r2 = weather.get_current(37.5665, 126.9780)
        assert r2["unavailable"] is True
        assert m.call_count == 4                          # 실패 캐시 비워져 재호출(2회 더)
