"""src/llm/monitor.py — 규칙 위험판정 + 중복방지(evaluate) + embed."""
from llm import monitor


def _r(hum, temp):
    return {"습도내부_평균": hum, "온도내부_평균": temp}


def test_assess_high_humidity_is_leafmold_warning():
    a = monitor.assess(_r(92, 25))
    assert len(a) == 1 and a[0]["key"] == "humidity"
    assert a[0]["level"] == "경고" and a[0]["disease"] == "leaf_mold"


def test_assess_moderate_humidity_is_caution():
    assert monitor.assess(_r(86, 25))[0]["level"] == "주의"


def test_assess_hot_and_cold():
    assert any(x["key"] == "temp_hot" for x in monitor.assess(_r(60, 37)))
    assert any(x["key"] == "temp_cold" for x in monitor.assess(_r(60, 3)))


def test_assess_normal_is_empty():
    assert monitor.assess(_r(60, 25)) == []


def test_assess_multiple_risks():
    assert {x["key"] for x in monitor.assess(_r(95, 37))} == {"humidity", "temp_hot"}


def test_evaluate_dedup_and_reset():
    danger = _r(95, 25)
    to_send, active = monitor.evaluate(danger, set())
    assert len(to_send) == 1 and active == {"humidity:경고"}       # 첫 진입 → 발송

    to_send2, active2 = monitor.evaluate(danger, active)
    assert to_send2 == [] and active2 == {"humidity:경고"}         # 지속 → 재발송 없음

    to_send3, active3 = monitor.evaluate(_r(60, 25), active2)
    assert to_send3 == [] and active3 == set()                    # 해소

    to_send4, _ = monitor.evaluate(danger, active3)
    assert len(to_send4) == 1                                     # 재발생 → 재발송


def test_evaluate_resends_on_escalation():
    """주의 → 경고로 악화되면 재발송(P1)."""
    to_send, active = monitor.evaluate(_r(87, 25), set())
    assert to_send[0]["level"] == "주의" and active == {"humidity:주의"}
    to_send2, active2 = monitor.evaluate(_r(96, 25), active)
    assert len(to_send2) == 1 and to_send2[0]["level"] == "경고"
    assert active2 == {"humidity:경고"}


def test_run_single_pass_counts_unique_days_and_dedups(monkeypatch):
    readings = [_r(60, 25), _r(95, 25), _r(95, 25), _r(60, 25)]   # 위험 1회 진입·지속

    class FakeVS:
        def __init__(self, year=None):
            self.series = list(range(len(readings) + monitor.infer.WINDOW - 1))
            self.farm = ("테스트",)
            self.i = 0

        def reading(self):
            return readings[self.i]

        def date(self):
            return f"d{self.i}"

        def window(self):
            return None

        def tick(self):
            self.i += 1

    monkeypatch.setattr(monitor, "VirtualSensor", FakeVS)
    sent = {"n": 0}
    monkeypatch.setattr(monitor.notify, "send_discord",
                        lambda e: (sent.__setitem__("n", sent["n"] + 1) or (True, "ok")))
    steps = monitor.run(interval=0)
    assert steps == len(readings)        # WINDOW 보정된 고유 일수만 순회(래핑 반복 없음)
    assert sent["n"] == 1                # 위험 1회 진입 → 1건(지속 재발송 없음)


def test_embed_has_color_and_fields():
    a = {"key": "humidity", "level": "경고", "disease": "leaf_mold", "reason": "고습 위험"}
    e = monitor._embed(a, _r(95.0, 25.0), "2024-01-01")
    assert e["color"] == 15158332 and e["title"].startswith("🚨")
    assert {"일시", "레벨", "측정", "사유", "관련 병해"} <= {f["name"] for f in e["fields"]}


def test_humidity_alert_carries_disease_list_p2_absorption():
    a = monitor.assess(_r(92, 25))[0]
    assert a["diseases"] == ["leaf_mold", "late_blight"]
    assert a["disease"] == "leaf_mold"          # 대표값 하위호환 유지


def test_embed_shows_disease_list_when_present():
    a = {"key": "humidity", "level": "경고", "disease": "leaf_mold",
         "diseases": ["leaf_mold", "late_blight"], "reason": "고습 위험"}
    e = monitor._embed(a, _r(95.0, 25.0), "2024-01-01")
    field = next(f for f in e["fields"] if f["name"] == "관련 병해")
    assert "잎곰팡이병" in field["value"] and "잎마름역병" in field["value"]


def _expect(mean, sigma_mean=1.4, low=None, sigma_low=1.9):
    d = {"평균": mean, "resid_sigma": {"평균": sigma_mean, "최저": sigma_low}}
    if low is not None:
        d["최저"] = low
    return d


def test_temp_cold_cause_outer_when_residual_small():
    """실측이 기대값 근처(|z|<2) → 외기 요인(모델도 이미 낮게 기대)."""
    a = monitor.assess(_r(60, 4.0), expect=_expect(mean=4.5, sigma_mean=1.0))[0]
    assert a["key"] == "temp_cold"
    assert a["cause"] == "외기 요인(한파로 설비 능력 한계)"


def test_temp_cold_cause_equipment_when_residual_large_negative():
    """실측이 기대값보다 훨씬 낮음(z<=-2) → 설비(난방) 고장 의심."""
    a = monitor.assess(_r(60, 2.0), expect=_expect(mean=10.0, sigma_mean=1.0))[0]
    assert a["key"] == "temp_cold"
    assert a["cause"] == "설비(난방) 고장 의심"


def test_equip_anom_fires_before_threshold_reached():
    """온도가 아직 TEMP_COLD 임계 미달이어도 잔차가 -2σ 이하면 equip_anom."""
    alerts = monitor.assess(_r(60, 8.0), expect=_expect(mean=10.0, sigma_mean=1.0))
    assert any(a["key"] == "equip_anom" and a["level"] == "주의" for a in alerts)


def test_equip_anom_critical_level():
    alerts = monitor.assess(_r(60, 8.0), expect=_expect(mean=13.0, sigma_mean=1.0))
    assert any(a["key"] == "equip_anom" and a["level"] == "경고" for a in alerts)


def test_equip_anom_suppressed_when_temp_alert_already_active():
    """temp_cold/temp_hot 임계를 이미 넘었으면 equip_anom은 별도 발송 생략(cause가 원인 담당)."""
    alerts = monitor.assess(_r(60, 2.0), expect=_expect(mean=10.0, sigma_mean=1.0))
    keys = {a["key"] for a in alerts}
    assert "temp_cold" in keys and "equip_anom" not in keys


def test_equip_anom_not_triggered_when_residual_normal():
    alerts = monitor.assess(_r(60, 20.0), expect=_expect(mean=20.5, sigma_mean=1.0))
    assert alerts == []


def test_assess_backward_compatible_without_expect():
    """expect 인자를 생략하면 기존 동작(cause 없음) 그대로."""
    a = monitor.assess(_r(60, 3.0))[0]
    assert "cause" not in a and a["key"] == "temp_cold"


def test_feedforward_alerts_dedup_by_date():
    model = {
        "doy_solar_climatology": {15: 5.0},
        "models": {"평균": _FakeReg(6.0), "최저": _FakeReg(3.5)},
        "features": ["온도외부_평균", "일사량_평균", "doy_sin", "doy_cos"],
        "resid_sigma": {"평균": 1.4, "최저": 1.9},
    }
    daily = [{"date": "2024-01-15", "tmn": -8.0, "tmx": -2.0}]
    alerts = monitor.feedforward_alerts(daily, expect_model=model)
    assert len(alerts) == 1 and alerts[0]["key"] == "forecast_cold:2024-01-15"

    active = {monitor._akey(alerts[0])}
    alerts2 = monitor.feedforward_alerts(daily, expect_model=model, active=active)
    assert alerts2 == []                       # 같은 날짜는 하루 1회만


def test_feedforward_alerts_empty_when_model_missing(monkeypatch):
    monkeypatch.setattr(monitor.expect_module, "load_model", lambda: None)
    out = monitor.feedforward_alerts([{"date": "2024-01-15", "tmn": -8.0, "tmx": -2.0}], expect_model=None)
    assert out == []


class _FakeReg:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        import numpy as np
        return np.array([self.value] * len(X))
