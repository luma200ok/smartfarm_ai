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
