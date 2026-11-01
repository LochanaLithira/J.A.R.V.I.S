import threading
import time

import pytest

from backend.gate.audit import AuditLog
from backend.gate.gate import Gate
from backend.gate.models import ToolError
from backend.tools.timers import MAX_TIMERS, TimerService, describe_seconds, make_timer_tools


@pytest.fixture
def clock():
    return {"now": 1000.0}


@pytest.fixture
def service(clock):
    return TimerService(clock=lambda: clock["now"], start=False)


@pytest.fixture
def gate(service, tmp_path):
    return Gate(make_timer_tools(service), AuditLog(tmp_path / "log.jsonl"), max_calls=1000)


@pytest.mark.parametrize("seconds, text", [(1, "1 second"), (45, "45 seconds"), (300, "5 minutes"),
                                           (3600, "1 hour"), (5430, "1 hour 30 minutes 30 seconds")])
def test_durations_read_naturally(seconds, text):
    assert describe_seconds(seconds) == text


def test_set_timer_and_reminder_without_repeating_the_label(gate):
    assert gate.execute("set_timer", {"seconds": 300, "label": ""}).message == "Timer set for 5 minutes."
    r = gate.execute("set_timer", {"seconds": 600, "label": "take my medicine"})
    assert r.message == "Reminder set for 10 minutes." and "medicine" not in r.message   # spoken out loud


def test_list_timers_is_private_and_sorted(gate, clock):
    gate.execute("set_timer", {"seconds": 600, "label": "tea"})
    gate.execute("set_timer", {"seconds": 60, "label": ""})
    clock["now"] += 30
    r = gate.execute("list_timers", {})
    assert r.private and r.message == "Timers: 30 seconds left; 9 minutes 30 seconds left for tea."


def test_due_timers_fire_in_order_and_once(service, clock):
    service.add(60, "second")
    service.add(30, "first")
    service.add(90, "")
    clock["now"] += 61
    assert service.check() == ["first", "second"]
    assert service.check() == []
    clock["now"] += 60
    assert service.check() == [""]


@pytest.mark.parametrize("label, left", [("tea", ["eggs"]), ("all", []), ("TEA", ["eggs"])])
def test_cancel(service, label, left):
    service.add(60, "tea")
    service.add(60, "eggs")
    service.cancel(label)
    assert [t.label for t in service._timers] == left


def test_cancel_the_only_timer_without_a_name(service):
    service.add(60, "")
    assert service.cancel("") == "Cancelled that timer." and service._timers == []


@pytest.mark.parametrize("label", ["", "coffee"])
def test_unclear_cancel_is_refused(service, label):
    service.add(60, "tea")
    service.add(60, "eggs")
    with pytest.raises(ToolError):
        service.cancel(label)


def test_cancel_with_no_timers(service):
    with pytest.raises(ToolError):
        service.cancel("all")


def test_at_most_ten_timers(service):
    for _ in range(MAX_TIMERS):
        service.add(60, "")
    with pytest.raises(ToolError):
        service.add(60, "")


@pytest.mark.parametrize("args", [{"seconds": 0, "label": ""}, {"seconds": 86401, "label": ""},
                                  {"seconds": True, "label": ""}, {"seconds": "300", "label": ""},
                                  {"seconds": 60, "label": "x" * 61}, {"seconds": 60, "label": "<script>"},
                                  {"seconds": 60}])
def test_bad_timer_details_are_refused_by_the_gate(gate, service, args):
    assert not gate.execute("set_timer", args).ok and service._timers == []


def test_the_background_thread_rings_due_timers(clock):
    rung = []
    done = threading.Event()
    s = TimerService(on_fire=lambda label: (rung.append(label), done.set()), clock=lambda: clock["now"])
    try:
        s.add(5, "tea")
        clock["now"] += 6
        assert done.wait(2) and rung == ["tea"]
    finally:
        s.stop()


def test_a_failing_announcement_does_not_stop_other_timers(clock):
    rung = []
    two = threading.Event()

    def on_fire(label):
        rung.append(label)
        if len(rung) == 2:
            two.set()
        raise RuntimeError("display broke")

    s = TimerService(on_fire=on_fire, clock=lambda: clock["now"])
    try:
        s.add(5, "a")
        s.add(10, "b")
        clock["now"] += 6
        time.sleep(0.5)
        clock["now"] += 6
        assert two.wait(2) and rung == ["a", "b"]
    finally:
        s.stop()
