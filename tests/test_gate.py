import json

import pytest

from backend.gate.gate import Gate
from backend.gate.models import Param, Risk, Tool, ToolError


def mk(risk=Risk.LOW, fn=None):
    return Tool("t", "test tool", risk, {"x": Param(str, max_len=10)}, fn or (lambda x: f"got {x}"))


def events(audit):
    return [json.loads(line)["event"] for line in audit.path.read_text().splitlines()]


def test_low_risk_runs_and_is_logged(audit):
    r = Gate([mk()], audit).execute("t", {"x": "hi"})
    assert (r.ok, r.message) == (True, "got hi")
    assert events(audit) == ["proposed", "ran"] and audit.verify()[0]


@pytest.mark.parametrize("name", ["nope", "T", " t", "t ", None, 5, ["t"], ""])
def test_unknown_or_odd_tool_names_denied(audit, name):
    r = Gate([mk()], audit).execute(name, {"x": "hi"})
    assert not r.ok and "denied" in events(audit)


def test_invalid_args_denied_and_tool_not_called(audit):
    called = []
    g = Gate([mk(fn=lambda x: called.append(x))], audit)
    assert not g.execute("t", {"x": "way too long for this"}).ok
    assert not g.execute("t", {"x": "hi", "extra": 1}).ok
    assert not g.execute("t", "x=hi").ok
    assert called == []


def test_high_risk_denied_by_default(audit):
    called = []
    g = Gate([mk(Risk.HIGH, lambda x: called.append(x))], audit)
    assert not g.execute("t", {"x": "hi"}).ok and called == []


def test_high_risk_runs_only_with_explicit_true(audit):
    for answer, expect in [(True, True), (False, False), ("yes", False), (1, False), (None, False)]:
        g = Gate([mk(Risk.HIGH)], audit, confirm=lambda n, a, ans=answer: ans)
        assert g.execute("t", {"x": "hi"}).ok is expect


def test_confirmation_error_means_deny(audit):
    def boom(n, a):
        raise RuntimeError("popup crashed")

    assert not Gate([mk(Risk.HIGH)], audit, confirm=boom).execute("t", {"x": "hi"}).ok


def test_tool_toolerror_message_passes_through(audit):
    def fn(x):
        raise ToolError("No such note.")

    r = Gate([mk(fn=fn)], audit).execute("t", {"x": "hi"})
    assert (r.ok, r.message) == (False, "No such note.")


def test_unexpected_exception_hides_details(audit):
    def fn(x):
        raise RuntimeError("secret path C:\\Users\\me\\.ssh")

    r = Gate([mk(fn=fn)], audit).execute("t", {"x": "hi"})
    assert not r.ok and "ssh" not in r.message
    assert "ssh" not in audit.path.read_text()


def test_rate_limit_and_recovery(audit):
    now = [0.0]
    g = Gate([mk()], audit, max_calls=3, window_seconds=60, clock=lambda: now[0])
    assert all(g.execute("t", {"x": "a"}).ok for _ in range(3))
    assert not g.execute("t", {"x": "a"}).ok
    now[0] = 61.0
    assert g.execute("t", {"x": "a"}).ok


def test_tool_specs_never_expose_callables(audit):
    specs = Gate([mk()], audit).tool_specs()
    assert specs[0]["name"] == "t" and "fn" not in specs[0]
    assert not any(callable(v) for v in specs[0].values())