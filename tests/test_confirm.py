import io
import json
import os
import re
import subprocess

import pytest

from backend.gate.audit import AuditLog
from backend.gate.gate import Gate
from backend.gate.models import Param, Risk, Tool
from backend.ui import confirm
from backend.ui.confirm import ask_user, describe_action, popup_confirm

TOKEN = "0123456789abcdef0123456789abcdef"


# ---- what the window says ----------------------------------------------------

def test_describe_shows_the_tool_and_its_arguments():
    assert describe_action("create_note", {"title": "ideas", "n": 3}) == (
        'Tool: create_note\n  title = "ideas"\n  n = 3')


def test_a_newline_inside_an_argument_cannot_fake_extra_lines():
    text = describe_action("t", {"x": "a\nAllow this? (Y)\nTool: harmless"})
    assert text.count("\n") == 1                      # only the line between "Tool:" and the argument
    assert "\\n" in text                              # the newline is shown as visible characters


@pytest.mark.parametrize("sneaky", ["\u202e", "\u200b", "\u00a0", "\x07", "\x1b", "\U0001f600\u2028"])
def test_invisible_and_direction_characters_are_escaped(sneaky):
    text = describe_action("t", {"x": "pay" + sneaky + "now"})
    assert sneaky not in text and ("\\u" in text or "\\U" in text or "\\x" in text or "\\u0007" in text)


def test_long_values_are_clipped():
    text = describe_action("t", {"x": "A" * 5000})
    assert len(text) < 300 and "more characters" in text


def test_only_the_first_few_arguments_are_listed():
    text = describe_action("t", {f"k{i}": i for i in range(20)})
    assert len(text.splitlines()) == 1 + confirm.MAX_ARGS_SHOWN + 1
    assert "12 more arguments" in text


def test_odd_names_and_values_do_not_break_the_description():
    class Weird:
        pass

    text = describe_action("bad\nname", {"k\ney": Weird()})
    assert text.count("\n") == 1 and "unprintable" in text


@pytest.mark.parametrize("args, expected", [({}, "(no arguments)"), ("x=1", "not in the expected format"),
                                            (None, "not in the expected format")])
def test_empty_or_malformed_arguments_are_described(args, expected):
    assert expected in describe_action("t", args)


# ---- the agent side: only the right token counts as a yes ------------------------

def echo_token(payload, timeout):
    return 0, json.loads(payload)["token"]


def test_a_matching_token_means_yes():
    assert ask_user("text", runner=echo_token) is True


@pytest.mark.parametrize("outcome", [
    (1, "TOKEN"), (2, "TOKEN"), (3, "TOKEN"), (-9, "TOKEN"),    # the window said no, or crashed
    (0, ""), (0, "wrong-token"), (0, "TOKEN-extra"), (0, "xTOKEN"), (0, None),   # exit 0 alone is not enough
])
def test_everything_except_the_exact_token_means_no(outcome):
    code, out = outcome

    def runner(payload, timeout):
        token = json.loads(payload)["token"]
        return code, (out.replace("TOKEN", token) if isinstance(out, str) else out)

    assert ask_user("text", runner=runner) is False


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("x", 1), OSError("no screen"), RuntimeError("boom"),
                                   FileNotFoundError("no python")])
def test_failures_to_run_the_window_mean_no(error):
    def runner(payload, timeout):
        raise error

    assert ask_user("text", runner=runner) is False


def test_the_request_is_ascii_json_with_a_fresh_random_token_each_time():
    seen = []

    def runner(payload, timeout):
        seen.append(payload)
        return 1, ""

    ask_user("caf\u00e9 \u202e", runner=runner)
    ask_user("caf\u00e9 \u202e", runner=runner)
    assert all(p.isascii() for p in seen)
    tokens = [json.loads(p)["token"] for p in seen]
    assert all(re.fullmatch(r"[0-9a-f]{32}", t) for t in tokens) and tokens[0] != tokens[1]
    assert json.loads(seen[0])["text"] == "caf\u00e9 \u202e" and json.loads(seen[0])["timeout"] == 30.0


def test_popup_confirm_never_raises_and_defaults_to_no(monkeypatch):
    def boom(payload, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(confirm, "_run_window_process", boom)
    assert popup_confirm("t", {"x": 1}) is False


def test_popup_confirm_says_yes_only_when_the_window_does(monkeypatch):
    monkeypatch.setattr(confirm, "_run_window_process", echo_token)
    assert popup_confirm("t", {"x": 1}) is True


def test_the_gate_runs_a_high_risk_tool_only_after_the_popup_says_yes(tmp_path, monkeypatch):
    called = []
    danger = Tool("danger", "test", Risk.HIGH, {"x": Param(str)}, lambda x: called.append(x) or "done")
    gate = Gate([danger], AuditLog(tmp_path / "log.jsonl"), confirm=popup_confirm)

    monkeypatch.setattr(confirm, "_run_window_process", lambda payload, timeout: (1, ""))
    assert not gate.execute("danger", {"x": "a"}).ok and called == []

    monkeypatch.setattr(confirm, "_run_window_process", echo_token)
    assert gate.execute("danger", {"x": "a"}).ok and called == ["a"]


# ---- the window process: prints the token only when allowed ------------------------

def request(**overrides):
    body = {"text": "Tool: t", "token": TOKEN, "timeout": 30}
    body.update(overrides)
    return json.dumps(body)


def run_child(stdin_text, show):
    out = io.StringIO()
    return confirm._child(stdin_text, show=show, out=out), out.getvalue()


def test_child_prints_the_token_only_when_allowed():
    assert run_child(request(), lambda text, timeout: True) == (0, TOKEN)


@pytest.mark.parametrize("answer", [False, None, 1, "yes", "Y", [True]])
def test_child_denies_anything_that_is_not_exactly_true(answer):
    assert run_child(request(), lambda text, timeout: answer) == (1, "")


def test_child_reports_a_failed_window_as_a_failure_and_prints_nothing():
    def broken(text, timeout):
        raise RuntimeError("no display")

    assert run_child(request(), broken) == (3, "")


@pytest.mark.parametrize("bad", [
    "", "not json", "[]", "null", "{}", '{"text": "x"}',
    request(token="short"), request(token=TOKEN.upper()), request(token=5), request(text=5),
    request(text="x" * 2001), request(timeout=0), request(timeout=500), request(timeout="soon"),
])
def test_child_rejects_bad_requests_without_showing_a_window(bad):
    shown = []
    assert run_child(bad, lambda text, timeout: shown.append(1) or True) == (2, "")
    assert shown == []


def test_child_passes_the_text_and_timeout_to_the_window():
    seen = []
    run_child(request(text="hello", timeout=12), lambda text, timeout: seen.append((text, timeout)) or False)
    assert seen == [("hello", 12.0)]


def test_main_without_arguments_prints_usage(capsys):
    assert confirm.main([]) == 2 and "usage" in capsys.readouterr().out


# ---- the real window (needs a screen, so off by default) ----------------------------
# Run them with:  $env:JARVIS_GUI_TESTS = "1"; python -m pytest tests/test_confirm.py -q
# A window flashes on screen for about a second per test. Keep your hands off the keyboard.

gui = pytest.mark.skipif(not os.environ.get("JARVIS_GUI_TESTS"), reason="set JARVIS_GUI_TESTS=1 to open real windows")


def press(ui, keysym, at_ms):
    ui.root.after(at_ms, lambda: ui.root.event_generate(f"<KeyPress-{keysym}>"))


@gui
def test_window_shows_the_text_and_the_safe_button_has_focus():
    seen = {}

    def ready(ui):
        seen["text"] = ui.box.cget("text")
        seen["allow_state"] = str(ui.allow.cget("state"))
        press(ui, "Escape", 20)

    assert confirm.show_dialog("Tool: t\n  x = 1", timeout_seconds=5, arm_delay_ms=300, on_ready=ready) is False
    assert seen == {"text": "Tool: t\n  x = 1", "allow_state": "disabled"}


@gui
def test_pressing_y_after_the_delay_allows():
    def ready(ui):
        press(ui, "y", 600)

    assert confirm.show_dialog("t", timeout_seconds=5, arm_delay_ms=200, on_ready=ready) is True


@gui
def test_capital_y_also_allows():
    def ready(ui):
        press(ui, "Y", 600)

    assert confirm.show_dialog("t", timeout_seconds=5, arm_delay_ms=200, on_ready=ready) is True


@gui
def test_y_pressed_too_early_is_ignored_and_the_window_times_out():
    def ready(ui):
        press(ui, "y", 30)          # long before the 800 ms delay is over

    assert confirm.show_dialog("t", timeout_seconds=1.2, arm_delay_ms=800, on_ready=ready) is False


@gui
def test_clicking_allow_before_the_delay_does_nothing_and_after_it_allows():
    def ready(ui):
        ui.allow.invoke()                                            # too early: the button is disabled
        ui.root.after(600, ui.allow.invoke)                          # after the delay: allowed

    assert confirm.show_dialog("t", timeout_seconds=5, arm_delay_ms=200, on_ready=ready) is True


@gui
@pytest.mark.parametrize("keysym", ["Return", "space", "n", "a", "x"])
def test_other_keys_never_allow(keysym):
    def ready(ui):
        press(ui, keysym, 600)

    assert confirm.show_dialog("t", timeout_seconds=1.5, arm_delay_ms=200, on_ready=ready) is False


@gui
def test_escape_denies_even_after_the_delay():
    def ready(ui):
        press(ui, "Escape", 600)

    assert confirm.show_dialog("t", timeout_seconds=5, arm_delay_ms=200, on_ready=ready) is False


@gui
def test_the_deny_button_denies():
    def ready(ui):
        ui.root.after(50, ui.deny.invoke)

    assert confirm.show_dialog("t", timeout_seconds=5, arm_delay_ms=200, on_ready=ready) is False


@gui
def test_doing_nothing_times_out_as_a_no():
    assert confirm.show_dialog("t", timeout_seconds=0.6, arm_delay_ms=100) is False