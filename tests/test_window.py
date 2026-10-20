import io
import json
import os
import queue
import threading
import time

import pytest

from backend import main
from backend.main import Status, screen_feed, sound_level
from backend.ui import tray, window
from backend.ui.window import ScreenLink, pump, to_js

import numpy as np


# ---- only a fixed set of screen calls, as JSON ----------------------------------------------

def test_allowed_calls_become_json_arguments():
    js = to_js({"call": "addMessage", "args": ["you", "open notepad", {"private": False}]})
    assert js == 'window.HUD && window.HUD.addMessage(...["you", "open notepad", {"private": false}])'


@pytest.mark.parametrize("message", [
    {"call": "eval", "args": ["alert(1)"]},
    {"call": "attach"},                                  # only the page itself may attach
    {"call": "setState", "args": ["a", "b", "c"]},       # too many arguments
    {"call": "setState", "args": "listening"},           # not a list
    {"call": "setLevel", "args": [float("nan")]},
    {"args": ["x"]},
    ["setState"],
    "setState",
    None,
])
def test_anything_else_is_dropped(message):
    assert to_js(message) is None


def test_tricky_text_stays_inert_text():
    js = to_js({"call": "addMessage", "args": ["you", "\");alert(1);//</script>  ", {}]})
    assert "\\u2028" in js and "\\u2029" in js and '\\")' in js       # escaped inside one JSON string
    assert js.count("alert") == 1 and js.startswith("window.HUD && window.HUD.addMessage(...[")


class FakeWindow:
    def __init__(self):
        self.js, self.restored, self.destroyed = [], 0, False

    def evaluate_js(self, js):
        self.js.append(js)

    def restore(self):
        self.restored += 1

    def show(self):
        pass

    def destroy(self):
        self.destroyed = True


def test_pump_applies_updates_and_closes_when_the_pipe_ends():
    w, ready = FakeWindow(), threading.Event()
    ready.set()
    lines = [json.dumps({"call": "setState", "args": ["listening", ""]}) + "\n",
             "not json\n",
             json.dumps({"call": "eval", "args": ["x"]}) + "\n",
             "x" * (window.MAX_LINE_CHARS + 1) + "\n",
             json.dumps({"call": "focus"}) + "\n"]
    pump(lines, w, ready)
    assert w.js == ['window.HUD && window.HUD.setState(...["listening", ""])']
    assert w.restored == 1 and w.destroyed


def test_the_page_can_only_ask_to_pause():
    out = io.StringIO()
    api = window._Api(out, threading.Lock())
    api.toggle_pause()
    assert json.loads(out.getvalue()) == {"cmd": "toggle_pause"}
    public = [n for n in dir(api) if not n.startswith("_")]
    assert public == ["toggle_pause"]


def test_page_is_a_local_file():
    assert window.PAGE.is_file() and window.PAGE.as_uri().startswith("file:///")
    html = window.PAGE.read_text(encoding="utf-8")
    assert "default-src 'none'" in html and "connect-src 'none'" in html and "<script>" not in html
    js = (window.PAGE.parent / "screen.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "eval(" not in js and "fetch(" not in js


# ---- the app's side ------------------------------------------------------------------------

class FakeProc:
    """Behaves like the window process: we read what the app writes, and can write back."""

    def __init__(self):
        self.written = []
        self.back = queue.Queue()
        self.alive = True
        proc = self

        class In:
            def write(self, text):
                proc.written.append(json.loads(text))

            def flush(self):
                pass

            def close(self):
                proc.alive = False
                proc.back.put(None)

        self.stdin = In()

    @property
    def stdout(self):
        while True:
            line = self.back.get()
            if line is None:
                return
            yield line

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False

    def reply(self, message):
        self.back.put(json.dumps(message) + "\n" if not isinstance(message, str) else message)


def wait_for(cond, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def link():
    procs, commands = [], []

    def launcher():
        procs.append(FakeProc())
        return procs[-1]

    s = ScreenLink(on_command=commands.append, get_systems=lambda: [{"name": "Microphone", "ok": True}],
                   get_timers=lambda: [{"label": "tea", "remaining": 30, "total": 60}], launcher=launcher)
    yield s, procs, commands
    s.close()


def calls(proc):
    return [m.get("call") for m in proc.written]


def test_nothing_is_sent_while_closed(link):
    s, procs, _ = link
    s.add_message("you", "hello")
    s.set_level(0.5)
    assert procs == [] and not s.is_open


def test_opening_sends_a_snapshot_once_the_page_is_ready(link):
    s, procs, _ = link
    s.set_state("listening")
    s.add_message("you", "open notepad")
    s.add_message("jarvis", "dear diary", private=True)
    s.add_activity("wake · hey jarvis 0.91")
    assert s.open()
    proc = procs[0]
    time.sleep(0.05)
    assert proc.written == []                               # nothing until the page says it is ready
    proc.reply({"event": "ready"})
    assert wait_for(lambda: len(proc.written) >= 6)
    assert calls(proc)[:3] == ["setState", "setSystems", "setTimers"]
    msgs = [m["args"] for m in proc.written if m["call"] == "addMessage"]
    assert msgs == [["you", "open notepad", {"private": False}], ["jarvis", "dear diary", {"private": True}]]


def test_second_open_brings_the_window_forward(link):
    s, procs, _ = link
    s.open()
    s.open()
    assert len(procs) == 1 and procs[0].written[-1] == {"call": "focus"}


def test_only_the_pause_command_is_accepted(link):
    s, procs, commands = link
    s.open()
    proc = procs[0]
    for message in ({"cmd": "toggle_pause"}, {"cmd": "run_tool", "tool": "delete_file"}, {"cmd": "quit"},
                    "garbage\n", ["toggle_pause"]):
        proc.reply(message)
    proc.reply({"cmd": "toggle_pause"})
    assert wait_for(lambda: len(commands) == 2)
    time.sleep(0.05)
    assert commands == ["toggle_pause", "toggle_pause"]


def test_sound_level_updates_are_rate_limited(link):
    s, procs, _ = link
    s.open()
    procs[0].reply({"event": "ready"})
    assert wait_for(lambda: "setTimers" in calls(procs[0]))
    for _ in range(50):
        s.set_level(0.7)
    assert calls(procs[0]).count("setLevel") == 1


def test_a_closed_window_can_be_opened_again(link):
    s, procs, _ = link
    s.open()
    procs[0].stdin.close()
    assert wait_for(lambda: not s.is_open)
    s.open()
    assert len(procs) == 2


def test_a_failed_launch_is_reported():
    def broken():
        raise OSError("no pythonw")

    assert ScreenLink(launcher=broken).open() is False


# ---- what the loop tells the screen ---------------------------------------------------------

class Recorder:
    def __init__(self):
        self.seen = []

    def __getattr__(self, name):
        return lambda *a, **k: self.seen.append((name, a, k))


def test_feed_turns_a_command_into_screen_updates():
    r = Recorder()
    on_status, on_event = screen_feed(r)
    on_status(Status.WORKING, "")
    on_event("wake", {"source": "hey jarvis", "score": 0.912})
    on_event("hearing", {"follow_up": False})
    on_event("heard", {"text": "read my note"})
    on_event("thinking", {})
    on_event("reply", {"text": "dear diary", "ok": True, "private": True})
    on_event("speaking", {})
    assert r.seen == [
        ("set_state", ("working", ""), {}),
        ("add_activity", ("wake · hey jarvis 0.91",), {}),
        ("set_state", ("hearing", "go ahead"), {}),
        ("add_message", ("you", "read my note"), {}),
        ("set_state", ("working", "thinking"), {}),
        ("add_message", ("jarvis", "dear diary"), {"private": True}),
        ("add_activity", ("done", "ok"), {}),
        ("refresh_timers", (), {}),
        ("set_state", ("speaking",), {}),
    ]


def test_feed_reports_pauses_problems_and_timers():
    r = Recorder()
    on_status, on_event = screen_feed(r)
    on_status(Status.LOCKED, "")
    on_status(Status.PROBLEM, "I couldn't open the microphone.")
    on_event("timer", {"label": "tea"})
    on_event("nothing", {"reason": "no speech"})
    assert ("add_activity", ("Windows locked: microphone off", "warn"), {}) in r.seen
    assert ("add_activity", ("I couldn't open the microphone.", "bad"), {}) in r.seen
    assert ("add_message", ("system", "Timer done: tea"), {}) in r.seen
    assert ("add_activity", ("heard nothing", "warn"), {}) in r.seen


def test_loop_emits_events_in_order_and_meters_the_microphone():
    from tests.test_main import FakeFlow, FakeMics, FakeSpeech, FakeTriggers, FakeVoice
    from backend.voice.listen import Heard

    events, levels = [], []
    trig = FakeTriggers()
    a = main.Assistant(trig, FakeSpeech(Heard(text="open notepad", reason="done")), lambda: FakeFlow(),
                       FakeVoice(trig), mic_factory=FakeMics(), chime=lambda: None,
                       on_event=lambda k, d: events.append(k), on_level=levels.append)
    a.start()
    try:
        assert wait_for(lambda: a.status is Status.LISTENING)
        trig.fire.set()
        assert wait_for(lambda: "spoke" in events)
    finally:
        a.stop()
    assert events[:7] == ["wake", "hearing", "heard", "thinking", "reply", "speaking", "spoke"]
    assert levels and all(0 <= x <= 1 for x in levels)


@pytest.mark.parametrize("rms, low, high", [(0, 0, 0.001), (1000, 0.5, 0.75), (20000, 1, 1)])
def test_sound_level_scale(rms, low, high):
    chunk = np.full(1280, rms, dtype=np.int16) if rms < 32768 else None
    assert low <= sound_level(chunk) <= high


def test_timer_snapshot(monkeypatch):
    from backend.tools.timers import TimerService

    now = {"t": 100.0}
    s = TimerService(clock=lambda: now["t"], start=False)
    s.add(60, "tea")
    now["t"] += 15
    assert s.snapshot() == [{"label": "tea", "remaining": 45, "total": 60}]


# ---- tray: left-click opens the screen --------------------------------------------------------

def test_tray_menu_offers_open_first():
    t = tray.Tray(on_open=lambda: None)
    assert t.menu_labels() == ["Open J.A.R.V.I.S", "Pause listening", "Quit J.A.R.V.I.S"]


def test_tray_open_command_and_left_click():
    opened = []
    t = tray.Tray(on_open=lambda: opened.append(1))
    t.handle_command(t.CMD_OPEN)
    t._wndproc(0, t.WM_TRAY, 0, tray.WM_LBUTTONUP)
    assert opened == [1, 1]


# ---- the real window (opens on screen) ------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("JARVIS_GUI_TESTS") != "1", reason="set JARVIS_GUI_TESTS=1 to open the real screen")
def test_real_screen_opens_and_takes_updates():
    commands = []
    s = ScreenLink(on_command=commands.append, get_systems=lambda: [{"name": "Microphone", "ok": True}],
                   get_timers=lambda: [])
    assert s.open()
    try:
        assert wait_for(lambda: s._ready, timeout=30)
        s.set_state("speaking")
        s.add_message("jarvis", "Real window test, sir.")
        time.sleep(2)
        assert s.is_open
    finally:
        s.close()
    assert wait_for(lambda: not s.is_open, timeout=10)
