import ctypes
import logging
import os
import struct
import threading
import time
import uuid

import numpy as np
import pytest

from backend import config, main
from backend.brain import BrainError
from backend.gate.models import Result
from backend.main import Assistant, Status, startup_checks
from backend.ui import confirm, tray
from backend.voice.listen import Heard
from backend.voice.triggers import CHUNK, Microphone, TriggerError, WakeEvent


# ---- fakes ----------------------------------------------------------------------------

class FakeMics:
    """Counts how often the microphone is opened and closed; can fail or go silent on purpose."""

    def __init__(self, fail_opens=0, stall_once=False):
        self.opens = self.closes = 0
        self.fail_opens, self.stall_once = fail_opens, stall_once

    def __call__(self, device):
        return FakeMic(self)

    @property
    def open_now(self):
        return self.opens - self.closes


class FakeMic:
    def __init__(self, owner):
        self.owner = owner
        self.drained = 0

    def __enter__(self):
        self.owner.opens += 1
        if self.owner.fail_opens > 0:
            self.owner.fail_opens -= 1
            self.owner.closes += 1
            raise TriggerError("I couldn't open the microphone.")
        return self

    def __exit__(self, *a):
        self.owner.closes += 1

    def chunks(self, stop=None, idle_timeout=None):
        if self.owner.stall_once:
            self.owner.stall_once = False
            raise TriggerError("The microphone stopped sending sound.")
        while stop is None or not stop.is_set():
            time.sleep(0.002)
            yield np.zeros(CHUNK, np.int16)

    def drain(self):
        self.drained += 1
        return 0


class FakeTriggers:
    _threshold = 0.25

    def __init__(self, crash_once=False):
        self.fire = threading.Event()
        self.crash_once = crash_once
        self.muted_log, self.resets, self.last_score = [], 0, 0.0
        self._muted = False

    def process(self, chunk):
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("secret words that must never be logged")
        if self.fire.is_set():
            self.fire.clear()
            return WakeEvent("hey jarvis", 0.9)
        return None

    def reset(self):
        self.resets += 1

    def set_muted(self, muted):
        self._muted = muted
        self.muted_log.append(muted)


class FakeSpeech:
    """Returns one Heard per call; the last one repeats. Records each call's no-speech wait."""

    def __init__(self, *heard):
        self.heard, self.waits = list(heard), []

    def hear(self, chunks, no_speech_seconds=None):
        self.waits.append(no_speech_seconds)
        return self.heard.pop(0) if len(self.heard) > 1 else self.heard[0]


class FakeFlow:
    """Returns one Result per call; the last one repeats. Records commands and history."""

    def __init__(self, *results, error=None):
        self.results = list(results) or [Result(True, "Opening notepad.")]
        self.error, self.seen, self.histories = error, [], []

    @property
    def result(self):
        return self.results[0]

    def run(self, text, history=()):
        self.seen.append(text)
        self.histories.append(tuple(history))
        if self.error:
            raise self.error
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class FakeVoice:
    def __init__(self, triggers, ok=True):
        self.triggers, self.ok, self.said = triggers, ok, []

    def say(self, result, wait=False):
        self.said.append((result.message, wait, self.triggers._muted))
        return self.ok


def wait_for(cond, timeout=4.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def run_assistant():
    started = []

    def make(triggers=None, heard=Heard(text="open notepad", reason="done"), flow=None, voice="auto", mics=None,
             **kw):
        triggers = triggers or FakeTriggers()
        mics = mics or FakeMics()
        statuses, notices, shown, chimes = [], [], [], []
        if voice == "auto":
            voice = FakeVoice(triggers)
        speech = FakeSpeech(heard) if isinstance(heard, Heard) else FakeSpeech(*heard)
        kw.setdefault("close_chime", lambda: chimes.append("close"))
        a = Assistant(triggers, speech, (lambda: flow) if flow else None, voice, None,
                      mic_factory=mics, chime=lambda: chimes.append(1), show_text=shown.append,
                      on_status=lambda s, p: statuses.append((s, p)), on_notice=notices.append,
                      retry_seconds=0.05, **kw)
        a.start()
        started.append(a)
        assert wait_for(lambda: a.status is Status.LISTENING)
        return a, triggers, mics, statuses, notices, shown, chimes

    yield make
    for a in started:
        a.stop(5)


# ---- the loop ---------------------------------------------------------------------------

def test_wake_hear_act_speak_with_wake_muted(run_assistant):
    flow = FakeFlow()
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(flow=flow)
    trig.fire.set()
    assert wait_for(lambda: flow.seen == ["open notepad"])
    assert wait_for(lambda: a.status is Status.LISTENING and len(statuses) >= 3)
    voice = a._voice
    assert voice.said == [("Opening notepad.", True, True)]          # spoken, waited for, while muted
    assert trig.muted_log == [True, False] and chimes == [1]
    assert (Status.WORKING, "") in statuses and shown == []          # public and spoken: no popup
    assert notices == []


def test_private_results_are_shown_on_screen(run_assistant):
    flow = FakeFlow(Result(True, "dear diary", private=True))
    a, trig, *_rest = run_assistant(flow=flow)
    shown = _rest[3]
    trig.fire.set()
    assert wait_for(lambda: shown == ["dear diary"])


@pytest.mark.parametrize("voice_ok", [False, None])
def test_replies_that_could_not_be_spoken_are_shown(run_assistant, voice_ok):
    triggers = FakeTriggers()
    voice = None if voice_ok is None else FakeVoice(triggers, ok=voice_ok)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(triggers=triggers, flow=FakeFlow(), voice=voice)
    trig.fire.set()
    assert wait_for(lambda: shown == ["Opening notepad."])


def test_hearing_problems_become_notices_and_nothing_runs(run_assistant):
    flow = FakeFlow()
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=Heard(reason="done", error="No Groq key is stored."))
    trig.fire.set()
    assert wait_for(lambda: notices == ["No Groq key is stored."])
    assert flow.seen == []


def test_ai_problems_become_notices(run_assistant):
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(flow=FakeFlow(error=BrainError("The AI service is busy.")))
    trig.fire.set()
    assert wait_for(lambda: notices == ["The AI service is busy."])


def test_nothing_heard_runs_nothing(run_assistant):
    flow = FakeFlow()
    a, trig, *_ = run_assistant(flow=flow, heard=Heard(reason="no speech"))
    trig.fire.set()
    time.sleep(0.2)
    assert flow.seen == [] and a.status is Status.LISTENING


# ---- the microphone is closed whenever you're not there ----------------------------------

@pytest.mark.parametrize("off, on, status", [
    ("pause", "resume", Status.PAUSED),
    ("lock", "unlock", Status.LOCKED),
    ("sleep", "wake", Status.ASLEEP),
])
def test_microphone_closed_while_paused_locked_or_asleep(run_assistant, off, on, status):
    a, trig, mics, *_ = run_assistant()
    actions = {"pause": a.pause, "resume": a.resume, "lock": lambda: a.set_locked(True),
               "unlock": lambda: a.set_locked(False), "sleep": lambda: a.set_asleep(True),
               "wake": lambda: a.set_asleep(False)}
    actions[off]()
    assert wait_for(lambda: a.status is status and mics.open_now == 0)
    time.sleep(0.2)
    assert mics.opens == 1                                   # stays closed
    actions[on]()
    assert wait_for(lambda: a.status is Status.LISTENING and mics.opens == 2 and mics.open_now == 1)


def test_toggle_pause(run_assistant):
    a, *_ = run_assistant()
    a.toggle_pause()
    assert wait_for(lambda: a.status is Status.PAUSED) and a.paused
    a.toggle_pause()
    assert wait_for(lambda: a.status is Status.LISTENING) and not a.paused


def test_unlock_while_paused_stays_paused(run_assistant):
    a, trig, mics, *_ = run_assistant()
    a.pause()
    a.set_locked(True)
    a.set_locked(False)
    time.sleep(0.2)
    assert a.status is Status.PAUSED and mics.open_now == 0


# ---- never crash ---------------------------------------------------------------------------

def test_missing_microphone_is_retried(run_assistant):
    mics = FakeMics(fail_opens=2)
    triggers = FakeTriggers()
    statuses = []
    a = Assistant(triggers, None, mic_factory=mics, chime=lambda: None,
                  on_status=lambda s, p: statuses.append((s, p)), retry_seconds=0.05)
    a.start()
    try:
        assert wait_for(lambda: a.status is Status.LISTENING)
        assert (Status.PROBLEM, "I couldn't open the microphone.") in statuses and mics.opens == 3
    finally:
        a.stop()


def test_silent_microphone_is_reopened(run_assistant):
    a, trig, mics, statuses, *_ = run_assistant(mics=FakeMics(stall_once=True))
    assert (Status.PROBLEM, "The microphone stopped sending sound.") in statuses and mics.opens >= 2


def test_watchdog_restarts_after_a_crash_and_logs_no_details(run_assistant):
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    main.log.addHandler(handler)
    try:
        a = Assistant(FakeTriggers(crash_once=True), None, mic_factory=FakeMics(), chime=lambda: None,
                      retry_seconds=0.05)
        a.start()
        assert wait_for(lambda: a.crashes == 1 and a.status is Status.LISTENING, timeout=5)
        a.stop()
    finally:
        main.log.removeHandler(handler)
    assert any("RuntimeError" in r for r in records)
    assert not any("secret" in r for r in records)


def test_stop_is_quick_and_final(run_assistant):
    a, trig, mics, *_ = run_assistant()
    t0 = time.monotonic()
    a.stop()
    assert time.monotonic() - t0 < 2 and a.status is Status.STOPPED and mics.open_now == 0


def test_your_words_are_never_logged(run_assistant):
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    main.log.addHandler(handler)
    old_level = main.log.level
    main.log.setLevel(logging.INFO)                         # the level the background app runs at
    try:
        flow = FakeFlow(Result(True, "the secret reply"))
        a, trig, *_ = run_assistant(flow=flow, heard=Heard(text="my secret plan", reason="done"))
        trig.fire.set()
        assert wait_for(lambda: flow.seen == ["my secret plan"])
        time.sleep(0.1)
    finally:
        main.log.removeHandler(handler)
        main.log.setLevel(old_level)
    assert records and not any("secret" in r for r in records)


def test_console_output_includes_the_conversation():
    lines = []
    trig = FakeTriggers()
    a = Assistant(trig, FakeSpeech(Heard(text="open notepad", reason="done")), lambda: FakeFlow(), None,
                  mic_factory=FakeMics(), chime=lambda: None, out=lines.append)
    a.start()
    try:
        assert wait_for(lambda: a.status is Status.LISTENING)
        trig.fire.set()
        assert wait_for(lambda: "OK: Opening notepad." in lines)
        assert lines[:2] == ["WAKE: hey jarvis (score 0.90)", 'Heard: "open notepad"']
    finally:
        a.stop()


# ---- follow-ups: answering without the wake word --------------------------------------------

ASK = Result(True, "Which app, sir?", question=True)
DONE = Result(True, "Opening notepad.")


def test_a_question_is_answered_without_the_wake_word(run_assistant):
    from backend.brain import Turn

    flow = FakeFlow(ASK, DONE)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=(Heard(text="open an app", reason="done"), Heard(text="notepad", reason="done")))
    trig.fire.set()
    assert wait_for(lambda: flow.seen == ["open an app", "notepad"])
    assert flow.histories == [(), (Turn("open an app", "Which app, sir?"),)]   # it knows what you're answering
    assert wait_for(lambda: a.status is Status.LISTENING)
    assert chimes == [1, 1]                         # wake chime, then the follow-up chime; no close tone needed
    assert a._speech.waits == [None, 5.0]           # the follow-up waits only 5 s for you to start


def test_silence_after_a_question_closes_the_window(run_assistant):
    flow = FakeFlow(ASK)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=(Heard(text="open an app", reason="done"), Heard(reason="no speech")))
    trig.fire.set()
    assert wait_for(lambda: chimes == [1, 1, "close"])
    assert flow.seen == ["open an app"]


@pytest.mark.parametrize("words", ["stop", "Cancel.", "Never mind, Jarvis.", "that's all", "Thanks, sir."])
def test_stop_words_close_the_window_and_run_nothing(run_assistant, words):
    flow = FakeFlow(ASK)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=(Heard(text="open an app", reason="done"), Heard(text=words, reason="done")))
    trig.fire.set()
    assert wait_for(lambda: chimes == [1, 1, "close"])
    assert flow.seen == ["open an app"]


def test_ordinary_replies_need_the_wake_word_again(run_assistant):
    flow = FakeFlow(DONE)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(flow=flow)
    trig.fire.set()
    assert wait_for(lambda: flow.seen == ["open notepad"] and a.status is Status.LISTENING)
    time.sleep(0.1)
    assert chimes == [1] and a._speech.waits == [None]


def test_follow_ups_stop_after_the_turn_limit(run_assistant):
    flow = FakeFlow(ASK)                            # it keeps asking questions
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=Heard(text="something", reason="done"), follow_up_max_turns=3)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert len(flow.seen) == 1 + 3 and chimes == [1, 1, 1, 1, "close"]


# ---- conversation mode ------------------------------------------------------------------------

NOTEPAD = Result(True, "Opening notepad.", action={"tool": "open_app", "args": {"name": "notepad"}})
CALC = Result(True, "Opening calculator.", action={"tool": "open_app", "args": {"name": "calculator"}})


def said(*texts):
    return tuple(Heard(text=t, reason="done") for t in texts)


def test_a_conversation_carries_on_without_the_wake_word_and_remembers(run_assistant):
    from backend.brain import Turn

    flow = FakeFlow(NOTEPAD, CALC)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=said("open notepad", "and the calculator") + (Heard(reason="no speech"),), conversation=True)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert flow.seen == ["open notepad", "and the calculator"]
    assert flow.histories == [(), (Turn("open notepad", action=NOTEPAD.action),)]   # its own proposal, remembered
    assert a._speech.waits == [None, 8.0, 8.0]                 # 8 s to start each new turn
    assert chimes == [1, 1, 1, "close"]


def test_tool_output_is_never_remembered(run_assistant):
    flow = FakeFlow(Result(True, "dear diary, my secrets"), DONE)     # no proposal recorded, not the AI's words
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=said("read my note", "thanks again") + (Heard(reason="no speech"),), conversation=True)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert flow.histories == [(), ()]


def test_the_ai_s_own_replies_are_remembered(run_assistant):
    from backend.brain import Turn

    hello = Result(True, "Good day, sir.", ai_reply=True)
    flow = FakeFlow(hello, DONE)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=said("hello jarvis", "open notepad") + (Heard(reason="no speech"),), conversation=True)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert flow.histories[1] == (Turn("hello jarvis", reply="Good day, sir."),)


@pytest.mark.parametrize("goodbye", ["That's all.", "Goodbye, Jarvis.", "Thanks, Jarvis.", "okay thanks", "stop"])
def test_saying_goodbye_ends_politely(run_assistant, goodbye):
    from backend.voice.style import Style

    flow = FakeFlow(NOTEPAD)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=said("open notepad", goodbye), conversation=True, style=Style(choose=first_choice))
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert flow.seen == ["open notepad"]                        # the goodbye is never sent to the AI
    assert a._voice.said[-1] == ("Very good, sir.", True, True)


def test_never_mind_straight_after_the_wake_word(run_assistant):
    from backend.voice.style import Style

    flow = FakeFlow()
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=said("never mind"), conversation=True, style=Style(choose=first_choice))
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert flow.seen == [] and a._voice.said == [("Very good, sir.", True, True)]


def test_conversation_ends_at_the_turn_limit(run_assistant):
    flow = FakeFlow(NOTEPAD)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=Heard(text="open notepad", reason="done"), conversation=True, conversation_max_turns=2)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert len(flow.seen) == 1 + 2


def test_conversation_ends_at_the_time_limit(run_assistant):
    flow = FakeFlow(NOTEPAD)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=Heard(text="open notepad", reason="done"), conversation=True, conversation_max_seconds=0.0)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert flow.seen == ["open notepad"]


def test_memory_holds_only_the_latest_turns(run_assistant):
    flow = FakeFlow(NOTEPAD)
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, heard=said("one", "two", "three", "four") + (Heard(reason="no speech"),),
        conversation=True, memory_turns=2)
    trig.fire.set()
    assert wait_for(lambda: chimes and chimes[-1] == "close")
    assert [len(h) for h in flow.histories] == [0, 1, 2, 2]
    assert [t.user for t in flow.histories[-1]] == ["two", "three"]


def test_screen_shows_in_conversation():
    from backend.main import screen_feed

    seen = []

    class S:
        def set_state(self, *a):
            seen.append(a)

    _, on_event = screen_feed(S())
    on_event("hearing", {"follow_up": True, "conversation": True})
    on_event("hearing", {"follow_up": True, "conversation": False})
    on_event("hearing", {"follow_up": False})
    assert seen == [("hearing", "in conversation"), ("hearing", "your answer"), ("hearing", "go ahead")]


def test_follow_up_on_question_can_be_turned_off(run_assistant):
    flow = FakeFlow(ASK)
    a, trig, *_rest = run_assistant(flow=flow, follow_up_on_question=False)
    trig.fire.set()
    assert wait_for(lambda: flow.seen == ["open notepad"] and a.status is Status.LISTENING)
    time.sleep(0.1)
    assert a._speech.waits == [None]


def test_own_voice_is_dropped_before_each_follow_up(run_assistant):
    drains = []

    class CountingMics(FakeMics):
        def __call__(self, device):
            mic = FakeMic(self)
            mic.drain = lambda: drains.append(1) or 0
            return mic

    flow = FakeFlow(ASK, DONE)
    a, trig, *_ = run_assistant(flow=flow, mics=CountingMics(),
                                heard=(Heard(text="open an app", reason="done"), Heard(text="notepad", reason="done")))
    trig.fire.set()
    assert wait_for(lambda: flow.seen == ["open an app", "notepad"] and a.status is Status.LISTENING)
    assert len(drains) >= 2                          # once before the follow-up, once after the command


def test_pausing_during_a_question_ends_the_follow_up(run_assistant):
    flow = FakeFlow(ASK)

    class PausingSpeech(FakeSpeech):
        def hear(self, chunks, no_speech_seconds=None):
            if no_speech_seconds is not None:
                holder["a"].pause()
                return Heard(reason="stopped")
            return Heard(text="open an app", reason="done")

    holder = {}
    a, trig, mics, *_ = run_assistant(flow=flow)
    holder["a"] = a
    a._speech = PausingSpeech(Heard(reason="stopped"))
    trig.fire.set()
    assert wait_for(lambda: a.status is Status.PAUSED and mics.open_now == 0)
    assert flow.seen == ["open an app"]


@pytest.mark.parametrize("text, stop", [("stop", True), ("Cancel!", True), ("Thank you, Jarvis.", True),
                                        ("notepad", False), ("stop the music", False), ("", False)])
def test_stop_phrases(text, stop):
    assert main.is_stop_phrase(text) is stop


# ---- JARVIS speaking style ------------------------------------------------------------------

def first_choice(options):
    return options[0]


def test_spoken_and_shown_replies_use_the_jarvis_style(run_assistant):
    from backend.voice.style import Style

    events = []
    flow = FakeFlow()
    a, trig, *_ = run_assistant(flow=flow, style=Style(choose=first_choice),
                                on_event=lambda k, d: events.append((k, d)))
    trig.fire.set()
    assert wait_for(lambda: a._voice.said)
    assert a._voice.said == [("Opening Notepad, sir.", True, True)]
    reply = next(d for k, d in events if k == "reply")
    assert reply["text"] == "Opening Notepad, sir." and reply["private"] is False


def test_private_replies_are_spoken_as_a_fixed_line_and_shown_in_full(run_assistant):
    from backend.voice.style import Style

    events = []
    flow = FakeFlow(Result(True, "dear diary", private=True))
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(
        flow=flow, style=Style(choose=first_choice), on_event=lambda k, d: events.append((k, d)))
    trig.fire.set()
    assert wait_for(lambda: shown == ["dear diary"])
    assert a._voice.said == [("It's on your screen, sir.", True, True)]
    reply = next(d for k, d in events if k == "reply")
    assert reply["text"] == "dear diary" and reply["private"] is True       # on the local screen only


def test_the_greeting_is_spoken_muted_and_shown(run_assistant):
    events = []
    a, trig, mics, *_ = run_assistant(on_event=lambda k, d: events.append((k, d)))
    a.say_line("Good morning, sir. All systems are online.")
    assert wait_for(lambda: a._voice.said)
    assert a._voice.said == [("Good morning, sir. All systems are online.", True, True)]
    assert ("said", {"text": "Good morning, sir. All systems are online."}) in events
    a.say_line("   ")
    time.sleep(0.1)
    assert len(a._voice.said) == 1                           # nothing to say: nothing spoken


def test_timer_lines_use_the_style_too(run_assistant):
    from backend.voice.style import Style

    a, trig, mics, statuses, notices, shown, chimes = run_assistant(alarm=lambda: None, style=Style(choose=first_choice))
    a.announce("tea")
    assert wait_for(lambda: a._voice.said)
    assert a._voice.said == [("Sir, a reminder for you. It's on your screen.", True, True)]
    assert shown == ["Timer done: tea"]


# ---- finished timers ------------------------------------------------------------------------

def test_a_reminder_rings_speaks_a_fixed_line_and_shows_its_label(run_assistant):
    alarms = []
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(alarm=lambda: alarms.append(1))
    a.announce("take my medicine")
    assert wait_for(lambda: shown == ["Timer done: take my medicine"])
    assert alarms == [1]
    assert a._voice.said == [(main.TIMER_DONE_LINE, True, True)]      # fixed line, spoken while muted
    assert "medicine" not in main.TIMER_DONE_LINE


def test_a_plain_timer_is_spoken_and_not_shown(run_assistant):
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(alarm=lambda: None)
    a.announce("")
    assert wait_for(lambda: a._voice.said == [(main.TIMER_DONE_LINE, True, True)])
    time.sleep(0.05)
    assert shown == []


def test_a_plain_timer_is_shown_when_it_cannot_be_spoken(run_assistant):
    a, trig, mics, statuses, notices, shown, chimes = run_assistant(alarm=lambda: None, voice=None)
    a.announce("")
    assert wait_for(lambda: shown == ["Your timer is done."])


def test_timers_still_ring_while_paused(run_assistant):
    alarms = []
    a, trig, mics, *_ = run_assistant(alarm=lambda: alarms.append(1))
    a.pause()
    assert wait_for(lambda: a.status is Status.PAUSED)
    a.announce("tea")
    assert wait_for(lambda: alarms == [1]) and mics.open_now == 0


def test_reminder_labels_are_never_logged(run_assistant):
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    main.log.addHandler(handler)
    old_level = main.log.level
    main.log.setLevel(logging.INFO)
    try:
        a, trig, mics, statuses, notices, shown, chimes = run_assistant(alarm=lambda: None)
        a.announce("secret appointment")
        assert wait_for(lambda: shown)
    finally:
        main.log.removeHandler(handler)
        main.log.setLevel(old_level)
    assert "timer finished" in records and not any("secret" in r for r in records)


def test_console_timer_message(capsys):
    from backend import cli

    said = []

    class V:
        def say(self, result, wait=False):
            said.append(result.message)
            return True

    cli._console_timer_done(V())("tea")
    cli._console_timer_done(None)("")
    out = capsys.readouterr().out
    assert "TIMER: Timer done: tea" in out and "TIMER: Your timer is done." in out
    assert len(said) == 1 and "reminder" in said[0].lower() and "tea" not in said[0]   # label never spoken


def test_connections_are_warmed_on_wake_and_timing_is_logged(run_assistant):
    warmed, records = [], []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    main.log.addHandler(handler)
    old_level = main.log.level
    main.log.setLevel(logging.INFO)
    try:
        flow = FakeFlow()
        a, trig, *_ = run_assistant(flow=flow, warm=lambda: warmed.append(1))
        trig.fire.set()
        assert wait_for(lambda: flow.seen and any(r.startswith("timing:") for r in records))
    finally:
        main.log.removeHandler(handler)
        main.log.setLevel(old_level)
    assert warmed == [1]
    timing = next(r for r in records if r.startswith("timing:"))
    assert "hearing" in timing and "notepad" not in timing


def test_a_failing_warm_up_changes_nothing(run_assistant):
    def boom():
        raise OSError("offline")

    flow = FakeFlow()
    a, trig, *_ = run_assistant(flow=flow, warm=boom)
    trig.fire.set()
    assert wait_for(lambda: flow.seen == ["open notepad"])


def test_without_ai_says_so():
    lines, notices = [], []
    a = Assistant(FakeTriggers(), None, None, None, out=lines.append, on_notice=notices.append)
    a._act("open notepad")
    assert lines == ["NO: The AI is not set up."] and notices == ["The AI is not set up."]


def test_build_assistant_from_config(monkeypatch, audit):
    monkeypatch.setattr(main, "WakeWordDetector", lambda d: "wake")
    monkeypatch.setattr(main, "ClapDetector", lambda level, on_clap=None: ("clap", on_clap))
    monkeypatch.setattr(main, "pick_microphone", lambda name: 7)
    monkeypatch.setattr(main, "make_speech", lambda audit, on_recorded=None: ("speech", on_recorded))
    monkeypatch.setattr(main, "SileroVAD", lambda d: "vad")
    monkeypatch.setattr(main, "VoiceCheck", lambda vad, seconds, prob: ("voice check", vad, seconds))
    monkeypatch.setattr(config, "CLAP_ENABLED", True)
    a = main.build_assistant(audit, None, None)
    assert a._device == 7 and a._triggers._clap == ("clap", None)
    assert a._triggers._voice_check == ("voice check", "vad", config.WAKE_MIN_SPEECH_SECONDS)
    assert a._speech == ("speech", main.play_done_chime) and a._warm is main.warm_all
    lines = []
    b = main.build_assistant(audit, None, None, out=lines.append)
    b._triggers._clap[1](1234)
    assert lines == ["  (clap heard, level 1234)"]
    monkeypatch.setattr(config, "CLAP_ENABLED", False)
    monkeypatch.setattr(config, "WAKE_SPEECH_CHECK", False)
    c = main.build_assistant(audit, None, None)
    assert c._triggers._clap is None and c._triggers._voice_check is None


def test_false_trigger_defaults():
    assert config.CLAP_ENABLED is False                         # knocks and taps looked like claps
    assert config.WAKE_SPEECH_CHECK is True
    assert 0.2 <= config.WAKE_MIN_SPEECH_SECONDS <= 0.8
    assert 0.5 <= config.WAKE_SPEECH_CONFIDENCE <= 0.9
    assert 0.5 <= config.LISTEN_START_CONFIDENCE <= 0.9
    assert 0.3 <= config.LISTEN_MIN_SPEECH_SECONDS <= 1.0


# ---- startup ----------------------------------------------------------------------------

def test_ai_is_prepared_at_startup_without_failing():
    built = []
    main._prepare(lambda: built.append(1))
    assert built == [1]

    def no_key():
        raise BrainError("No Gemini key is stored.")

    main._prepare(no_key)                                  # must not raise


def test_faster_model_is_tried_first():
    assert config.GEMINI_MODELS[0] == "gemini-3.5-flash-lite"


def test_startup_checks_list_missing_keys_without_values():
    stored = {"gemini_api_key": "x" * 20, "stt_api_key": None, "fish_api_key": ""}
    problems = startup_checks(lambda name: stored[name])
    assert len(problems) == 2
    assert "stt_api_key" in problems[0] and "fish_api_key" in problems[1]
    assert startup_checks(lambda name: "k" * 20) == []


def test_startup_checks_without_credential_manager():
    from backend.credentials import CredentialError

    def broken(name):
        raise CredentialError("no vault")

    assert "Credential Manager" in startup_checks(broken)[0]


def test_only_one_copy_can_run():
    name = f"Local\\JARVIS-test-{uuid.uuid4().hex}"
    first = main.acquire_single_instance(name)
    try:
        assert first is not None
        assert main.acquire_single_instance(name) is None
    finally:
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(first))
    second = main.acquire_single_instance(name)
    assert second is not None
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(second))


def test_lower_priority():
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    try:
        assert main.lower_priority()
        assert k.GetPriorityClass(k.GetCurrentProcess()) == 0x4000
    finally:
        k.SetPriorityClass(k.GetCurrentProcess(), 0x20)         # back to normal for the other tests


def test_error_log_is_capped_and_outside_the_workspace(tmp_path):
    handler = main.setup_logging(tmp_path / "jarvis.log", console=False)
    try:
        assert handler.maxBytes == 1_000_000 and handler.backupCount == 2
        main.log.info("hello")
        handler.flush()
        assert "hello" in (tmp_path / "jarvis.log").read_text()
    finally:
        main.log.removeHandler(handler)
        handler.close()
    assert config.WORKSPACE not in config.APP_LOG.parents


def test_microphone_that_goes_silent_raises():
    class Stream:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    with Microphone(stream_factory=Stream) as mic:
        t0 = time.monotonic()
        with pytest.raises(TriggerError, match="stopped sending sound"):
            next(mic.chunks(idle_timeout=0.3))
        assert time.monotonic() - t0 < 1.5


# ---- the private reply popup ---------------------------------------------------------------

def test_message_text_keeps_lines_but_escapes_tricks():
    text = confirm.message_text("line one\nline two‮\x07\r\nthree")
    assert text.split("\n")[:2] == ["line one", "line two\\u202e\\u0007"] and text.endswith("three")
    assert confirm.message_text("x" * 9000).endswith("(cut short)")


def test_show_message_hands_over_escaped_text():
    sent = []
    assert confirm.show_message("hi‮", launcher=sent.append)
    assert "\\\\u202e" in sent[0]                      # escaped before it ever reaches the window
    assert confirm.show_message("x", launcher=lambda p: (_ for _ in ()).throw(OSError())) is False


@pytest.mark.parametrize("stdin, code", [
    ('{"text": "hello", "timeout": 5}', 0),
    ('{"text": 5, "timeout": 5}', 2),
    ('{"text": "hi", "timeout": 9999}', 2),
    ("not json", 2),
])
def test_message_window_process_checks_its_input(stdin, code):
    assert confirm._message_child(stdin, show=lambda text, timeout: True) == code


def test_message_window_failure_is_reported():
    def boom(text, timeout):
        raise RuntimeError("no screen")

    assert confirm._message_child('{"text": "hello", "timeout": 5}', show=boom) == 3


# ---- tray icon --------------------------------------------------------------------------

def test_tooltips_say_what_is_happening():
    assert tray.tooltip_text(Status.LISTENING) == "J.A.R.V.I.S: Listening"
    assert "microphone off" in tray.tooltip_text("locked")
    assert tray.tooltip_text("problem", "x" * 300).endswith("...") and len(tray.tooltip_text("problem", "x" * 300)) <= 127


def test_icon_image_is_a_valid_32bit_icon():
    data = tray.icon_image(16, (46, 204, 113))
    size, w, h, planes, bits = struct.unpack("<IiiHH", data[:16])
    assert (size, w, h, planes, bits) == (40, 16, 32, 1, 32)
    assert len(data) == 40 + 16 * 16 * 4 + 4 * 16
    centre = 40 + (8 * 16 + 8) * 4
    assert data[centre + 3] == 255 and data[40 + 3] == 0       # solid middle, transparent corner


def test_menu_labels_follow_pause_state():
    paused = [False]
    t = tray.Tray(is_paused=lambda: paused[0])
    assert t.menu_labels()[0] == "Pause listening"
    paused[0] = True
    assert t.menu_labels() == ["Resume listening", "Quit J.A.R.V.I.S"]


def test_menu_commands():
    calls = []
    t = tray.Tray(on_toggle_pause=lambda: calls.append("pause"), on_quit=lambda: calls.append("quit"))
    t.handle_command(t.CMD_PAUSE)
    t.handle_command(t.CMD_QUIT)
    t.handle_command(0)                                       # menu dismissed: nothing happens
    assert calls == ["pause", "quit"]


def test_repeated_notices_are_not_shown_twice():
    t = tray.Tray()
    t.notify("J.A.R.V.I.S", "No Groq key.")
    t.notify("J.A.R.V.I.S", "No Groq key.")
    t.notify("J.A.R.V.I.S", "Something else.")
    assert list(t._notes) == [("J.A.R.V.I.S", "No Groq key."), ("J.A.R.V.I.S", "Something else.")]


def test_hidden_window_receives_lock_and_sleep_events():
    """A real (never shown) Windows window, sent the same messages Windows sends."""
    events = []
    t = tray.Tray(on_lock=lambda v: events.append(("lock", v)), on_sleep=lambda v: events.append(("sleep", v)),
                  show_icon=False)
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert t.ready.wait(5)
    api = tray._load_api()
    api.SendMessageW(t._hwnd, tray.WM_WTSSESSION_CHANGE, tray.WTS_SESSION_LOCK, 0)
    api.SendMessageW(t._hwnd, tray.WM_WTSSESSION_CHANGE, tray.WTS_SESSION_UNLOCK, 0)
    api.SendMessageW(t._hwnd, tray.WM_POWERBROADCAST, tray.PBT_APMSUSPEND, 0)
    api.SendMessageW(t._hwnd, tray.WM_POWERBROADCAST, tray.PBT_APMRESUMEAUTOMATIC, 0)
    t.set_status(Status.WORKING)                              # from another thread: must not crash
    t.quit()
    th.join(5)
    assert not th.is_alive()
    assert events == [("lock", True), ("lock", False), ("sleep", True), ("sleep", False)]


def test_icons_can_be_made_by_windows():
    api = tray._load_api()
    data = tray.icon_image(16, (231, 76, 60))
    icon = api.CreateIconFromResourceEx(data, len(data), True, 0x00030000, 16, 16, 0)
    assert icon
    api.DestroyIcon(icon)


@pytest.mark.skipif(os.environ.get("JARVIS_GUI_TESTS") != "1", reason="set JARVIS_GUI_TESTS=1 to show a real tray icon")
def test_real_tray_icon_appears_and_goes():
    t = tray.Tray()
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    assert t.ready.wait(5) and t._icon_added
    t.set_status(Status.LISTENING)
    time.sleep(1)
    t.quit()
    th.join(5)
    assert not th.is_alive()
