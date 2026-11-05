"""Teaching your voice from the screen: one sample at a time, at your pace. Fakes only: no
microphone, no model, no audio kept."""
import threading
import time

import numpy as np
import pytest

from backend import config
from backend.settings import COMMANDS, VoiceSettings
from backend.voice.teaching import TeachingSession
from backend.voice.triggers import CHUNK
from backend.voice.voiceprint import Voiceprint, VoiceprintError, load_voiceprint, normalize, save_voiceprint

rng = np.random.default_rng(3)
ME = normalize(rng.normal(size=256))


def like_me():
    return normalize(ME + rng.normal(scale=0.015, size=256))


def someone_else():
    return normalize(rng.normal(size=256))


def moment():
    return [np.zeros(CHUNK, np.int16) for _ in range(22)]


def wait_for(cond, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Harness:
    """A teaching session with a fake microphone: each Record press hears whatever `heard` holds."""

    def __init__(self, voice=like_me, heard=moment, prepare_error=None, others=None, current=None):
        self.updates, self.paused, self.saved, self.sounds_seen = [], [], [], []
        self.voice, self.heard = voice, heard
        self.release = threading.Event()
        self.release.set()

        def embed(samples):
            self.sounds_seen.append(len(samples))
            if len(samples) < 8000:
                raise VoiceprintError("Too little sound to recognise a voice.")
            return self.voice()

        def listen(stop, on_ready):
            on_ready()
            self.release.wait(5)
            return None if stop.is_set() else self.heard()

        def prepare():
            if prepare_error:
                raise prepare_error
            return embed, listen

        others = others if others is not None else [("David", np.zeros(32000, np.int16))] * 4
        self.session = TeachingSession(prepare, lambda: others, self.saved.append, self.paused.append,
                                       self.updates.append, current=current)

    def phase(self):
        return self.updates[-1]["phase"] if self.updates else None

    def wait_phase(self, phase):
        assert wait_for(lambda: self.phase() == phase), (phase, self.updates[-3:])

    def record(self):
        assert self.session.record()


def test_start_pauses_listening_and_waits_for_you():
    h = Harness()
    assert h.session.start()
    h.wait_phase("waiting")
    assert h.paused == [True] and h.updates[-1]["done"] == 0 and h.updates[-1]["of"] == 20


def test_each_sample_waits_for_a_press():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    h.record()
    h.wait_phase("waiting")
    assert h.updates[-1]["done"] == 1 and h.updates[-1]["outcome"] == "got_it"
    time.sleep(0.1)
    assert h.updates[-1]["done"] == 1                      # nothing more happens until the next press
    phases = [u["phase"] for u in h.updates]
    assert "say_it" in phases and phases.index("getting_ready") < phases.index("say_it")


def test_record_is_ignored_while_a_sample_is_being_taken():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    h.release.clear()
    h.record()
    h.wait_phase("say_it")
    assert not h.session.record()
    h.release.set()
    h.wait_phase("waiting")
    assert h.updates[-1]["done"] == 1


def test_twenty_samples_save_and_switch_back_on():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    for _ in range(19):
        h.record()
        assert wait_for(lambda: h.phase() == "waiting" and not h.session._busy)
    voices = iter([like_me()] + [someone_else() for _ in range(10)])   # your 20th, then Windows' voices
    h.voice = lambda: next(voices)
    h.record()
    h.wait_phase("result")
    final = h.updates[-1]
    assert final["saved"] is True and isinstance(final["bar"], float) and len(h.saved) == 1
    assert h.paused == [True, False] and not h.session.active
    assert all(n <= int(1.5 * 16000) for n in h.sounds_seen)


def test_a_miss_or_an_odd_sample_is_retaken():
    h = Harness(heard=lambda: None)
    h.session.start()
    h.wait_phase("waiting")
    h.record()
    h.wait_phase("waiting")
    assert h.updates[-1]["outcome"] == "missed" and h.updates[-1]["done"] == 0
    h.heard = moment
    for _ in range(6):
        h.record()
        assert wait_for(lambda: h.phase() == "waiting" and not h.session._busy)
    h.voice = someone_else
    h.record()
    assert wait_for(lambda: h.phase() == "waiting" and not h.session._busy)
    assert h.updates[-1]["outcome"] == "odd" and h.updates[-1]["done"] == 6


def test_redo_takes_the_last_one_off():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    assert not h.session.redo()                            # nothing to take off yet
    h.record()
    assert wait_for(lambda: h.updates[-1].get("done") == 1 and not h.session._busy)
    assert h.updates[-1]["can_redo"] is True
    assert h.session.redo() and h.updates[-1]["done"] == 0


def test_cancel_throws_everything_away_even_mid_sample():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    h.record()
    assert wait_for(lambda: h.updates[-1].get("done") == 1 and not h.session._busy)
    h.release.clear()
    h.record()
    h.wait_phase("say_it")
    h.session.cancel()
    h.release.set()
    h.wait_phase("cancelled")
    assert h.saved == [] and h.paused == [True, False] and h.session._kept == [] and not h.session.active


def test_cancel_while_waiting():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    h.session.cancel()
    assert h.phase() == "cancelled" and h.paused == [True, False]


def test_an_abandoned_session_cancels_itself():
    now = {"t": 0.0}
    h = Harness()
    h.session._clock = lambda: now["t"]
    h.session.start()
    h.wait_phase("waiting")
    now["t"] = 299
    h.session.check_idle()
    assert h.session.active
    now["t"] = 301
    h.session.check_idle()
    assert h.phase() == "cancelled" and not h.session.active and h.paused[-1] is False


def test_a_broken_model_is_reported_and_listening_comes_back():
    h = Harness(prepare_error=VoiceprintError("The voice-check model is not downloaded."))
    h.session.start()
    h.wait_phase("failed")
    assert "not downloaded" in h.updates[-1]["message"] and h.paused == [True, False]


def test_only_one_session_at_a_time():
    h = Harness()
    assert h.session.start() and not h.session.start()


def test_updates_hold_no_sound_or_voice_numbers():
    h = Harness()
    h.session.start()
    h.wait_phase("waiting")
    h.record()
    h.wait_phase("waiting")
    for u in h.updates:
        for value in u.values():
            assert not isinstance(value, (np.ndarray, bytes, bytearray, list)) or len(value) <= 2


def test_test_mode_scores_one_try():
    vp = Voiceprint(tuple(ME), 0.5, 20)
    h = Harness(current=vp)
    h.session.start("test")
    h.wait_phase("waiting")
    h.record()
    h.wait_phase("result")
    final = h.updates[-1]
    assert final["recognised"] is True and final["score"] > final["bar"] and h.saved == []


def test_test_mode_needs_a_voiceprint():
    h = Harness()
    assert not h.session.start("test")
    assert "no voiceprint" in h.updates[-1]["message"].lower()


# ---- the Settings page's commands ------------------------------------------------------------------

class FakeAssistant:
    def __init__(self, check=None):
        self.teaching, self.voice_check = [], check

    def set_teaching(self, on):
        self.teaching.append(on)

    def set_voice_check(self, check):
        self.voice_check = check


class Secrets:
    def __init__(self):
        self.store = {}

    def get(self, n):
        return self.store.get(n)

    def set(self, n, v):
        self.store[n] = v

    def delete(self, n):
        return self.store.pop(n, None) is not None


def make_settings(assistant=None, confirm=lambda tool, args: True, sessions=None):
    shown_teach, shown_voice, secrets = [], [], Secrets()
    sessions = sessions if sessions is not None else []

    class FakeSession:
        def __init__(self, **kw):
            self.kw, self.calls, self.active = kw, [], False
            sessions.append(self)

        def start(self, mode):
            self.calls.append(("start", mode))
            self.active = True

        def record(self):
            self.calls.append("record")

        def redo(self):
            self.calls.append("redo")

        def cancel(self):
            self.calls.append("cancel")
            self.active = False

        def check_idle(self):
            self.calls.append("idle")

    s = VoiceSettings(lambda: assistant, shown_teach.append, shown_voice.append, confirm, secrets.get,
                      secrets.set, secrets.delete, FakeSession, model_factory=lambda d: object())
    return s, shown_voice, secrets, sessions


def test_unknown_commands_are_dropped():
    s, shown, _, sessions = make_settings()
    for cmd in ("run_tool", "set_secret", "teach_start; rm", "", None, 7, "TEACH_START", "_teach_start", "install"):
        s.handle(cmd)
    assert sessions == [] and shown == []


def test_buttons_reach_the_session():
    s, _, _, sessions = make_settings()
    for cmd in ("teach_start", "teach_record", "teach_redo", "teach_cancel"):
        s.handle(cmd)
    assert sessions[0].calls == [("start", "teach"), "record", "redo", "cancel"]


def test_a_second_start_while_teaching_is_ignored():
    s, _, _, sessions = make_settings()
    s.handle("teach_start")
    s.handle("teach_start")
    s.handle("voice_test")
    assert len(sessions) == 1


def test_status_without_a_voiceprint():
    s, shown, _, _ = make_settings()
    s.handle("voice_status")
    assert shown[-1] == {"taught": False, "on": False}


def test_saving_switches_the_check_on_without_a_restart():
    from backend.voice.voiceprint import AllChecks

    silero = lambda chunks: True
    assistant = FakeAssistant(AllChecks([("no voice", silero)]))
    s, shown, secrets, _ = make_settings(assistant)
    s.install(Voiceprint(tuple(ME), 0.45, 20))
    names = [n for n, _ in assistant.voice_check._checks]
    assert names == ["no voice", "not your voice"] and assistant.voice_check._checks[0][1] is silero
    assert shown[-1]["taught"] and shown[-1]["on"] and shown[-1]["bar"] == 0.45
    assert load_voiceprint(config.VOICEPRINT, secrets.get) is not None


def test_forget_needs_the_keypress_and_switches_the_check_off():
    from backend.voice.voiceprint import AllChecks

    asked = []
    assistant = FakeAssistant(AllChecks([("no voice", lambda c: True)]))
    s, shown, secrets, _ = make_settings(assistant, confirm=lambda tool, args: asked.append(tool) or True)
    s.install(Voiceprint(tuple(ME), 0.45, 20))
    s.handle("voice_forget")
    assert asked == ["forget_my_voice"] and not config.VOICEPRINT.exists()
    assert [n for n, _ in assistant.voice_check._checks] == ["no voice"]
    assert shown[-1]["taught"] is False and "deleted" in shown[-1]["message"]


def test_forget_is_kept_without_the_keypress():
    s, shown, secrets, _ = make_settings(FakeAssistant(), confirm=lambda tool, args: False)
    s.install(Voiceprint(tuple(ME), 0.45, 20))
    s.handle("voice_forget")
    assert config.VOICEPRINT.exists() and "Kept" in shown[-1]["message"]


def test_a_changed_voiceprint_shows_as_a_problem():
    s, shown, secrets, _ = make_settings()
    save_voiceprint(Voiceprint(tuple(ME), 0.45, 20), config.VOICEPRINT, secrets.set)
    config.VOICEPRINT.write_text(config.VOICEPRINT.read_text().replace("0.45", "0.05"))
    s.handle("voice_status")
    assert shown[-1]["taught"] is False and "changed outside" in shown[-1]["problem"]


def test_settings_commands_are_exactly_the_documented_ones():
    assert set(COMMANDS) == {"teach_start", "teach_record", "teach_redo", "teach_cancel", "voice_test",
                             "voice_forget", "voice_status"}


# ---- the main loop lets go of the microphone while you teach ---------------------------------------------

def test_teaching_closes_the_main_microphone():
    from backend.main import Assistant, Status

    class Mics:
        def __call__(self, device):
            raise AssertionError("the microphone must stay closed while teaching")

    a = Assistant(type("T", (), {"reset": lambda s: None, "set_muted": lambda s, m: None,
                                 "set_voice_check": lambda s, c: None})(), None, mic_factory=Mics())
    a.set_teaching(True)
    assert not a.listening_allowed() and a._idle_status() is Status.PAUSED
    a.set_teaching(False)
    assert a.listening_allowed()
