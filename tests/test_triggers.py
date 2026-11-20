import shutil
import threading

import numpy as np
import pytest

from backend import cli, config, registry
from backend.gate.audit import AuditLog
from backend.voice import triggers as tr
from backend.voice.speak import normalize_wav
from backend.voice.triggers import (CHUNK, ClapDetector, Microphone, TriggerError, Triggers, WakeEvent,
                                    WakeWordDetector, download_models, pick_microphone, verify_models)


# ---- helpers ------------------------------------------------------------------------

def slices(audio: np.ndarray):
    audio = np.concatenate([audio.astype(np.int16), np.zeros((-len(audio)) % CHUNK, np.int16)])
    return [audio[i:i + CHUNK] for i in range(0, len(audio), CHUNK)]


# ---- model files --------------------------------------------------------------------

def test_missing_models_explained(tmp_path):
    with pytest.raises(TriggerError, match="download"):
        verify_models(tmp_path)


def test_changed_model_refused(tmp_path, models_dir):
    for name in tr.MODEL_FILES:
        shutil.copy(models_dir / name, tmp_path / name)
    verify_models(tmp_path)
    with open(tmp_path / "hey_jarvis_v0.1.onnx", "ab") as f:
        f.write(b"\x00")
    with pytest.raises(TriggerError, match="changed"):
        verify_models(tmp_path)


def test_download_saves_only_matching_files_from_fixed_urls(tmp_path, models_dir):
    asked = []

    def fetch(url):
        asked.append(url)
        return (models_dir / url.rsplit("/", 1)[1]).read_bytes()

    assert sorted(download_models(tmp_path, fetch)) == sorted(tr.MODEL_FILES)
    assert asked == [tr._RELEASE + n for n in tr.MODEL_FILES]
    assert all(u.startswith("https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/") for u in asked)
    verify_models(tmp_path)
    assert download_models(tmp_path, fetch) == [] and len(asked) == len(tr.MODEL_FILES)   # nothing fetched twice


@pytest.mark.parametrize("data", [b"not a model", b"x" * (tr.MAX_MODEL_BYTES + 1), None],
                         ids=["wrong-bytes", "too-big", "not-bytes"])
def test_download_refuses_wrong_content_and_saves_nothing(tmp_path, data):
    with pytest.raises(TriggerError):
        download_models(tmp_path, lambda url: data)
    assert list(tmp_path.iterdir()) == []


def test_download_network_error_hides_details(tmp_path):
    def fetch(url):
        raise OSError("TLS failure at 10.1.2.3")

    with pytest.raises(TriggerError) as exc:
        download_models(tmp_path, fetch)
    assert "10.1.2.3" not in str(exc.value)


def test_models_cli_status_and_usage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert tr.main(["status"]) == 1 and "download" in capsys.readouterr().out
    assert tr.main(["delete"]) == 2


# ---- "hey jarvis" ------------------------------------------------------------------------

def test_silence_and_noise_score_low(models_dir):
    det = WakeWordDetector(models_dir)
    silence = [det.feed(c) for c in slices(np.zeros(32000))]
    noise = [det.feed(c) for c in slices(np.random.default_rng(1).normal(0, 2000, 32000))]
    assert max(silence) < 0.05 and max(noise) < 0.25


@pytest.mark.parametrize("bad", [np.zeros(1279, np.int16), np.zeros(1280, np.float32), np.zeros((2, 640), np.int16)])
def test_feed_rejects_wrong_slices(models_dir, bad):
    with pytest.raises(ValueError):
        WakeWordDetector(models_dir).feed(bad)


def test_hey_jarvis_wakes_and_other_speech_does_not(models_dir, spoken):
    det = WakeWordDetector(models_dir)
    assert max(det.feed(c) for c in slices(spoken["wake"])) > 0.9
    det.reset()
    assert max(det.feed(c) for c in slices(spoken["other"])) < config.WAKE_THRESHOLD


def test_silence_gate_skips_work_without_changing_results(models_dir, spoken):
    quiet = np.zeros(48000, np.int16)
    audio = np.concatenate([quiet, spoken["wake"], quiet, spoken["wake"]])
    gated, plain = WakeWordDetector(models_dir), WakeWordDetector(models_dir, gate_level=None)
    a = np.array([gated.feed(c) for c in slices(audio)])
    b = np.array([plain.feed(c) for c in slices(audio)])
    assert gated.skipped > 40 and plain.skipped == 0
    # Every wake decision is the same. Scores can differ slightly where this synthetic audio jumps
    # from faint hiss to perfect digital silence, which a real room never does.
    assert ((a > config.WAKE_THRESHOLD) == (b > config.WAKE_THRESHOLD)).all()
    assert (a > config.WAKE_THRESHOLD).sum() > 5
    assert np.abs(a - b).max() < 0.1


# ---- double clap --------------------------------------------------------------------

def clap(level: float = 12000) -> np.ndarray:
    t = np.arange(320) / 16000                                   # 20 ms burst that dies away fast
    return np.random.default_rng(int(level)).normal(0, 1, 320) * level * np.exp(-t / 0.004)


def scene(events, seconds=3.0) -> np.ndarray:
    audio = np.random.default_rng(7).normal(0, 20, int(16000 * seconds))  # a quiet room
    for start, sound in events:
        i = int(start * 16000)
        audio[i:i + len(sound)] += sound
    return np.clip(audio, -32768, 32767).astype(np.int16)


def claps_found(audio, **kw) -> int:
    det = ClapDetector(**kw)
    return sum(det.feed(c) for c in slices(audio))


def test_double_clap_detected_once():
    assert claps_found(scene([(0.5, clap()), (0.9, clap())])) == 1


def test_quiet_uneven_claps_like_the_real_mic_are_detected():
    # Measured on the built-in mic: levels 897 and 1993, 0.29 s apart.
    assert claps_found(scene([(0.5, clap(2900)), (0.79, clap(6400))])) == 1


def test_clap_that_builds_over_a_few_milliseconds_is_detected():
    # Measured on the built-in mic: 33, 429, 588, 685, 702, 844, 897 in 10 ms steps, then fading.
    t = np.arange(2400) / 16000
    env = np.interp(t, [0, 0.01, 0.02, 0.03, 0.06, 0.15], [0.0, 0.48, 0.66, 0.77, 1.0, 0.0])
    built = np.random.default_rng(6).normal(0, 1, 2400) * 2400 * env
    assert claps_found(scene([(0.5, built), (0.79, clap(6400))])) == 1


def test_slow_rising_sound_is_not_a_clap():
    t = np.arange(2400) / 16000                                 # swells up over 150 ms, then stops
    swell = np.random.default_rng(5).normal(0, 1, 2400) * 8000 * (1 / 400) ** (1 - t / 0.15)
    assert claps_found(scene([(0.5, swell), (0.9, swell)])) == 0


@pytest.mark.parametrize("events", [
    [(0.5, clap())],                                             # one clap
    [(0.5, clap()), (0.6, clap())],                              # too close together
    [(0.5, clap()), (1.7, clap())],                              # too far apart
    [(0.5, clap(300)), (0.9, clap(300))],                        # too quiet
    [(0.5, clap(20000)), (0.9, clap(3500))],                     # very different loudness (5.7x)
    [(0.5, clap()), (0.9, clap()), (1.1, clap())],               # a third clap straight after
])
def test_not_a_double_clap(events):
    assert claps_found(scene(events)) == 0


def test_long_loud_sounds_are_never_claps():
    burst = np.random.default_rng(3).normal(0, 9000, 4800)       # 0.3 s of loud noise
    assert claps_found(scene([(0.4, burst), (1.0, burst)])) == 0


def test_speech_between_claps_cancels():
    talk = np.random.default_rng(4).normal(0, 6000, 3200)        # 0.2 s sustained sound
    assert claps_found(scene([(0.4, clap()), (0.6, talk), (1.0, clap())])) == 0


def test_each_clap_level_is_reported():
    levels = []
    claps_found(scene([(0.5, clap()), (0.9, clap())]), on_clap=levels.append)
    assert len(levels) == 2 and all(level > 3000 for level in levels)


# ---- wake rules ------------------------------------------------------------------------

class FakeWake:
    def __init__(self, scores):
        self.scores = iter(scores)

    def feed(self, chunk):
        return next(self.scores)


class FakeClap:
    def __init__(self, results):
        self.results = iter(results)

    def feed(self, chunk):
        return next(self.results)


CH = np.zeros(CHUNK, np.int16)


def run_rules(scores, claps=None, clock_step=0.08, **kw):
    now = [0.0]

    def clock():
        return now[0]

    t = Triggers(FakeWake(scores), FakeClap(claps or [False] * len(scores)), clock=clock, **kw)
    out = []
    for _ in scores:
        now[0] += clock_step
        out.append(t.process(CH))
    return out, t, now


def test_wakes_above_threshold_with_patience_one():
    out, _, _ = run_rules([0.1, 0.3, 0.1], threshold=0.25, patience=1)
    assert out == [None, WakeEvent("hey jarvis", 0.3), None]


def test_patience_needs_slices_in_a_row():
    out, _, _ = run_rules([0.3, 0.1, 0.3, 0.3], threshold=0.25, patience=2)
    assert [e is not None for e in out] == [False, False, False, True]


def test_cooldown_blocks_a_quick_second_wake():
    scores = [0.9] + [0.0] * 5 + [0.9] + [0.0] * 20 + [0.9]
    out, _, _ = run_rules(scores, threshold=0.25, patience=1, cooldown_seconds=1.5)
    assert sum(e is not None for e in out) == 2               # the one 0.48 s later is ignored


def test_muted_while_speaking_and_for_a_moment_after():
    now = [0.0]
    t = Triggers(FakeWake([0.9] * 100), None, threshold=0.25, patience=1, cooldown_seconds=0.1,
                 clock=lambda: now[0])
    t.set_muted(True)
    now[0] = 5.0
    assert t.process(CH) is None
    t.set_muted(False)
    now[0] = 5.5
    assert t.process(CH) is None                              # still inside the 1 s tail
    now[0] = 6.2
    assert t.process(CH) == WakeEvent("hey jarvis", 0.9)


def test_double_clap_wakes_and_every_wake_is_audited(audit):
    out, _, _ = run_rules([0.0, 0.0], claps=[False, True], audit=audit)
    assert out[1] == WakeEvent("double clap")
    rows = audit.path.read_text().splitlines()
    assert len(rows) == 1 and '"event":"wake"' in rows[0] and "double clap" in rows[0]
    assert audit.verify()[0]


def test_reset_forgets_old_sound(models_dir, spoken):
    det, clap_det = WakeWordDetector(models_dir), ClapDetector()
    t = Triggers(det, clap_det)
    for c in slices(spoken["wake"][:24000]):                 # half of the phrase, then a gap
        t.process(c)
    t.reset()
    assert len(det._embeddings) == 0 and clap_det._last_clap is None
    assert all(t.process(c) is None for c in slices(np.zeros(16000)))


# ---- the voice check at wake-up (synthetic audio only) ---------------------------------------

def test_wake_without_a_voice_is_ignored_and_audited(audit):
    asked = []

    def check(chunks):
        asked.append(len(chunks))
        return False

    t = Triggers(FakeWake([0.1, 0.4]), threshold=0.25, audit=audit, voice_check=check)
    assert [t.process(CH), t.process(CH)] == [None, None]
    assert asked == [2] and t.rejected == 1
    row = audit.path.read_text()
    assert '"event":"wake_rejected"' in row and '"reason":"no voice"' in row and '"event":"wake"' not in row


def test_wake_with_a_voice_goes_ahead():
    t = Triggers(FakeWake([0.4]), threshold=0.25, voice_check=lambda chunks: True)
    assert t.process(CH) == WakeEvent("hey jarvis", 0.4)


def test_a_rejected_moment_is_not_rechecked_every_slice():
    now, asked = [0.0], []
    t = Triggers(FakeWake([0.4] * 20), threshold=0.25, clock=lambda: now[0],
                 voice_check=lambda chunks: asked.append(1) or False)
    for _ in range(10):                                       # 0.8 s of high scores
        now[0] += 0.08
        t.process(CH)
    assert len(asked) == 3                                    # checked at most every 0.3 s, not every slice
    assert Triggers.REJECT_PAUSE_SECONDS == 0.3               # a quick repeat isn't lost in a long pause
    now[0] += 1.0
    t.process(CH)
    assert len(asked) == 4


def test_a_broken_check_never_makes_it_deaf():
    def boom(chunks):
        raise RuntimeError("model missing")

    assert Triggers(FakeWake([0.4]), threshold=0.25, voice_check=boom).process(CH) is not None


def test_recent_sound_is_available_for_one_breath_commands():
    t = Triggers(FakeWake([0.0] * 30))
    for i in range(30):
        t.process(np.full(CHUNK, i, np.int16))
    recent = t.recent(1.6)
    assert len(recent) == 20 and int(recent[0][0]) == 10 and int(recent[-1][0]) == 29
    assert len(t.recent(10)) == Triggers.RECENT_CHUNKS and t.recent(0) == []


def test_only_about_two_seconds_are_kept_and_reset_clears_them():
    t = Triggers(FakeWake([0.0] * 40), voice_check=lambda chunks: True)
    for _ in range(40):
        t.process(CH)
    assert len(t._recent) == Triggers.RECENT_CHUNKS
    t.reset()
    assert len(t._recent) == 0


def test_claps_are_not_voice_checked():
    asked = []
    t = Triggers(FakeWake([0.0]), FakeClap([True]), voice_check=lambda c: asked.append(1) or False)
    assert t.process(CH) == WakeEvent("double clap") and asked == []


def _hum(seconds, hz=120):
    t = np.arange(int(16000 * seconds)) / 16000
    return (np.sin(2 * np.pi * hz * t) * 4000).astype(np.int16)


def _clicks(seconds):
    audio = np.zeros(int(16000 * seconds))
    for start in range(0, len(audio) - 200, 2400):          # a knock every 0.15 s
        audio[start:start + 200] = np.random.default_rng(start).normal(0, 9000, 200)
    return audio.astype(np.int16)


def _typing(seconds):
    audio = np.zeros(int(16000 * seconds))
    for start in range(0, len(audio) - 80, 1760):             # a key every 0.11 s
        audio[start:start + 80] = np.random.default_rng(start).normal(0, 3000, 80) * np.exp(-np.arange(80) / 60)
    return audio.astype(np.int16)


def real_check(models_dir):
    from backend.voice.listen import SileroVAD, VoiceCheck

    return VoiceCheck(SileroVAD(models_dir), config.WAKE_MIN_SPEECH_SECONDS, config.WAKE_SPEECH_CONFIDENCE)


@pytest.mark.parametrize("make", [
    lambda: np.random.default_rng(2).normal(0, 2500, 32000).astype(np.int16),          # hiss / fan
    lambda: np.cumsum(np.random.default_rng(3).normal(0, 300, 32000)).clip(-12000, 12000).astype(np.int16),
    lambda: _hum(2.0),                                                                  # mains hum
    lambda: _clicks(2.0),                                                               # fast knocking
    lambda: _typing(2.0),                                                               # keyboard
], ids=["hiss", "rumble", "hum", "knocks", "typing"])
def test_real_voice_check_ignores_background_sounds(models_dir, make):
    assert real_check(models_dir)(slices(make())) is False


def test_real_voice_check_hears_a_spoken_wake_word(models_dir, spoken):
    wake = spoken["wake"][16000:-16000]                       # the phrase without its padding
    at_wake_time = np.concatenate([np.zeros(16000, np.int16), wake, np.zeros(6400, np.int16)])[-32000:]
    assert real_check(models_dir)(slices(at_wake_time)) is True


def test_reset_calls_both_detectors():
    seen = []

    class D:
        def reset(self):
            seen.append(1)

    Triggers(D(), D()).reset()
    assert seen == [1, 1]


@pytest.mark.parametrize("kw", [{"threshold": 0}, {"threshold": 1}, {"patience": 0}, {"patience": 1.5}])
def test_bad_rules_refused(kw):
    with pytest.raises(ValueError):
        Triggers(**kw)


# ---- microphone ----------------------------------------------------------------------

DEVICES = [
    {"name": "Virtual Mic (AudioRelay)", "max_input_channels": 2, "hostapi": 0},
    {"name": "Microphone Array (AMD Audio Device)", "max_input_channels": 2, "hostapi": 1},
    {"name": "Microphone Array (AMD Audio Dev", "max_input_channels": 2, "hostapi": 0},
    {"name": "Speakers", "max_input_channels": 0, "hostapi": 0},
]
HOSTAPIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]


def test_pick_microphone_prefers_mme_and_ignores_outputs():
    assert pick_microphone("microphone array", DEVICES, HOSTAPIS) == 2
    assert pick_microphone(None, DEVICES, HOSTAPIS) is None
    with pytest.raises(TriggerError, match="MIC_DEVICE"):
        pick_microphone("Speakers", DEVICES, HOSTAPIS)


class FakeStream:
    def __init__(self, fail=False, **kw):
        if fail:
            raise OSError("PortAudio error -9999")
        self.kw, self.started, self.closed = kw, False, False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


def test_microphone_streams_slices_and_closes():
    made = []

    def factory(**kw):
        made.append(FakeStream(**kw))
        return made[-1]

    with Microphone(device=2, stream_factory=factory) as mic:
        s = made[0]
        assert s.started and s.kw["samplerate"] == 16000 and s.kw["blocksize"] == CHUNK and s.kw["device"] == 2
        s.kw["callback"](np.full((CHUNK, 1), 7, np.int16), CHUNK, None, None)
        stop = threading.Event()
        got = next(mic.chunks(stop))
        assert got.dtype == np.int16 and got.shape == (CHUNK,) and got[0] == 7
    assert s.closed


def test_microphone_drops_oldest_when_behind():
    made = []
    with Microphone(stream_factory=lambda **kw: made.append(FakeStream(**kw)) or made[-1], max_queue=2) as mic:
        for value in (1, 2, 3):
            made[0].kw["callback"](np.full((CHUNK, 1), value, np.int16), CHUNK, None, None)
        it = mic.chunks()
        assert [next(it)[0], next(it)[0]] == [2, 3] and mic.dropped == 1


def test_microphone_drain_drops_queued_slices():
    made = []
    with Microphone(stream_factory=lambda **kw: made.append(FakeStream(**kw)) or made[-1]) as mic:
        for value in (1, 2, 3):
            made[0].kw["callback"](np.full((CHUNK, 1), value, np.int16), CHUNK, None, None)
        assert mic.drain() == 3 and mic.drain() == 0


def test_microphone_open_failure_hides_details():
    with pytest.raises(TriggerError) as exc:
        with Microphone(stream_factory=lambda **kw: FakeStream(fail=True)):
            pass
    assert "9999" not in str(exc.value)


# ---- chime ------------------------------------------------------------------------------

def test_chime_is_a_short_clean_wav():
    assert normalize_wav(tr.CHIME) == tr.CHIME
    assert len(tr.CHIME) < 22050 * 2 * 0.3                       # under 0.3 s


def test_play_chime_does_not_wait():
    played = threading.Event()

    class Player:
        def play(self, audio):
            played.set()
            return True

    tr.play_chime(Player())
    assert played.wait(2)


# ---- console ---------------------------------------------------------------------------

@pytest.fixture
def console(tmp_path):
    log = tmp_path / "log.jsonl"
    return registry.build_gate(tmp_path / "ws", log), AuditLog(log)


def test_listen_without_listener(console):
    assert cli.handle_line(*console, "listen") == "NO: Listening is not set up."


def test_listen_reports_setup_errors(console):
    def broken():
        raise TriggerError("The wake-word models are not downloaded.")

    assert cli.handle_line(*console, "listen", get_listener=broken) == "NO: The wake-word models are not downloaded."


def test_listen_stops_on_ctrl_c(console):
    def listener():
        raise KeyboardInterrupt

    assert cli.handle_line(*console, "listen", get_listener=lambda: listener) == "Stopped listening."


def test_help_mentions_listen(console):
    assert "listen" in cli.handle_line(*console, "help")


def test_near_misses_reported_only_below_threshold():
    from backend.main import NearMisses

    near = NearMisses(0.25)
    seen = [near.update(s, False) for s in (0.0, 0.18, 0.2, 0.0, 0.0, 0.0)]
    assert seen[-1] == 0.2 and seen[:-1] == [None] * 5
    seen = [near.update(s, s > 0.25) for s in (0.4, 0.0, 0.0, 0.0)]
    assert seen == [None] * 4                                   # a real wake is not a near miss


def test_console_listen_uses_the_shared_loop(monkeypatch, tmp_path):
    seen = {}

    class FakeAssistant:
        def run(self):
            raise KeyboardInterrupt

    def fake_build(audit, get_flow, voice, **kw):
        seen.update(kw)
        return FakeAssistant()

    monkeypatch.setattr(cli, "build_assistant", fake_build)
    lines = []
    listen = cli._make_listener(AuditLog(tmp_path / "log.jsonl"), out=lines.append)()
    with pytest.raises(KeyboardInterrupt):
        listen()
    assert seen["out"] == lines.append and "show_text" not in seen    # the console prints instead of popups
    seen["on_status"](cli.Status.LISTENING, "")
    seen["on_status"](cli.Status.LISTENING, "")
    seen["on_status"](cli.Status.PROBLEM, "I couldn't open the microphone.")
    assert lines[0].startswith("Listening.") and len(lines) == 2
    assert lines[1] == "(problem: I couldn't open the microphone.)"


# ---- weak tries, second chance, near misses, and the self-test (14 Nov) ----------------------

class FakeSpeaker:
    """Stands in for AllChecks with your voice check: passes when the voice score reaches 0.45 + extra."""
    has_speaker = True

    def __init__(self, voice=0.55, speech=0.4):
        self.voice, self.speech, self.calls = voice, speech, []
        self.failed, self.score = None, None

    def __call__(self, chunks, extra=0.0):
        self.calls.append(extra)
        self.score = self.voice
        ok = self.voice >= 0.45 + extra
        self.failed = None if ok else "not your voice"
        return ok

    def diagnose(self, chunks):
        return {"speech": self.speech, "speech_needed": 0.25, "voice": self.voice, "bar": 0.45}


class Audit:
    def __init__(self):
        self.events = []

    def record(self, event, **fields):
        self.events.append((event, fields))


def weak_triggers(scores, check, audit=None, clock=None):
    now = [0.0]
    t = Triggers(FakeWake(scores), threshold=0.25, clock=clock or (lambda: now[0]), voice_check=check, audit=audit,
                 weak_threshold=0.15, weak_extra=0.05, second_chance_seconds=5.0, second_chance_threshold=0.08)
    return t, now


def run(t, now, n):
    events = []
    for _ in range(n):
        now[0] += 0.08
        e = t.process(CH)
        if e is not None:
            events.append(e)
    return events


def test_a_weak_try_wakes_when_the_voice_passes_with_room_to_spare():
    check = FakeSpeaker(voice=0.55)
    t, now = weak_triggers([0.1, 0.2, 0.18, 0.0, 0.0, 0.0, 0.0], check)
    events = run(t, now, 7)
    assert [(e.source, e.score, e.weak) for e in events] == [("hey jarvis", 0.2, True)]
    assert check.calls == [0.05]                               # the voice check was asked for 0.45 + 0.05


def test_a_weak_try_in_a_voice_just_over_the_pass_mark_is_refused():
    check = FakeSpeaker(voice=0.47)                            # enough for a strong try, not a weak one
    audit = Audit()
    t, now = weak_triggers([0.2, 0.0, 0.0, 0.0], check, audit=audit)
    assert run(t, now, 4) == []
    assert audit.events == [("wake_rejected", {"source": "hey jarvis", "score": 0.2, "reason": "not your voice",
                                               "weak": True, "voice_score": 0.47})]


def test_weak_tries_need_your_voice_check():
    plain = lambda chunks: True                                  # no voiceprint: only the speech check
    t, now = weak_triggers([0.2, 0.0, 0.0, 0.0], plain)
    assert run(t, now, 4) == []


def test_a_strong_try_is_decided_at_once_and_not_again_at_its_end():
    check = FakeSpeaker(voice=0.55)
    t, now = weak_triggers([0.1, 0.3, 0.2, 0.0, 0.0, 0.0], check)
    events = run(t, now, 6)
    assert [(e.score, e.weak) for e in events] == [(0.3, False)] and check.calls == [0.0]


def test_second_chance_after_a_near_miss():
    check = FakeSpeaker(voice=0.6)
    scores = [0.1, 0.0, 0.0, 0.0] + [0.0] * 5 + [0.11, 0.0, 0.0, 0.0]   # 0.10 misses; a 0.11 repeat counts
    t, now = weak_triggers(scores, check)
    events = run(t, now, len(scores))
    assert [(e.score, e.weak) for e in events] == [(0.11, True)]


def test_no_second_chance_after_it_closes():
    check = FakeSpeaker(voice=0.6)
    scores = [0.1, 0.0, 0.0, 0.0] + [0.0] * 70 + [0.11, 0.0, 0.0, 0.0]  # the repeat comes ~6 s later
    t, now = weak_triggers(scores, check)
    assert run(t, now, len(scores)) == []


def test_near_misses_are_noted_with_the_next_wake_up():
    check = FakeSpeaker(voice=0.6)
    audit = Audit()
    scores = [0.07, 0.0, 0.0, 0.0] + [0.0] * 10 + [0.3]
    t, now = weak_triggers(scores, check, audit=audit)
    run(t, now, len(scores))
    event, fields = audit.events[-1]
    assert event == "wake" and fields["score"] == 0.3
    assert fields["tries_before"] == [[0.9, 0.07, "low"]]      # seconds since that try ended, score, why


def test_the_self_test_measures_every_try_and_wakes_nothing():
    check = FakeSpeaker(voice=0.55, speech=0.4)
    t, now = weak_triggers([0.3, 0.0, 0.0, 0.0, 0.18, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], check)
    tries, done = [], []
    t.start_test(tries.append, done.append, tries=3, seconds=60)
    assert run(t, now, 12) == []                               # nothing woke during the test
    assert [r["verdict"] for r in tries] == ["passed", "passed (weak)", "hey jarvis score too low"]
    assert tries[1]["voice_needed"] == 0.5 and tries[0]["voice_needed"] == 0.45
    assert done == [{"count": 3, "passed": 2, "stops": {"hey jarvis score too low": 1}}]
    assert t.testing is False


def test_the_self_test_stops_after_its_time():
    t, now = weak_triggers([0.0] * 20, FakeSpeaker())
    done = []
    t.start_test(lambda r: None, done.append, tries=10, seconds=1.0)
    run(t, now, 20)
    assert done == [{"count": 0, "passed": 0, "stops": {}}]


def test_weak_settings_are_checked():
    for bad in ({"weak_threshold": 0.3}, {"weak_threshold": 0.0}, {"second_chance_threshold": 0.25},
                {"weak_extra": 1.0}):
        with pytest.raises(ValueError):
            Triggers(threshold=0.25, **bad)


def test_config_wake_settings():
    assert 0 < config.WAKE_SECOND_CHANCE_THRESHOLD < config.WAKE_WEAK_THRESHOLD < config.WAKE_THRESHOLD
    assert 0 < config.WAKE_WEAK_EXTRA <= 0.1 and 0 < config.WAKE_SECOND_CHANCE_SECONDS <= 10
