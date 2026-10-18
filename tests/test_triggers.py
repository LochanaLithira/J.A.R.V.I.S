import hashlib
import shutil
import subprocess
import sys
import threading
import wave

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


@pytest.fixture(scope="session")
def models_dir():
    try:
        verify_models(config.MODELS_DIR)
    except TriggerError:
        pytest.skip("wake-word models not downloaded (python -m backend.voice.triggers download)")
    return config.MODELS_DIR


@pytest.fixture(scope="session")
def spoken(tmp_path_factory):
    """Phrases spoken by Windows' built-in offline voice, as 16 kHz WAV files."""
    if sys.platform != "win32":
        pytest.skip("needs Windows text-to-speech")
    out = tmp_path_factory.mktemp("spoken")
    phrases = {"wake": "Hey Jarvis", "other": "I was talking to Travis about the weather yesterday"}
    lines = ["Add-Type -AssemblyName System.Speech",
             "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
             "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer"]
    for name, text in phrases.items():
        lines += [f"$s.SetOutputToWaveFile('{out / (name + '.wav')}', $f)", f"$s.Speak('{text}')"]
    lines.append("$s.Dispose()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "; ".join(lines)],
                       check=True, capture_output=True, timeout=90)
    except Exception:
        pytest.skip("Windows text-to-speech is not available")

    def load(name):
        with wave.open(str(out / f"{name}.wav")) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        pad = np.zeros(16000, np.int16)
        return np.concatenate([pad, audio, pad])

    return {name: load(name) for name in phrases}


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
    assert download_models(tmp_path, fetch) == [] and len(asked) == 3        # nothing fetched twice


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
    near = cli._NearMisses(0.25)
    seen = [near.update(s, False) for s in (0.0, 0.18, 0.2, 0.0, 0.0, 0.0)]
    assert seen[-1] == 0.2 and seen[:-1] == [None] * 5
    seen = [near.update(s, s > 0.25) for s in (0.4, 0.0, 0.0, 0.0)]
    assert seen == [None] * 4                                   # a real wake is not a near miss


def test_listener_wiring_chimes_and_prints(monkeypatch, tmp_path):
    audio = slices(np.zeros(CHUNK * 3))

    class Mic:
        name = "Test Mic"

        def __init__(self, device):
            assert device == 5

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def chunks(self):
            yield from audio
            raise KeyboardInterrupt

    chimes = []
    monkeypatch.setattr(cli, "WakeWordDetector", lambda d: FakeWake([0.0, 0.9, 0.0]))
    monkeypatch.setattr(cli, "ClapDetector", lambda *a, **k: FakeClap([False] * 3))
    monkeypatch.setattr(cli, "pick_microphone", lambda name: 5)
    monkeypatch.setattr(cli, "Microphone", Mic)
    monkeypatch.setattr(cli, "play_chime", lambda: chimes.append(1))
    lines = []
    audit = AuditLog(tmp_path / "log.jsonl")
    listen = cli._make_listener(audit, out=lines.append)()
    with pytest.raises(KeyboardInterrupt):
        listen()
    assert chimes == [1]
    assert lines[0].startswith("Listening on Test Mic") and "WAKE: hey jarvis (score 0.90)" in lines
    assert "wake" in audit.path.read_text()
