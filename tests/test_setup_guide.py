"""The setup guide: preferences, the guide's buttons, the key box and the microphone list."""
import io
import json

import pytest

from backend import preferences
from backend.credentials import CredentialError
from backend.setup_guide import CHOICES, COMMANDS, KEYS, SetupGuide, mic_levels
from backend.voice.triggers import list_microphones
from frontend import keybox


# ---------------------------------------------------------------------------
# preferences.json
# ---------------------------------------------------------------------------

def test_preferences_round_trip(tmp_path):
    path = tmp_path / "p.json"
    assert preferences.load(path) == {"setup_done": False}
    preferences.update(path, mic="Headset", setup_done=True)
    assert preferences.load(path) == {"mic": "Headset", "setup_done": True}
    assert preferences.microphone(path) == "Headset"
    preferences.update(path, mic="")
    assert preferences.microphone(path) is None                  # "" = the Windows default


def test_preferences_fall_back_to_config(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences.config, "MIC_DEVICE", "Microphone Array")
    assert preferences.microphone(tmp_path / "none.json") == "Microphone Array"


@pytest.mark.parametrize("content", ["not json", "[1, 2]", json.dumps({"mic": 5, "setup_done": "yes"}),
                                     json.dumps({"mic": "a\nb"}), json.dumps({"mic": "x" * 500}), "x" * 5000])
def test_bad_preferences_are_ignored(tmp_path, content):
    path = tmp_path / "p.json"
    path.write_text(content, encoding="utf-8")
    prefs = preferences.load(path)
    assert "mic" not in prefs and prefs["setup_done"] is False


def test_preferences_never_save_a_bad_name(tmp_path):
    with pytest.raises(ValueError):
        preferences.save({"mic": "bad\x00name"}, tmp_path / "p.json")
    assert not (tmp_path / "p.json").exists()


# ---------------------------------------------------------------------------
# The guide
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, **over):
        self.secrets, self.shown, self.opened, self.saved_keys = {}, [], [], []
        self.prefs = {"setup_done": False}
        self.mic = "Microphone Array"
        self.listening, self.started, self.paused = False, 0, []
        self.models, self.levels = False, []
        self.key_answer = over.pop("key_answer", "k" * 20)
        self.asked = []

        def set_secret(name, value):
            if " " in value:
                raise CredentialError("That key looks invalid (8-256 visible characters, no spaces).")
            self.secrets[name] = value

        def start():
            self.started += 1
            self.listening = True
            return ""

        def download():
            self.models = True

        args = dict(get_secret=self.secrets.get, set_secret=set_secret,
                    ask_key=lambda s: (self.asked.append(s), self.key_answer)[1],
                    on_key_saved=self.saved_keys.append, open_url=self.opened.append,
                    key_check=lambda name, key: "works", voice_check=lambda: (True, ""),
                    models_ready=lambda: self.models, download_models=download,
                    list_mics=lambda: ["Microphone Array (Realtek)", "Headset (USB)"],
                    current_mic=lambda: self.mic, save_mic=self._save_mic,
                    mic_levels=lambda name, seconds: iter([0.05, 0.4, 0.1]),
                    voice_taught=lambda: False, autostart=lambda action: (True, "Autostart is OFF."),
                    start_listening=start, is_listening=lambda: self.listening,
                    pause=self.paused.append, load_prefs=lambda: dict(self.prefs),
                    update_prefs=lambda **c: self.prefs.update(c), level=self.levels.append, background=False)
        args.update(over)
        self.guide = SetupGuide(self.shown.append, **args)

    def _save_mic(self, name):
        self.mic = name or None

    @property
    def last(self):
        return self.shown[-1]


def test_status_shows_only_whether_keys_are_set():
    h = Harness()
    h.secrets["gemini_api_key"] = "secret-value-123"
    h.guide.handle("setup_status")
    state = h.last
    assert state["keys"] == {"gemini": True, "groq": False, "fish": False}
    assert "secret-value-123" not in json.dumps(state)
    assert state["mics"] == ["Windows default", "Microphone Array (Realtek)", "Headset (USB)"]
    assert state["mic"] == 1 and state["autostart"] == "off" and state["models"] is False


def test_adding_a_key_goes_from_the_box_to_the_vault():
    h = Harness()
    h.guide.handle("setup_key_groq")
    assert h.asked == ["groq"] and h.secrets == {"stt_api_key": "k" * 20}
    assert h.saved_keys == ["stt_api_key"]
    assert h.last["keys"]["groq"] is True and h.last["busy"] == ""
    assert "k" * 20 not in json.dumps(h.shown)


def test_a_cancelled_or_bad_key_is_not_saved():
    h = Harness(key_answer=None)
    h.guide.handle("setup_key_gemini")
    assert h.secrets == {} and h.last["notes"]["keys"]["kind"] == "warn"
    h = Harness(key_answer="has a space")
    h.guide.handle("setup_key_gemini")
    assert h.secrets == {} and "invalid" in h.last["notes"]["keys"]["text"]
    assert "has a space" not in json.dumps(h.shown)


def test_sign_up_links_are_fixed():
    h = Harness()
    for service in KEYS:
        h.guide.handle(f"setup_get_{service}")
    assert h.opened == [url for _, url in KEYS.values()]
    assert all(url.startswith("https://") for url in h.opened)


def test_unknown_commands_and_bad_choices_are_dropped():
    h = Harness()
    for cmd in ("setup_get_evil", "setup_key_x", "rm -rf", None, 5):
        h.guide.handle(cmd)
    for bad in (-1, 17, "1", True, 1.0, None):
        h.guide.handle("setup_mic", bad)
    h.guide.handle("setup_status", 3)          # a bare button ignores data
    assert h.opened == [] and h.asked == [] and h.mic == "Microphone Array"


def test_a_microphone_is_picked_from_the_list_sent():
    h = Harness()
    h.guide.handle("setup_mic", 2)               # before any list was sent: not in it
    assert h.mic == "Microphone Array"
    h.guide.handle("setup_status")
    h.guide.handle("setup_mic", 2)
    assert h.mic == "Headset (USB)" and h.started == 1 and h.last["mic"] == 2
    h.guide.handle("setup_mic", 3)               # past the end of the list
    assert h.mic == "Headset (USB)"
    h.guide.handle("setup_mic", 0)
    assert h.mic is None and h.last["mic"] == 0


def test_a_missing_configured_mic_shows_as_not_found():
    h = Harness()
    h.mic = "Not plugged in"
    h.guide.handle("setup_status")
    assert h.last["mic"] == -1


def test_mic_test_pauses_listening_and_reports_the_level():
    h = Harness()
    h.guide.handle("setup_mic_test")
    assert h.paused == [True, False]
    assert h.levels == [0.05, 0.4, 0.1, 0.0]
    assert h.last["notes"]["mic"] == {"text": "I heard you clearly.", "kind": "ok"}
    quiet = Harness(mic_levels=lambda name, seconds: iter([0.01, 0.02]))
    quiet.guide.handle("setup_mic_test")
    assert quiet.last["notes"]["mic"]["kind"] == "bad"


def test_testing_keys():
    h = Harness(key_check=lambda name, key: "refused" if name == "stt_api_key" else "works")
    h.guide.handle("setup_test_keys")
    assert h.last["notes"]["keys"]["text"] == "Add a key first."
    h.secrets.update(gemini_api_key="g" * 20, stt_api_key="s" * 20, fish_api_key="f" * 20)
    h.guide.handle("setup_test_keys")
    assert h.last["key_notes"] == {"gemini": "works", "groq": "refused", "fish": "works"}
    assert h.last["notes"]["keys"]["kind"] == "warn"


def test_downloading_models_starts_listening():
    h = Harness()
    h.guide.handle("setup_models")
    assert h.models and h.started == 1 and h.last["models"] is True


def test_a_failed_download_says_so():
    from backend.voice.triggers import TriggerError

    def fail():
        raise TriggerError("The downloaded x did not match its fingerprint, so it was not saved.")

    h = Harness(download_models=fail)
    h.guide.handle("setup_models")
    assert "fingerprint" in h.last["notes"]["models"]["text"] and h.started == 0


def test_autostart_uses_only_on_and_off():
    calls = []
    h = Harness(autostart=lambda action: (calls.append(action), (True, "Autostart is ON."))[1])
    h.guide.handle("setup_autostart_on")
    h.guide.handle("setup_autostart_off")
    assert {c for c in calls} <= {"on", "off", "status"} and "on" in calls and "off" in calls


def test_finishing_marks_setup_done_and_starts_listening():
    h = Harness()
    assert h.guide.needed()
    h.guide.handle("setup_finish")
    assert h.prefs["setup_done"] is True and h.started == 1
    # still needed while a key or the models are missing
    assert h.guide.needed()
    h.models = True
    h.secrets.update({name: "x" * 20 for name, _ in KEYS.values()})
    assert not h.guide.needed()


def test_the_guide_opens_only_when_asked():
    h = Harness()
    h.guide.screen_ready()
    assert h.shown == []
    h.guide.request_open()
    h.guide.screen_ready()
    assert h.last["open"] is True
    h.guide.screen_ready()
    assert len(h.shown) == 1


def test_a_crashing_step_never_escapes():
    def boom():
        raise RuntimeError("x")

    h = Harness(download_models=boom)
    h.guide.handle("setup_models")
    assert h.last["busy"] == ""


def test_the_main_program_can_build_the_guide():
    from backend import main

    class Screen:
        def setup(self, state):
            pass

        def set_level(self, level):
            pass

    guide = main.make_setup_guide(Screen(), lambda: None, lambda: "", lambda name: None)
    assert isinstance(guide, SetupGuide)
    assert isinstance(main.models_ready(), bool)


def test_backend_lists_match():
    assert set(CHOICES) == {"setup_mic"} and all(c.startswith("setup_") for c in COMMANDS)


def test_mic_levels_closes_after_the_time(monkeypatch):
    class Mic:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            Mic.closed = True

        def chunks(self, stop, idle_timeout=None):
            while not stop.is_set():
                yield [1]

    clock = iter(range(100))
    monkeypatch.setattr("backend.setup_guide.time.monotonic", lambda: next(clock))
    levels = list(mic_levels("x", 3, pick=lambda n: 1, mic_factory=lambda d: Mic(), level=lambda c: 0.5))
    assert levels == [0.5] * 3 and Mic.closed


# ---------------------------------------------------------------------------
# The microphone list
# ---------------------------------------------------------------------------

def test_microphone_list():
    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    devices = [{"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
               {"name": "Microphone Array (Realtek(R) Au", "hostapi": 0, "max_input_channels": 2},
               {"name": "Speakers", "hostapi": 0, "max_input_channels": 0},
               {"name": "Microphone Array (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2},
               {"name": "Microphone Array (Realtek(R) Au", "hostapi": 0, "max_input_channels": 1}]
    assert list_microphones(devices, hostapis) == ["Microphone Array (Realtek(R) Au"]


# ---------------------------------------------------------------------------
# The key box
# ---------------------------------------------------------------------------

def test_key_box_child_prints_only_a_given_key():
    out = io.StringIO()
    assert keybox._child("gemini", show=lambda s: "abc12345", out=out) == 0 and out.getvalue() == "abc12345"
    for answer in (None, "", "a\nb", "x" * 300):
        out = io.StringIO()
        assert keybox._child("gemini", show=lambda s, a=answer: a, out=out) == 1 and out.getvalue() == ""
    assert keybox._child("evil", show=lambda s: "abc12345") == 2


def test_key_box_window_failure_means_no_key():
    def crash(service):
        raise RuntimeError("no screen")

    assert keybox._child("groq", show=crash, out=io.StringIO()) == 3


def test_ask_key_needs_a_clean_answer():
    assert keybox.ask_key("fish", runner=lambda s: (0, "abc12345\n")) == "abc12345"
    assert keybox.ask_key("fish", runner=lambda s: (1, "abc12345")) is None
    assert keybox.ask_key("fish", runner=lambda s: (0, "")) is None
    assert keybox.ask_key("nope", runner=lambda s: (0, "abc12345")) is None

    def boom(s):
        raise TimeoutError

    assert keybox.ask_key("fish", runner=boom) is None


def test_key_box_services_match_the_guide():
    assert set(keybox.SERVICES) == set(KEYS)
