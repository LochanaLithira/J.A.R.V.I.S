"""The setup guide on the screen: keys, models, microphone, your voice, starting with Windows.

Like the Settings page, the screen can only press buttons. Every command is a bare name from
COMMANDS, except one: "setup_mic" carries a single small number, the position of a microphone in
the list J.A.R.V.I.S itself sent. It is range-checked by the window, by the app's side of the
pipe and again here against that same list, so it can only ever pick one of those microphones.

- Keys never pass through the screen. "Add key" opens the key box (frontend/keybox.py), a native
  window in its own process; the key comes back over a private pipe and goes straight into the
  Windows Credential Manager. The screen only learns "set" or "not set".
- "Get a free key" opens one of three fixed sign-up addresses written below.
- "Test" makes one small read-only call to Gemini and Groq (listing models: no tokens used), and
  has Fish Audio say one short line.
- Models are downloaded over HTTPS and saved only if their fingerprints match.
- The microphone test listens for a few seconds and shows the level. Nothing is recorded or kept.
- Starting with Windows runs scripts/install_autostart.ps1 with its fixed switches only.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .credentials import CredentialError

log = logging.getLogger("jarvis")

# service id on the screen -> (Credential Manager name, where to get a free key)
KEYS = {
    "gemini": ("gemini_api_key", "https://aistudio.google.com/apikey"),
    "groq": ("stt_api_key", "https://console.groq.com/keys"),
    "fish": ("fish_api_key", "https://fish.audio/app/api-keys/"),
}
COMMANDS = ("setup_status", "setup_key_gemini", "setup_key_groq", "setup_key_fish", "setup_get_gemini",
            "setup_get_groq", "setup_get_fish", "setup_test_keys", "setup_models", "setup_mic_test",
            "setup_autostart_on", "setup_autostart_off", "setup_finish")
CHOICES = {"setup_mic": 17}                      # Windows default + up to 16 microphones
MIC_TEST_SECONDS = 4.0
HEARD_LEVEL, QUIET_LEVEL = 0.25, 0.1
_ROOT = Path(__file__).resolve().parents[1]
AUTOSTART_SCRIPT = _ROOT / "scripts" / "install_autostart.ps1"


# ---------------------------------------------------------------------------
# The real checks (replaced by fakes in the tests)
# ---------------------------------------------------------------------------

_KEY_CHECKS = {
    "gemini_api_key": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", "x-goog-api-key", ""),
    "stt_api_key": ("https://api.groq.com/openai/v1/models", "Authorization", "Bearer "),
}


def check_key(name: str, key: str) -> str:
    """One read-only request with the key: "works", "refused" or "unreachable"."""
    import httpx

    url, header, prefix = _KEY_CHECKS[name]
    try:
        response = httpx.get(url, headers={header: prefix + key}, timeout=10.0)
    except Exception:
        return "unreachable"
    if response.status_code == 200:
        return "works"
    return "refused" if response.status_code in (400, 401, 403) else "unreachable"


def run_autostart(action: str, script: Path = AUTOSTART_SCRIPT) -> tuple[bool, str]:
    """action: "on", "off" or "status". Returns (it worked, the script's first line)."""
    flags = {"on": [], "off": ["-Remove"], "status": ["-Status"]}[action]
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *flags],
                              capture_output=True, text=True, timeout=60,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return False, "Windows PowerShell couldn't be started."
    text = (done.stdout.strip() or done.stderr.strip()).splitlines()
    return done.returncode == 0, (text[0][:200] if text else "")


# ---------------------------------------------------------------------------
# The guide
# ---------------------------------------------------------------------------

class SetupGuide:
    def __init__(self, show: Callable[[dict], None], *, get_secret: Callable, set_secret: Callable,
                 ask_key: Callable[[str], str | None], on_key_saved: Callable[[str], None],
                 open_url: Callable[[str], Any], key_check: Callable[[str, str], str] = check_key,
                 voice_check: Callable[[], tuple[bool, str]],
                 models_ready: Callable[[], bool], download_models: Callable[[], None],
                 list_mics: Callable[[], list[str]], current_mic: Callable[[], str | None],
                 save_mic: Callable[[str], None], mic_levels: Callable[[str | None, float], Any],
                 voice_taught: Callable[[], bool], autostart: Callable[[str], tuple[bool, str]] = run_autostart,
                 start_listening: Callable[[], str], is_listening: Callable[[], bool],
                 pause: Callable[[bool], None], load_prefs: Callable[[], dict], update_prefs: Callable[..., Any],
                 level: Callable[[float], None] = lambda x: None, background: bool = True):
        self._show = show
        self._get, self._set, self._ask_key, self._on_key_saved = get_secret, set_secret, ask_key, on_key_saved
        self._open_url, self._key_check, self._voice_check = open_url, key_check, voice_check
        self._models_ready, self._download = models_ready, download_models
        self._list_mics, self._current_mic, self._save_mic, self._mic_levels = list_mics, current_mic, save_mic, mic_levels
        self._voice_taught, self._autostart = voice_taught, autostart
        self._start_listening, self._is_listening, self._pause = start_listening, is_listening, pause
        self._load_prefs, self._update_prefs, self._level = load_prefs, update_prefs, level
        self._background = background
        self._mics: list[str] = []              # the list last sent to the screen
        self._notes: dict[str, dict] = {}       # a short message per step
        self._key_notes: dict[str, str] = {}    # "works" / "refused" / ... per key
        self._busy = ""                          # what is running now ("" = nothing)
        self._lock = threading.Lock()
        self._open_next = False

    # ---- commands from the screen ----------------------------------------------------------
    def handle(self, command: Any, arg: Any = None) -> None:
        try:
            if command in CHOICES:
                if type(arg) is int and 0 <= arg < CHOICES[command]:
                    self._choose_mic(arg)
                return
            if command not in COMMANDS:
                return                            # anything else from the screen is dropped
            if command == "setup_status":
                self._work("", self.refresh)
            elif command.startswith("setup_key_"):
                self._work("key", self._add_key, command.removeprefix("setup_key_"))
            elif command.startswith("setup_get_"):
                self._open_url(KEYS[command.removeprefix("setup_get_")][1])
            elif command == "setup_test_keys":
                self._work("test", self._test_keys)
            elif command == "setup_models":
                self._work("models", self._get_models)
            elif command == "setup_mic_test":
                self._work("mic", self._test_mic)
            elif command in ("setup_autostart_on", "setup_autostart_off"):
                self._work("autostart", self._set_autostart, command.removeprefix("setup_autostart_"))
            elif command == "setup_finish":
                self._work("finish", self._finish)
        except Exception as exc:                  # never let a button crash the program
            log.error("setup command failed: %s", type(exc).__name__)

    def _work(self, name: str, fn: Callable, *args: Any) -> None:
        """Run one slow step in the background; a second slow step waits for the first to end."""
        if name:
            with self._lock:
                if self._busy:
                    return
                self._busy = name
            self.refresh()

        def run() -> None:
            try:
                fn(*args)
            except Exception as exc:
                log.error("setup step failed: %s", type(exc).__name__)
                self._note(name or "general", "Something went wrong. Please try again.", "bad")
            finally:
                if name:
                    with self._lock:
                        self._busy = ""
                self.refresh()

        if self._background:
            threading.Thread(target=run, name="jarvis-setup", daemon=True).start()
        else:
            run()

    def _note(self, step: str, text: str, kind: str = "ok") -> None:
        self._notes[step] = {"text": text, "kind": kind}

    # ---- keys ----------------------------------------------------------------------------------
    def _add_key(self, service: str) -> None:
        name = KEYS[service][0]
        key = self._ask_key(service)
        if key is None:
            self._note("keys", "No key was added.", "warn")
            return
        try:
            self._set(name, key)
        except CredentialError as exc:
            self._note("keys", str(exc), "bad")
            return
        finally:
            key = None                            # drop our copy as soon as it is stored
        self._key_notes.pop(service, None)
        self._on_key_saved(name)
        self._note("keys", "Saved in the Windows Credential Manager.")

    def _has(self, name: str) -> bool:
        try:
            return bool(self._get(name))
        except CredentialError:
            return False

    def _test_keys(self) -> None:
        tested = 0
        for service, (name, _) in KEYS.items():
            if not self._has(name):
                continue
            tested += 1
            if service == "fish":
                ok, problem = self._voice_check()
                self._key_notes[service] = "works" if ok else (problem or "refused")
            else:
                self._key_notes[service] = self._key_check(name, self._get(name))
        if not tested:
            self._note("keys", "Add a key first.", "warn")
        elif all(self._key_notes.get(s) == "works" for s, (n, _) in KEYS.items() if self._has(n)):
            self._note("keys", "Every key you added works.")
        else:
            self._note("keys", "Some keys didn't work. Add them again, or check your internet connection.", "warn")

    # ---- models ----------------------------------------------------------------------------------
    def _get_models(self) -> None:
        try:
            self._download()
        except Exception as exc:
            safe = str(exc) if exc.__class__.__name__ in ("TriggerError", "VoiceprintError") else ""
            self._note("models", safe or "The download failed. Check the internet connection.", "bad")
            return
        self._note("models", "Downloaded, and every file matched its fingerprint.")
        if not self._is_listening():
            problem = self._start_listening()
            if problem:
                self._note("mic", problem, "bad")

    # ---- microphone ----------------------------------------------------------------------------
    def _mic_index(self, mics: list[str]) -> int:
        """Where the microphone in use sits in the screen's list: 0 = Windows default, -1 = not found."""
        current = self._current_mic()
        if current is None:
            return 0
        for i, name in enumerate(mics, start=1):
            if name == current:
                return i
        for i, name in enumerate(mics, start=1):
            if current.lower() in name.lower():
                return i
        return -1

    def _choose_mic(self, index: int) -> None:
        if index > len(self._mics) or self._busy:
            return                                # not in the list we sent, or a step is running
        name = "" if index == 0 else self._mics[index - 1]
        self._save_mic(name)
        self._note("mic", "Saved. Now try it: press Test and say something.")
        self._work("mic", self._restart)

    def _restart(self) -> None:
        problem = self._start_listening()
        if problem:
            self._note("mic", problem, "bad")

    def _test_mic(self) -> None:
        self._pause(True)
        peak = 0.0
        try:
            for level in self._mic_levels(self._current_mic(), MIC_TEST_SECONDS):
                peak = max(peak, level)
                self._level(level)
        except Exception as exc:
            self._note("mic", str(exc) if exc.__class__.__name__ == "TriggerError" else
                       "That microphone couldn't be opened.", "bad")
            return
        finally:
            self._level(0.0)
            self._pause(False)
        if peak >= HEARD_LEVEL:
            self._note("mic", "I heard you clearly.")
        elif peak >= QUIET_LEVEL:
            self._note("mic", "I heard you, but quietly. Move closer, or try another microphone.", "warn")
        else:
            self._note("mic", "I heard nothing. Try another microphone.", "bad")

    # ---- starting with Windows ---------------------------------------------------------------
    def _set_autostart(self, action: str) -> None:
        ok, text = self._autostart(action)
        self._note("autostart", text or ("Done." if ok else "That didn't work."), "ok" if ok else "bad")

    # ---- finishing -------------------------------------------------------------------------------
    def _finish(self) -> None:
        self._update_prefs(setup_done=True)
        if not self._is_listening():
            problem = self._start_listening()
            if problem:
                self._note("done", problem, "bad")
                return
        self._note("done", "All set. Say \"hey jarvis\".")

    # ---- what the screen shows ----------------------------------------------------------------
    def needed(self) -> bool:
        """Open the guide by itself when setup was never finished or something essential is missing."""
        try:
            done = self._load_prefs().get("setup_done") is True
        except Exception:
            done = False
        return not done or not self._models_ready() or not all(self._has(n) for n, _ in KEYS.values())

    def request_open(self) -> None:
        """Open the guide the next time the screen is ready."""
        self._open_next = True

    def screen_ready(self) -> None:
        if self._open_next:
            self._open_next = False
            self.refresh(open_guide=True)

    def status(self) -> dict:
        try:
            mics = self._list_mics()
        except Exception:
            mics = []
        self._mics = list(mics)
        try:
            autostart = self._autostart_state()
        except Exception:
            autostart = "unknown"
        return {
            "keys": {s: self._has(n) for s, (n, _) in KEYS.items()},
            "key_notes": dict(self._key_notes),
            "models": bool(self._models_ready()),
            "mics": ["Windows default"] + self._mics,
            "mic": self._mic_index(self._mics),
            "voice": bool(self._voice_taught()),
            "autostart": autostart,
            "listening": bool(self._is_listening()),
            "busy": self._busy,
            "notes": dict(self._notes),
        }

    def _autostart_state(self) -> str:
        ok, text = self._autostart("status")
        if not ok:
            return "unknown"
        return "on" if "is ON" in text else "off"

    def refresh(self, open_guide: bool = False) -> None:
        state = self.status()
        if open_guide:
            state["open"] = True
        self._show(state)


def mic_levels(device_name: str | None, seconds: float, *, pick: Callable, mic_factory: Callable,
               level: Callable[[Any], float]) -> Any:
    """Open the microphone for a few seconds and yield the sound level of each slice. The sound
    itself is dropped at once: nothing is recorded or kept."""
    with mic_factory(pick(device_name)) as mic:
        end = time.monotonic() + seconds
        stop = threading.Event()
        for chunk in mic.chunks(stop, idle_timeout=3.0):
            yield level(chunk)
            if time.monotonic() >= end:
                stop.set()
                break
