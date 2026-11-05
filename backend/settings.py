"""What the Settings page may ask for, and nothing more.

The screen can only press buttons: each command is a bare name from COMMANDS, with no data
attached. Everything happens here, in the background program, and the screen is told only
what to show (progress and status, never audio or voice numbers).

- teach_start / teach_record / teach_redo / teach_cancel: teach your voice, one sample at a time
- voice_test: say "hey jarvis" once and see your score
- voice_forget: delete your voiceprint. This weakens security, so it needs the same Y-keypress
  confirmation window as deleting a file
- voice_status: what the Voice section should show
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from . import config
from .voice.voiceprint import (AllChecks, SpeakerCheck, StrictCheck, Voiceprint, VoiceprintError,
                               delete_voiceprint, load_voiceprint, save_voiceprint)

log = logging.getLogger("jarvis")

COMMANDS = ("teach_start", "teach_record", "teach_redo", "teach_cancel", "voice_test", "voice_forget",
            "voice_status")
SPEAKER = "not your voice"                       # the name of your voice check among the wake checks


class VoiceSettings:
    def __init__(self, get_assistant: Callable[[], Any], show_teach: Callable[[dict], None],
                 show_voice: Callable[[dict], None], confirm: Callable[[str, dict], bool],
                 get_secret: Callable, set_secret: Callable, delete_secret: Callable,
                 make_session: Callable[..., Any], model_factory: Callable[[Any], Any] | None = None,
                 voices: Callable[[], list] | None = None):
        self._assistant, self._show_teach, self._show_voice = get_assistant, show_teach, show_voice
        self._confirm = confirm
        self._get, self._set, self._delete = get_secret, set_secret, delete_secret
        self._make_session, self._model_factory, self._voices = make_session, model_factory, voices
        self._session: Any = None
        self._lock = threading.Lock()

    # ---- commands from the screen ----------------------------------------------------------
    def handle(self, command: Any) -> None:
        if command not in COMMANDS:
            return                                # anything else from the screen is dropped
        try:
            getattr(self, "_" + command)()
        except Exception as exc:                  # never let a button crash the program
            log.error("settings command failed: %s", type(exc).__name__)

    def _teach_start(self) -> None:
        self._start("teach")

    def _voice_test(self) -> None:
        self._start("test")

    def _start(self, mode: str) -> None:
        with self._lock:
            if self._session is not None and self._session.active:
                return
            current = self._load()
            self._session = self._make_session(save=self.install, pause_listening=self._pause,
                                               on_update=self._show_teach, current=current)
            self._session.start(mode)

    def _teach_record(self) -> None:
        if self._session is not None:
            self._session.record()

    def _teach_redo(self) -> None:
        if self._session is not None:
            self._session.redo()

    def _teach_cancel(self) -> None:
        if self._session is not None:
            self._session.cancel()

    def _voice_forget(self) -> None:
        if self._session is not None and self._session.active:
            return
        if not config.VOICEPRINT.is_file():
            self._voice_status()
            return
        if self._confirm("forget_my_voice", {"what": "your voiceprint (anyone could wake J.A.R.V.I.S again)"}) is not True:
            self._show_voice({**self.status(), "message": "Kept your voiceprint."})
            return
        delete_voiceprint(config.VOICEPRINT, self._delete)
        self._swap_check(None)
        self._show_voice({**self.status(), "message": "Your voiceprint was deleted."})

    def _voice_status(self) -> None:
        self._show_voice(self.status())

    def check_idle(self) -> None:
        if self._session is not None:
            self._session.check_idle()

    # ---- the voice check itself ----------------------------------------------------------------
    def _pause(self, teaching: bool) -> None:
        assistant = self._assistant()
        if assistant is not None:
            assistant.set_teaching(teaching)

    def _load(self) -> Voiceprint | None:
        try:
            return load_voiceprint(config.VOICEPRINT, self._get)
        except VoiceprintError:
            return None

    def install(self, voiceprint: Voiceprint) -> None:
        """Save a newly taught voiceprint and switch the check on straight away."""
        save_voiceprint(voiceprint, config.VOICEPRINT, self._set)
        if config.VOICE_CHECK and self._model_factory is not None:
            check = SpeakerCheck(self._model_factory(config.MODELS_DIR), voiceprint)
            self._swap_check(StrictCheck(check) if config.VOICE_CHECK_STRICT else check)
        self._voice_status()

    def _swap_check(self, speaker: Any) -> None:
        """Put your voice check in (or take it out of) the running wake-up checks."""
        assistant = self._assistant()
        if assistant is None:
            return
        current = assistant.voice_check
        checks = [(n, c) for n, c in getattr(current, "_checks", []) if n != SPEAKER]
        if current is not None and not hasattr(current, "_checks"):
            checks = [("no voice", current)]
        if speaker is not None:
            checks.append((SPEAKER, speaker))
        assistant.set_voice_check(AllChecks(checks) if checks else None)

    def status(self) -> dict:
        """What the Voice section shows. Numbers and dates only."""
        try:
            vp = load_voiceprint(config.VOICEPRINT, self._get)
        except VoiceprintError as exc:
            return {"taught": False, "problem": str(exc), "on": False}
        if vp is None:
            return {"taught": False, "on": False}
        assistant = self._assistant()
        names = [n for n, _ in getattr(getattr(assistant, "voice_check", None), "_checks", [])]
        return {"taught": True, "created": vp.created, "samples": vp.samples, "bar": round(vp.threshold, 2),
                "on": SPEAKER in names and config.VOICE_CHECK, "strict": config.VOICE_CHECK_STRICT}
