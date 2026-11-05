"""Teach J.A.R.V.I.S your voice: say "hey jarvis" 20 times, and only numbers are kept.

    python -m backend.voice.enroll download   fetch the voice-check model (once, ~26 MB, fingerprint-checked)
    python -m backend.voice.enroll teach      teach it your voice (about 2 minutes)
    python -m backend.voice.enroll test       say "hey jarvis" and see your score
    python -m backend.voice.enroll status     is a voiceprint set up?
    python -m backend.voice.enroll delete     forget your voiceprint

Privacy: each "hey jarvis" is turned into 256 voice numbers the moment it is heard, and its sound
is dropped. The 20 recordings are never stored, never sent anywhere, and never even held together
in memory. Only the averaged numbers and the bar are saved (see voiceprint.py).
"""
from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np

from .. import config
from .other_voices import other_voices
from .voiceprint import (CHECK_SECONDS, Calibration, SpeakerModel, VoiceprintError, calibrate,
                         delete_voiceprint, download_speaker_model, load_voiceprint, normalize,
                         save_voiceprint, similarity, speaker_model_path)

SAMPLES = 20
MAX_TRIES = 40
WARM_UP_CHUNKS = 25               # ~2 s of listening before each chime, so the detector is ready
LISTEN_CHUNKS = 100               # then wait ~8 s for each "hey jarvis"
SAMPLE_MIN_MATCH = 0.35           # a sample far from the others (someone else, a bang) is retaken

TIPS = ("normally", "a little quieter", "a little louder", "a bit faster", "from a bit further away")


def collect_samples(listen_once: Callable[[], list | None], embed: Callable[[np.ndarray], np.ndarray],
                    say: Callable[[str], None] = print, want: int = SAMPLES,
                    max_tries: int = MAX_TRIES) -> list[np.ndarray]:
    """Gather `want` voice-number samples. listen_once() returns the sound of one wake moment
    (a list of slices) or None if nothing was heard in time. The sound is dropped right after its
    numbers are made."""
    kept: list[np.ndarray] = []
    for _ in range(max_tries):
        if len(kept) >= want:
            break
        tip = TIPS[(len(kept) // 4) % len(TIPS)]
        say(f"  {len(kept) + 1}/{want}: after the chime, say \"hey jarvis\" {tip}...")
        chunks = listen_once()
        if not chunks:
            say("     I didn't hear the wake word. Try again.")
            continue
        audio = np.concatenate([np.asarray(c, dtype=np.int16).reshape(-1) for c in chunks])
        chunks = None
        try:
            numbers = embed(audio[-int(CHECK_SECONDS * 16000):])
        except VoiceprintError:
            say("     That was too short. Try again.")
            continue
        finally:
            audio = None                                     # the sound is gone from here on
        if len(kept) >= 5:
            middle = normalize(np.mean(kept, axis=0))
            if similarity(numbers, middle) < SAMPLE_MIN_MATCH:
                say("     That didn't sound like the others (another voice or a noise?). Try again.")
                continue
        kept.append(numbers)
        say("     Got it.")
    return kept


def teach(listen_once: Callable[[], list | None], model: SpeakerModel,
          voices: Callable[[], list] = other_voices, say: Callable[[str], None] = print,
          save: Callable[[object], None] | None = None) -> Calibration:
    """The whole teaching run: your samples, other voices, the bar, and saving the voiceprint."""
    yours = collect_samples(listen_once, model.embed, say)
    if len(yours) < SAMPLES:
        raise VoiceprintError("Not enough clear samples. Try again somewhere quieter.")
    say("  Comparing with other voices (made by Windows, in memory)...")
    others = []
    for _name, clip in voices():
        try:
            others.append(model.embed(clip[: int(CHECK_SECONDS * 16000)]))
        except VoiceprintError:
            continue
    result = calibrate(yours, others)
    if result.voiceprint is not None and save is not None:
        save(result.voiceprint)
    return result


# ---------------------------------------------------------------------------
# The real microphone
# ---------------------------------------------------------------------------

def _wake_listener(mic, triggers, chime: Callable[[], None] | None = None, stop=None,
                   on_level: Callable[[Any], None] | None = None) -> Callable[[], list | None]:
    """Listen for one "hey jarvis". After a reset the wake-word detector needs ~1.5 s of sound
    before it can recognise anything, so it listens for WARM_UP_CHUNKS first and only then chimes.
    Counted in slices, not seconds, so it behaves the same however fast the sound arrives.
    `stop` (a threading.Event) ends it at once; `on_level` gets each slice for a sound meter."""
    if chime is None:
        from .triggers import play_chime as chime

    def listen_once() -> list | None:
        mic.drain()
        triggers.reset()
        heard = 0
        for chunk in mic.chunks(stop):
            if on_level is not None:
                on_level(chunk)
            heard += 1
            if heard <= WARM_UP_CHUNKS:
                triggers.process(chunk)                      # getting ready: nothing counts yet
                if heard == WARM_UP_CHUNKS:
                    chime()                                  # now say it
                continue
            if triggers.process(chunk) is not None:
                moment = triggers.recent(CHECK_SECONDS + 0.2)
                time.sleep(0.4)                              # let you finish before the next prompt
                return moment
            if heard > WARM_UP_CHUNKS + LISTEN_CHUNKS:
                return None
        return None

    return listen_once


def screen_prepare(on_level: Callable[[Any], None] | None = None) -> Callable[[], tuple[Callable, Callable]]:
    """For teaching from the screen: loads the model (downloading it once if needed) and returns
    (embed, listen). listen(stop, on_ready) opens the microphone for one sample only."""

    def prepare() -> tuple[Callable, Callable]:
        from .listen import SileroVAD, VoiceCheck
        from .triggers import Microphone, Triggers, WakeWordDetector, pick_microphone, play_chime

        download_speaker_model(config.MODELS_DIR)
        model = SpeakerModel(config.MODELS_DIR)
        voiced = VoiceCheck(SileroVAD(config.MODELS_DIR), config.WAKE_MIN_SPEECH_SECONDS,
                            config.WAKE_SPEECH_CONFIDENCE)
        triggers = Triggers(WakeWordDetector(config.MODELS_DIR), None, config.WAKE_THRESHOLD,
                            config.WAKE_PATIENCE, cooldown_seconds=0.5, voice_check=voiced)
        device = pick_microphone(config.MIC_DEVICE)

        def listen(stop, on_ready: Callable[[], None]) -> list | None:
            def chime() -> None:
                play_chime()
                on_ready()

            with Microphone(device) as mic:
                return _wake_listener(mic, triggers, chime=chime, stop=stop, on_level=on_level)()

        return model.embed, listen

    return prepare


def _open_listening():
    from ..main import acquire_single_instance
    from .listen import SileroVAD, VoiceCheck
    from .triggers import Microphone, Triggers, WakeWordDetector, pick_microphone

    probe = acquire_single_instance()
    if probe is None:
        print("J.A.R.V.I.S is running in the background. Right-click its tray icon and choose\n"
              "'Pause listening' first, so it doesn't answer each \"hey jarvis\".\n")
    voiced = VoiceCheck(SileroVAD(config.MODELS_DIR), config.WAKE_MIN_SPEECH_SECONDS, config.WAKE_SPEECH_CONFIDENCE)
    triggers = Triggers(WakeWordDetector(config.MODELS_DIR), None, config.WAKE_THRESHOLD, config.WAKE_PATIENCE,
                        cooldown_seconds=0.5, voice_check=voiced)
    return probe, Microphone(pick_microphone(config.MIC_DEVICE)), triggers


def _secrets():
    from ..credentials import delete_secret, get_secret, set_secret
    return get_secret, set_secret, delete_secret


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if len(argv) == 1 else ""
    if command not in ("download", "teach", "test", "status", "delete"):
        print(__doc__)
        return 2
    get_secret, set_secret, delete_secret = _secrets()
    try:
        if command == "download":
            print("Downloaded and checked." if download_speaker_model(config.MODELS_DIR) else "Already downloaded and checked.")
        elif command == "status":
            speaker_model_path(config.MODELS_DIR)
            vp = load_voiceprint(config.VOICEPRINT, get_secret)
            print("No voiceprint yet. Teach one with: python -m backend.voice.enroll teach" if vp is None else
                  f"Voiceprint from {vp.created}, {vp.samples} samples, bar {vp.threshold:.2f}. "
                  f"Voice check is {'on' if config.VOICE_CHECK else 'off (VOICE_CHECK in config.py)'}.")
        elif command == "delete":
            print("Voiceprint deleted." if delete_voiceprint(config.VOICEPRINT, delete_secret) else "There was no voiceprint.")
        elif command == "teach":
            download_speaker_model(config.MODELS_DIR)
            model = SpeakerModel(config.MODELS_DIR)
            probe, mic, triggers = _open_listening()
            print(f"Teaching J.A.R.V.I.S your voice: say \"hey jarvis\" {SAMPLES} times, after each chime.\n"
                  "Nothing is recorded: each one becomes numbers straight away.\n")
            with mic:
                result = teach(_wake_listener(mic, triggers), model,
                               save=lambda vp: save_voiceprint(vp, config.VOICEPRINT, set_secret))
            print("\n" + result.message)
            if result.voiceprint is not None:
                print("Saved. Restart J.A.R.V.I.S (or resume listening) to use it.")
            del probe
            return 0 if result.voiceprint is not None else 1
        elif command == "test":
            vp = load_voiceprint(config.VOICEPRINT, get_secret)
            if vp is None:
                print("No voiceprint yet. Teach one with: python -m backend.voice.enroll teach")
                return 1
            model = SpeakerModel(config.MODELS_DIR)
            probe, mic, triggers = _open_listening()
            with mic:
                for _ in range(3):
                    print("Say \"hey jarvis\"...")
                    moment = _wake_listener(mic, triggers)()
                    if not moment:
                        print("  I didn't hear the wake word.")
                        continue
                    audio = np.concatenate(moment)[-int(CHECK_SECONDS * 16000):]
                    score = similarity(model.embed(audio), np.array(vp.centroid))
                    audio = moment = None
                    print(f"  score {score:.2f} (bar {vp.threshold:.2f}): "
                          + ("that's you." if score >= vp.threshold else "not recognised."))
            del probe
    except KeyboardInterrupt:
        print("\nStopped. Nothing was saved.")
        return 1
    except VoiceprintError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:                     # microphone or Credential Manager problems
        from ..credentials import CredentialError
        from .triggers import TriggerError

        safe = isinstance(exc, (CredentialError, TriggerError))
        print(f"Error: {exc}" if safe else f"Error: something went wrong ({type(exc).__name__}).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
