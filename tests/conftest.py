import subprocess
import sys
import wave

import numpy as np
import pytest

from backend import config
from backend.gate.audit import AuditLog
from backend.voice.triggers import TriggerError, verify_models


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "logs" / "audit.jsonl")


@pytest.fixture(scope="session")
def models_dir():
    try:
        verify_models(config.MODELS_DIR)
    except TriggerError:
        pytest.skip("models not downloaded (python -m backend.voice.triggers download)")
    return config.MODELS_DIR


@pytest.fixture(scope="session")
def spoken(tmp_path_factory):
    """Phrases spoken by Windows' built-in offline voice, as 16 kHz int16 arrays with 1 s of
    silence before and after."""
    if sys.platform != "win32":
        pytest.skip("needs Windows text-to-speech")
    out = tmp_path_factory.mktemp("spoken")
    phrases = {"wake": "Hey Jarvis", "other": "I was talking to Travis about the weather yesterday",
               "command": "Open notepad please"}
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