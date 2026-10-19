"""Central settings. Edit paths for your machine."""
from __future__ import annotations

import os
from pathlib import Path

# The only folder the agent is allowed to write to.
WORKSPACE = Path.home() / "jarvis_workspace"

# The audit log lives OUTSIDE the workspace, so no tool can ever reach it.
_base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
AUDIT_LOG = _base / "jarvis" / "audit.jsonl"

# The background app's error log: error types and fixed messages only, never your words.
# Capped at about 3 MB in total (1 MB plus two older copies).
APP_LOG = _base / "jarvis" / "jarvis.log"

# Short name -> full path of the program. Add apps one at a time, with full paths.
ALLOWED_APPS = {
    "notepad": r"C:\Windows\System32\notepad.exe",
    "calculator": r"C:\Windows\System32\calc.exe",
}

# At most this many proposals per time window, enforced by the gate.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

# The AI model. Model names change often, so check Google AI Studio for the current
# free Flash or Flash-Lite names, and edit this line if the name stops working.
# The first is tried first; measured from this laptop, 3.5 answered in ~0.8-1.1 s and 3.1 in ~1.2-1.7 s.
GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
GEMINI_TIMEOUT_SECONDS = 15.0


# Fish Audio voice. Leave empty for the default voice, or paste a voice ID (32 lowercase hex
# characters) from fish.audio: open a voice and copy its model ID.
FISH_VOICE_ID = "05b36da8574341d0803391491850db20"


# ---- Wake-up triggers ("hey jarvis" and a double clap), all detected on this laptop ----

# The wake-word model files. Outside the workspace, so no tool can ever change them.
# Fetch them once with:  python -m backend.voice.triggers download
MODELS_DIR = _base / "jarvis" / "models"

# Part of the microphone's name, or None for the Windows default input. The Windows default on
# this laptop is a virtual mic (AudioRelay), so the built-in microphone is named here.
MIC_DEVICE = "Microphone Array"

# Tuned on your own voice (26 tries): 0.25 with patience 1 caught 25 of 26, and a minute of
# normal talk never scored above 0.006. Raise the threshold if it wakes by mistake.
WAKE_THRESHOLD = 0.25
WAKE_PATIENCE = 1                # slices of 80 ms in a row that must pass the threshold
WAKE_COOLDOWN_SECONDS = 1.5      # after a wake, ignore triggers for this long

# Double clap. CLAP_MIN_LEVEL is how loud a clap must be. Your claps measured 815-1993 on the
# built-in mic; claps are told apart from speech mainly by how suddenly they start and how fast
# they fade. The listen command prints the level of each clap it hears, so you can tune it.
CLAP_ENABLED = True
CLAP_MIN_LEVEL = 600


# ---- Speech input (Groq Whisper, free tier resets daily) ----
# After a wake-up, only the part where you speak is sent; silence is trimmed on this laptop.
# Set the key once with:  python -m backend.credentials set stt_api_key
STT_MODEL = "whisper-large-v3-turbo"
STT_LANGUAGE = "en"
LISTEN_MAX_SECONDS = 15.0        # longest command
LISTEN_NO_SPEECH_SECONDS = 5.0   # give up if you don't start speaking within this time
LISTEN_END_SILENCE_SECONDS = 0.8 # this much quiet means you've finished

# ---- Follow-ups: answering without "hey jarvis" ----
# When J.A.R.V.I.S asks you a question ("Which app, sir?"), it chimes and listens for your answer.
FOLLOW_UP_ON_QUESTION = True
# Alexa-style: also listen for a few seconds after EVERY reply. Off by default, because anything
# said in that window (by anyone, or a TV) is sent to Groq and may be acted on.
FOLLOW_UP_ALWAYS = False
FOLLOW_UP_SECONDS = 5.0          # how long to wait for you to start speaking
FOLLOW_UP_MAX_TURNS = 3          # after this many follow-ups in a row, say "hey jarvis" again