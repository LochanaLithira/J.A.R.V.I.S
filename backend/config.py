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

# The only folders the file tools may touch, by name. Remove a name to keep J.A.R.V.I.S out of it.
# "workspace" is the folder above; the others are found where Windows really keeps them (even if
# moved to OneDrive). "Deleted" files go to a trash folder inside the workspace and can be restored.
FILE_FOLDERS = ("workspace", "documents", "downloads", "desktop", "pictures", "music", "videos")

# At most this many proposals per time window, enforced by the gate.
RATE_LIMIT_CALLS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0

# The AI model. Model names change often, so check Google AI Studio for the current
# free Flash or Flash-Lite names, and edit this line if the name stops working.
# The first is tried first; measured from this laptop, 3.5 answered in ~0.8-1.1 s and 3.1 in ~1.2-1.7 s.
GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
GEMINI_TIMEOUT_SECONDS = 15.0


# How J.A.R.V.I.S addresses you when it speaks ("sir", "ma'am", or a name such as "Mr Lithira").
USER_ADDRESS = "sir"
# Say a short greeting ("Good morning, sir. All systems are online.") when it starts.
GREETING_ON_START = True

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
# Only wake up if the moment "hey jarvis" was heard really contained a human voice (checked on
# this laptop). Over 4 days, 28 of 41 wake-ups were background noise; this check drops those.
WAKE_SPEECH_CHECK = True
# Measured with synthetic sounds: at 0.7 confidence a spoken "hey jarvis" gives ~0.7 s of voice,
# fast knocking 0.38 s, keyboard typing and hiss 0 s. If your real "hey jarvis" is ever ignored,
# the audit log shows "wake_rejected": lower these a little.
WAKE_SPEECH_CONFIDENCE = 0.7     # how sure the local voice detector must be, per 32 ms frame
WAKE_MIN_SPEECH_SECONDS = 0.4    # at least this much voice in the last ~2 seconds

# Double clap. OFF by default: knocks, desk taps and doors look like claps to a laptop mic (13 of
# 17 clap wake-ups in 4 days were false). CLAP_MIN_LEVEL is how loud a clap must be; your claps
# measured 815-1993 on the built-in mic. The listen command prints each clap's level for tuning.
CLAP_ENABLED = False
CLAP_MIN_LEVEL = 600


# ---- Speech input (Groq Whisper, free tier resets daily) ----
# After a wake-up, only the part where you speak is sent; silence is trimmed on this laptop.
# Set the key once with:  python -m backend.credentials set stt_api_key
STT_MODEL = "whisper-large-v3-turbo"
STT_LANGUAGE = "en"
LISTEN_MAX_SECONDS = 15.0        # longest command
LISTEN_NO_SPEECH_SECONDS = 5.0   # give up if you don't start speaking within this time
LISTEN_END_SILENCE_SECONDS = 1.2 # this much quiet means you've finished (room to pause mid-sentence)
# If what you said clearly stops mid-sentence ("open the...", "make a note called..."), keep
# listening this much longer for the rest and join the two parts. 0 turns this off.
LISTEN_UNFINISHED_EXTRA_SECONDS = 2.0
# What counts as speech, so background sound is never sent: how sure the local voice detector
# must be that speech has started (0-1), and the least speech that makes a command.
LISTEN_START_CONFIDENCE = 0.6
LISTEN_MIN_SPEECH_SECONDS = 0.4

# ---- Follow-ups: answering without "hey jarvis" ----
# When J.A.R.V.I.S asks you a question ("Which app, sir?"), it chimes and listens for your answer.
# (Used when conversation mode, below, is off.)
FOLLOW_UP_ON_QUESTION = True
FOLLOW_UP_SECONDS = 5.0          # how long to wait for you to start speaking
FOLLOW_UP_MAX_TURNS = 3          # after this many follow-ups in a row, say "hey jarvis" again

# ---- Conversation mode: one "hey jarvis", then keep talking ----
# After every reply J.A.R.V.I.S chimes and keeps listening, so you can carry on without the wake
# word. It ends when you say "that's all", "goodbye", "thanks, Jarvis" or "stop", when you stay
# quiet, at the limits below, or when Windows is locked. While a conversation is open, anything
# said near the laptop (by anyone, or a TV) is sent to Groq and may be acted on; silence never is,
# and risky actions still need your keypress. Set to False to need "hey jarvis" for every command.
CONVERSATION_MODE = True
CONVERSATION_SECONDS = 8.0       # how long it waits for you to start speaking again
CONVERSATION_MAX_TURNS = 10      # commands in one conversation, then "hey jarvis" again
CONVERSATION_MAX_SECONDS = 180.0 # longest conversation, then "hey jarvis" again
CONVERSATION_MEMORY_TURNS = 6    # recent exchanges sent to Gemini so follow-ups make sense (max 6)