# J.A.R.V.I.S.

A security-focused, voice-enabled personal assistant for Windows. J.A.R.V.I.S. turns typed or spoken commands into carefully validated tool proposals, then executes only approved actions through a central safety gate.

> **Development status:** This project is under active development and is currently designed for Windows.

## Highlights

- **Gemini-powered command understanding** using Google's Gemini API.
- **Voice interaction** with:
  - `hey jarvis` wake-word detection.
  - Optional double-clap activation.
  - Local voice activity detection.
  - Groq Whisper speech-to-text.
  - Optional Fish Audio text-to-speech.
- **Runs in the background** with a tray icon, and an on-demand futuristic screen showing live status, the conversation, timers and activity.
- **Smart follow-ups:** when J.A.R.V.I.S. asks you a question, answer without saying the wake word again.
- **Safety gate for every action:** the AI can suggest an action, but it cannot execute tools directly.
- **Approved tools only:** notes, configured Windows applications, file tools limited to named folders, and timers are registered explicitly.
- **Confirmation for risky actions** and rate limiting for AI requests.
- **Tamper-evident audit logging** stored outside the writable workspace.
- **Credential protection** through Windows Credential Manager via `keyring`.
- **Privacy-conscious design:** wake-word detection and voice activity detection run locally; only the spoken command after the listening chime is sent for transcription.

## Requirements

- Windows with access to a microphone and speakers.
- Python 3.11 or newer recommended (developed and tested on Python 3.14).
- Microsoft Edge WebView2 for the on-demand screen (already included in Windows 11).
- API keys for the cloud services you plan to use:
  - Google Gemini for typed commands and AI suggestions.
  - Groq Whisper for speech-to-text.
  - Fish Audio for optional spoken replies.
- A working audio input device. The default configuration expects a microphone containing `Microphone Array` in its name; update `MIC_DEVICE` in `backend/config.py` if necessary.

## Installation

Clone the repository and switch to the development branch:

```powershell
git clone https://github.com/LochanaLithira/J.A.R.V.I.S.git
cd J.A.R.V.I.S
git checkout dev
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install runtime and development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

If the generated requirements files are unavailable, install from the input files with your preferred dependency compiler, or install the packages listed in `requirements.in` and `requirements-dev.in`.

## Configure credentials

J.A.R.V.I.S. stores supported API keys in Windows Credential Manager rather than in project files or environment variables.

Set the credentials you need. Input is hidden and is not written to shell history:

```powershell
python -m backend.credentials set gemini_api_key
python -m backend.credentials set stt_api_key
python -m backend.credentials set fish_api_key
```

Check or remove a credential without displaying its value:

```powershell
python -m backend.credentials status gemini_api_key
python -m backend.credentials delete gemini_api_key
```

The supported credential names are:

- `gemini_api_key`
- `stt_api_key`
- `fish_api_key`

The Fish Audio key is optional if voice output is disabled. The speech-to-text key is only needed for `hear` and `listen`.

## Download voice models

Wake-word and voice activity models are downloaded once and verified with fixed SHA-256 fingerprints:

```powershell
python -m backend.voice.triggers download
```

Verify already-downloaded models:

```powershell
python -m backend.voice.triggers status
```

A changed or corrupted model is refused instead of being loaded.

## Run J.A.R.V.I.S. in the background

This is the everyday way to use it. From the repository root:

```powershell
.\.venv\Scripts\pythonw.exe -m backend.main
```

No console window opens. A round icon appears next to the clock:

| Colour | Meaning |
| --- | --- |
| Green | Listening for `hey jarvis` or a double clap |
| Blue | Working on a command |
| Grey | Paused, Windows locked, or asleep (microphone closed) |
| Red | A problem (hover over the icon to see what) |

- **Left-click** the icon to open the J.A.R.V.I.S. screen.
- **Right-click** for Open J.A.R.V.I.S., Pause / Resume listening, and Quit.

Say `hey jarvis`, wait for the chime, then speak your command. A falling tone means it heard you. If J.A.R.V.I.S. asks you a question, just answer after the chime; say "stop" or "cancel" to end. Only one copy can run at a time. Errors are written to `%LOCALAPPDATA%\jarvis\jarvis.log` (never your words).

## Start with Windows

Turn on autostart. This adds one shortcut to your own Startup folder; no administrator rights or system changes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
```

Check it, or turn it off again:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Status
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
```

You can also switch it off in Task Manager > Startup apps. `-ExecutionPolicy Bypass` applies to that one run only.

## Run the development console

Start J.A.R.V.I.S. from the repository root:

```powershell
python -m backend.cli
```

Available commands:

| Command | Description |
| --- | --- |
| `help` | Show the command list. |
| `tools` | List registered tools and their risk levels. |
| `ask <sentence>` | Dry run: ask Gemini what it would suggest; nothing is executed. |
| `say <sentence>` | Send a command through Gemini, the safety gate, and the approved tools. |
| `hear` | Record one spoken command and display its transcription; nothing is executed. |
| `listen` | Wait for `hey jarvis` or a double clap, then listen for and execute commands. Press `Ctrl+C` to stop. |
| `voice on` / `voice off` | Enable or disable spoken replies. |
| `log` | Display the latest audit entries. |
| `verify` | Check the audit log for tampering. |
| `quit` | Exit the console. |

Examples:

```text
> tools
> ask open notepad
> say open notepad
> list_notes
> create_note {"title": "ideas", "text": "first note"}
> read_note {"title": "ideas"}
> list_files {"folder": "downloads"}
> set_timer {"seconds": 300, "label": "tea"}
> listen
```

Things you can say once it is listening, for example:

- "open notepad", "make a note called shopping and write buy milk"
- "what's in my downloads", "find my budget file", "open my documents folder"
- "move cv from downloads to documents", "rename draft to final in documents"
- "delete report from my desktop" (moves it to the workspace trash after you press Y)
- "set a timer for five minutes", "remind me in ten minutes to call mom", "cancel all my timers"

`ask` is the safest way to inspect an AI suggestion before running it. High-risk operations may still require confirmation at the gate.

## Configuration

Most local settings are in `backend/config.py`, including:

- The writable workspace: `WORKSPACE`.
- The audit log location: `AUDIT_LOG`.
- The error log location: `APP_LOG`.
- Allowed Windows applications: `ALLOWED_APPS`.
- Folders the file tools may use: `FILE_FOLDERS`.
- AI models and request timeout.
- Microphone selection and wake-word sensitivity.
- Speech recording limits.
- Double-clap detection settings.
- Follow-ups: `FOLLOW_UP_ON_QUESTION`, `FOLLOW_UP_ALWAYS` (off by default), `FOLLOW_UP_SECONDS`, `FOLLOW_UP_MAX_TURNS`.

Only applications explicitly added to `ALLOWED_APPS` can be opened by the application tools. Use full executable paths and review changes carefully.

## Security and privacy

- The AI suggests tools; it does not execute them directly.
- All tool calls pass through the gate, which validates names and arguments and can require confirmation.
- Wake-word detection and voice activity detection run locally.
- Audio before the wake-up event and silence are discarded rather than uploaded.
- The microphone is closed while J.A.R.V.I.S. is paused, while Windows is locked, and while the laptop sleeps.
- Private results such as note text, file names and reminder labels are shown on screen only, never sent to the voice service.
- The on-demand screen opens no network port.
- See [SECURITY.md](SECURITY.md) for exactly what leaves your laptop, the threat model, and its limits.
- Do not send private information in `ask` or `say` commands without reviewing the privacy policies and account settings of the configured providers.
- Never commit API keys, credential exports, model files, or local audit logs.
- LangSmith tracing is intentionally refused by the flow because it could send commands to a third party.

## Testing

Run the test suite with:

```powershell
python -m pytest
```

The tests cover the command flow, safety gate, configuration, credentials, tools (notes, apps, files, timers), voice input/output, wake triggers, follow-ups, the background app, the tray, the screen, autostart, and audit logging.

Tests never touch your real Documents or Downloads, and never change your real Startup folder. Tests that open real windows on screen are skipped unless you set `JARVIS_GUI_TESTS=1`.

## Project structure

```text
.
├── backend/
│   ├── brain.py          # Gemini integration and strict response parsing
│   ├── cli.py            # Interactive development console
│   ├── config.py         # Local settings and safety limits
│   ├── credentials.py    # Windows Credential Manager integration
│   ├── flow.py           # Brain -> gate -> result workflow
│   ├── gate/             # Validation, confirmation, execution, and audit logging
│   ├── main.py           # Background app: listening loop, watchdog, startup
│   ├── registry.py       # Explicit approved-tool registry
│   ├── tools/            # Notes, application, file and timer tools
│   ├── ui/               # Confirmation popup, tray icon, screen window and its page
│   └── voice/             # Wake word, microphone, speech input, and speech output
├── scripts/               # Setup utilities (autostart)
├── tests/                 # Automated tests
├── requirements.in        # Runtime dependency inputs
├── requirements-dev.in    # Development dependency inputs
├── SECURITY.md            # Security model, threat model, and reporting
└── LICENSE                # MIT License
```

## Contributing

1. Create a feature branch from `dev`.
2. Keep new capabilities behind the safety gate and register tools explicitly.
3. Add or update tests for your changes.
4. Run `python -m pytest`.
5. Open a pull request describing the security and privacy impact of the change.

## License

J.A.R.V.I.S. is released under the [MIT License](LICENSE).

## Security reports

Please review [SECURITY.md](SECURITY.md) for guidance on reporting security vulnerabilities.
