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
- **Safety gate for every action:** the AI can suggest an action, but it cannot execute tools directly.
- **Approved tools only:** notes and configured Windows applications are registered explicitly.
- **Confirmation for risky actions** and rate limiting for AI requests.
- **Tamper-evident audit logging** stored outside the writable workspace.
- **Credential protection** through Windows Credential Manager via `keyring`.
- **Privacy-conscious design:** wake-word detection and voice activity detection run locally; only the spoken command after the listening chime is sent for transcription.

## Requirements

- Windows with access to a microphone and speakers.
- Python 3.11 or newer recommended.
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
> listen
```

`ask` is the safest way to inspect an AI suggestion before running it. High-risk operations may still require confirmation at the gate.

## Configuration

Most local settings are in `backend/config.py`, including:

- The writable workspace: `WORKSPACE`.
- The audit log location: `AUDIT_LOG`.
- Allowed Windows applications: `ALLOWED_APPS`.
- AI models and request timeout.
- Microphone selection and wake-word sensitivity.
- Speech recording limits.
- Double-clap detection settings.

Only applications explicitly added to `ALLOWED_APPS` can be opened by the application tools. Use full executable paths and review changes carefully.

## Security and privacy

- The AI suggests tools; it does not execute them directly.
- All tool calls pass through the gate, which validates names and arguments and can require confirmation.
- Wake-word detection and voice activity detection run locally.
- Audio before the wake-up event and silence are discarded rather than uploaded.
- Do not send private information in `ask` or `say` commands without reviewing the privacy policies and account settings of the configured providers.
- Never commit API keys, credential exports, model files, or local audit logs.
- LangSmith tracing is intentionally refused by the flow because it could send commands to a third party.

## Testing

Run the test suite with:

```powershell
python -m pytest
```

The tests cover the command flow, safety gate, configuration, credentials, tools, voice input/output, wake triggers, and audit logging.

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
│   ├── registry.py       # Explicit approved-tool registry
│   ├── tools/            # Notes and application tools
│   ├── ui/               # Confirmation UI
│   └── voice/             # Wake word, microphone, speech input, and speech output
├── scripts/               # Setup utilities
├── tests/                 # Automated tests
├── requirements.in        # Runtime dependency inputs
├── requirements-dev.in    # Development dependency inputs
├── SECURITY.md            # Security reporting guidance
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
