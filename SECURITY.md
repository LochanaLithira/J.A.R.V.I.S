# Security

J.A.R.V.I.S. is a voice assistant that can act on your Windows laptop, under your own account. Security is its first design goal. This document explains how it protects you, what leaves your laptop, what it does not protect against, and how to report a problem.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

- Use GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**.
- Include the steps to reproduce it, what you expected, and what happened. Never include real API keys or private data.
- The `dev` branch is the supported version; fixes land there first.

## The core idea

**The AI only suggests. A gate decides.**

Gemini turns your sentence into a *proposal*, such as "open_app with name notepad". It has no way to run anything itself. Every proposal goes through the gate (`backend/gate/`), which:

1. refuses any tool that is not in the approved registry (`backend/registry.py`);
2. checks every argument against a strict schema: exact keys, exact types, lengths, patterns, allowed values;
3. applies a rate limit;
4. for high-risk tools, shows a confirmation window that only a real **Y** keypress or click can approve (after a one-second delay, with a 30-second auto-deny);
5. writes every step to a tamper-evident audit log.

Tool results are **never sent back to the AI**. This blocks the most dangerous kind of prompt injection, where text inside a file or web page tells the AI what to do next.

## What leaves your laptop

| Service | What is sent | When | Notes |
| --- | --- | --- | --- |
| Google Gemini (free tier) | Your command text, the list of tool descriptions, and during a conversation up to six earlier exchanges: your sentences plus the AI's own replies or proposed actions (never what a tool returned) | For each command | On the free tier Google may review requests to improve its products. Do not say private things to J.A.R.V.I.S. The conversation memory is cleared when the conversation ends. |
| Groq Whisper (free tier) | The audio of your spoken command only, with silence trimmed | Only after the wake word, double clap, or the console's `hear` | Groq does not keep inputs by default. Turn on **Zero Data Retention** in the Groq console to be sure. |
| Fish Audio (free model) | The text of J.A.R.V.I.S.'s spoken reply | For each spoken reply | Private results (note text, file names, reminder labels) are **never** sent; a fixed line is spoken instead. Fish may keep free-tier requests. |
| GitHub (openWakeWord release) | A normal download request | Once, when you run the model download | Every model file is checked against a fixed SHA-256 fingerprint before it is saved or loaded. |

Nothing else is sent anywhere. In particular:

- Wake-word and clap detection, and the detection of where your speech starts and ends, run **entirely on your laptop**.
- Audio before the wake-up, silence, and anything said while J.A.R.V.I.S. is speaking are discarded. If you say nothing after the chime, nothing is uploaded.
- Every upload uses HTTPS to one fixed host, never follows redirects (so a key can never be forwarded elsewhere), has timeouts, and caps the size of the reply.
- The on-demand screen opens **no network port**. It talks to J.A.R.V.I.S. only through private pipes between the two processes.

## What stays on your laptop

| Item | Where | Contents |
| --- | --- | --- |
| API keys | Windows Credential Manager | Never stored in files, logged, printed, or put in error messages. A plain-text fallback keyring is refused. |
| Audit log | `%LOCALAPPDATA%\jarvis\audit.jsonl` | Each command's text, the AI's proposals and their arguments (clipped to 200 characters), decisions, wake-ups, and upload lengths. Hash-chained so edits are detectable (`verify` in the console). **It contains your commands**: treat it as private. |
| Error log | `%LOCALAPPDATA%\jarvis\jarvis.log` | Error types, fixed messages and timings only, never your words. Capped at about 3 MB. |
| Models | `%LOCALAPPDATA%\jarvis\models` | The four ONNX model files, fingerprint-checked on every load. |
| Workspace | `~\jarvis_workspace` | Notes, plus `trash\` for "deleted" files, which can be moved back. |
| Recent conversation | Memory only | Shown on the screen while J.A.R.V.I.S. runs; gone when it quits. Timers are also memory only. |

## Protections in place

**Actions**
- Only explicitly registered tools exist. Apps open only from a fixed allowlist of full paths, with no shell and no arguments.
- File tools reach only a named list of folders (`FILE_FOLDERS` in `config.py`), by file *name*, never by path. Hidden files, system files, links and junctions are invisible, and each file's real location must still be inside its folder.
- Only documents and media can be opened, never programs, scripts or shortcuts. Nothing is ever overwritten. A rename cannot change a file's type. "Delete" moves the file to the workspace trash, after a keypress.
- Notes cannot be overwritten and live only inside the workspace.

**Voice**
- Waking up grants no permission. It only starts listening, and every action still goes through the gate.
- The microphone is **closed** while J.A.R.V.I.S. is paused, while Windows is locked, and while the laptop sleeps, so nobody can command your laptop while you are away.
- J.A.R.V.I.S. can never wake itself or answer itself: the wake-up is muted while it speaks, and audio captured meanwhile is thrown away.
- Each spoken command is capped at 15 seconds (a sentence that stops mid-way may get 2 seconds more).
- A conversation (one `hey jarvis`, then talking freely) ends when you say goodbye or "that's all", after 8 seconds of silence, after 10 commands or 3 minutes, or when Windows is locked. While it is open, anything said near the laptop may be sent to Groq; silence never is, and risky actions still need the keypress.

**Program**
- One copy only, enforced with a Windows lock (two copies would break the audit chain).
- A watchdog restarts listening after unexpected errors; a microphone that goes silent after sleep is reopened.
- Below-normal priority. Runs under your account, never as administrator.
- Dependencies are pinned with SHA-256 hashes (`requirements.txt`).
- LangSmith tracing is refused, because it would send your commands to a third party.

**Screen**
- Separate process; loaded from local files only, with a strict Content-Security-Policy; text is always shown as text, never as HTML.
- It accepts only a fixed list of display updates, and can send back exactly one command: pause or resume.
- Private mode, no downloads, no right-click menu, no developer tools.

## Threat model

### What we protect
1. Your files and the actions taken on your laptop.
2. Your words and audio.
3. Your API keys.
4. The integrity of the audit log.

### Threats, and how they are handled

| Threat | Example | Mitigation |
| --- | --- | --- |
| The AI misunderstands or invents an action | "delete" heard as something else; a made-up tool | Approved tools only; strict argument checks; keypress for high-risk tools; moves and deletes are reversible |
| Prompt injection in your speech | "Ignore your rules and..." | Rules are in the system prompt, never mixed with your words; the AI has no tools of its own; the gate re-checks everything |
| Prompt injection in data | A note or file that contains instructions | Tool results never go back to the AI |
| Someone else talks to your laptop | A visitor, or a TV saying "hey jarvis" | Waking gives no permission; risky actions need a keypress; the mic is closed while Windows is locked |
| Misheard or false wake-up | A video triggers the wake word | Silence is never sent; nothing runs without a clear command; wake-ups are audited |
| Path tricks on files | `..\`, `C:\Windows`, a junction pointing elsewhere | Names only; links and junctions invisible; real location checked |
| Running a disguised program | `invoice.pdf.exe` | Only document and media types open |
| Key theft from files or logs | A leaked log or screenshot | Keys live only in Credential Manager and never appear in output |
| Network attacker | A changed download or a redirect | HTTPS, no redirects, fingerprint-checked models, hash-pinned packages |
| A website talking to J.A.R.V.I.S. | A page in your browser probing local servers | There is no local server or port |
| Audit log tampering | Someone edits the log to hide an action | Hash chain makes edits, removals and reordering detectable |
| Repeated or runaway requests | A loop or a stuck microphone | Rate limits on the AI and the gate; 15-second command cap |

### What is NOT protected against

Being honest about limits matters as much as the protections:

- **Malware already running as you.** It can read your files, read the same Credential Manager entries, synthesize the Y keypress, and rewrite the whole audit log. J.A.R.V.I.S. cannot defend against software with your own rights.
- **The cloud providers.** What you say goes to Google, and the spoken command audio goes to Groq, under their own terms. On free tiers they may keep or review it.
- **Medium-risk actions run without asking.** Opening a document, creating a note, or moving and renaming a file run straight away; they are reversible, but a misheard command can do one of them.
- **Opening a document runs its default app.** A malicious document in your Downloads could attack that app. Only document and media types are allowed, but the app itself is outside J.A.R.V.I.S.'s control.
- **The audit log is evidence, not a vault.** It is tamper-evident, not tamper-proof, and it holds your command text in plain form on your disk.
- **Voice is not identity.** J.A.R.V.I.S. does not recognise *who* is speaking.

## Adding a new tool safely

Every new tool must pass this checklist:

1. **Register it** in `backend/registry.py`, and nowhere else.
2. **Give it the honest risk level.** Use HIGH for anything irreversible or affecting other people, such as sending, paying, or permanently deleting. HIGH needs a keypress.
3. **Type every argument strictly**: exact type, maximum length, a pattern or a fixed list of choices. Never accept a path, a URL, or a shell command as free text.
4. **Never overwrite, never permanently delete.** Prefer actions that can be undone.
5. **Mark it `private_output=True`** if its result can contain the user's data, so the result is never spoken by the cloud voice.
6. **Raise `ToolError`** with a plain, safe message for expected failures. Never show stack traces, paths outside the allowed folders, or secrets.
7. **Never feed its result back to the AI.**
8. **Add tests for refusals**, not only for success: bad arguments, tricks, and a missing confirmation.

## Hardening tips

- Turn on **Zero Data Retention** in the Groq console (Settings, then Data Controls).
- Remove folders you never want touched from `FILE_FOLDERS` in `backend/config.py`.
- Lock Windows (Windows key + L) when you step away; J.A.R.V.I.S. stops listening at once.
- Set `CONVERSATION_MODE = False` if you prefer to say `hey jarvis` before every command, so the microphone is never left open between commands.
- Run `verify` in the console now and then to check the audit chain.
- Turn autostart off with `scripts\install_autostart.ps1 -Remove` if you only want J.A.R.V.I.S. on demand.
