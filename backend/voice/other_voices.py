"""Other people's voices for comparison, made on the fly by Windows' built-in text-to-speech.

Used when teaching J.A.R.V.I.S your voice, to place the bar between you and someone else saying
"hey jarvis". The sound is made inside one PowerShell process, handed over through a pipe as
base64, and kept in memory only: nothing is written to disk and nothing goes online.
"""
from __future__ import annotations

import base64
import subprocess
import sys

import numpy as np

PHRASES = ("Hey Jarvis", "Hey Jarvis, open notepad", "Hey Jarvis, what time is it")
RATES = (-2, 0, 2)                      # slower, normal, faster
MAX_BYTES = 20_000_000

# Classic (SAPI) voices first, then the newer OneCore voices that most Windows 11 PCs also have.
_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
  [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($v in $s.GetInstalledVoices()) {
  foreach ($rate in @(RATES)) { foreach ($text in @(PHRASES)) {
    $m = New-Object System.IO.MemoryStream
    $s.SelectVoice($v.VoiceInfo.Name); $s.Rate = $rate
    $s.SetOutputToAudioStream($m, $fmt); $s.Speak($text); $s.SetOutputToNull()
    [Console]::Out.WriteLine('SAPI|' + $v.VoiceInfo.Name + '|' + [Convert]::ToBase64String($m.ToArray()))
  } }
}
$s.Dispose()
try {
  [void][Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media, ContentType = WindowsRuntime]
  [void][Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
  Add-Type -AssemblyName System.Runtime.WindowsRuntime
  $await = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
  $w = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
  foreach ($v in [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices) {
    if (-not $v.Language.StartsWith('en')) { continue }
    $w.Voice = $v
    foreach ($text in @(PHRASES)) {
      $task = $await.MakeGenericMethod([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]).Invoke($null, @($w.SynthesizeTextToStreamAsync($text)))
      $stream = $task.Result
      $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
      $load = $await.MakeGenericMethod([UInt32]).Invoke($null, @($reader.LoadAsync([UInt32]$stream.Size)))
      [void]$load.Result
      $bytes = New-Object byte[] $stream.Size
      $reader.ReadBytes($bytes)
      [Console]::Out.WriteLine('WAV|' + $v.DisplayName + '|' + [Convert]::ToBase64String($bytes))
    }
  }
} catch { }
"""


def _script() -> str:
    phrases = ",".join("'" + p.replace("'", "''") + "'" for p in PHRASES)
    return _SCRIPT.replace("@(RATES)", "@(" + ",".join(map(str, RATES)) + ")").replace("@(PHRASES)", "@(" + phrases + ")")


def _from_wav(data: bytes) -> np.ndarray | None:
    """OneCore voices give a WAV (often 24 kHz or 22.05 kHz); turn it into 16 kHz int16."""
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2:
                return None
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except (wave.Error, EOFError):
        return None
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != 16000:                      # simple resampling is fine for voice numbers
        length = int(len(pcm) * 16000 / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, length), np.arange(len(pcm)), pcm)
    return np.asarray(pcm, dtype=np.int16)


def other_voices(timeout: float = 120.0) -> list[tuple[str, np.ndarray]]:
    """(voice name, 16 kHz int16 clip) of Windows' voices saying "hey jarvis" and similar.
    [] if unavailable."""
    if sys.platform != "win32":
        return []
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _script()],
                              capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if len(done.stdout) > MAX_BYTES:
        return []
    clips = []
    for line in done.stdout.decode("ascii", "ignore").splitlines():
        kind, _, rest = line.strip().partition("|")
        name, _, payload = rest.partition("|")
        name = " ".join(name.replace("Microsoft", "").replace("Desktop", "").split())[:40]
        try:
            data = base64.b64decode(payload, validate=True)
        except ValueError:
            continue
        if kind == "SAPI" and len(data) >= 3200:
            clips.append((name, np.frombuffer(data[: len(data) // 2 * 2], dtype=np.int16).copy()))
        elif kind == "WAV":
            pcm = _from_wav(data)
            if pcm is not None and len(pcm) >= 3200:
                clips.append((name, pcm))
    return clips
