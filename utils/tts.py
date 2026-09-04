"""Offline text-to-speech: synthesizes text to a local WAV file.

Two engines are supported, both fully local/offline (nothing is sent to a
remote service):

  - Kokoro (default, if installed): an 82M-parameter neural TTS model
    (StyleTTS 2 based) that sounds substantially more natural than
    classic OS voices, while still being small/fast enough to run on
    CPU. Optional dependency -- see README -- since it pulls in PyTorch
    and requires the espeak-ng system package.
  - pyttsx3 (fallback): uses the OS's built-in voices. Robotic but
    lightweight, no model download, and always available, so it's what
    keeps the app working if Kokoro isn't installed or fails to load.

TTS_ENGINE controls which is used:
  - "auto" (default): try Kokoro, fall back to pyttsx3 if unavailable.
  - "kokoro": use Kokoro only (still falls back to pyttsx3 on failure,
    so a missing optional dependency never breaks a chat response).
  - "pyttsx3": skip Kokoro entirely (useful on low-resource machines).
"""
import os
import threading

_kokoro_pipeline = None
_kokoro_lock = threading.Lock()

KOKORO_SAMPLE_RATE = 24000


def _load_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        with _kokoro_lock:
            if _kokoro_pipeline is None:
                from kokoro import KPipeline
                lang_code = os.environ.get("TTS_LANG_CODE", "a")  # 'a' = American English
                device = os.environ.get("TTS_DEVICE") or None  # None = auto-select cuda/cpu
                _kokoro_pipeline = KPipeline(lang_code=lang_code, device=device)
    return _kokoro_pipeline


def _synthesize_with_kokoro(text, out_path):
    import numpy as np
    import soundfile as sf

    pipeline = _load_kokoro()
    voice = os.environ.get("TTS_VOICE", "af_heart")
    chunks = [result.audio.detach().cpu().numpy() for result in pipeline(text, voice=voice)]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    sf.write(out_path, audio, KOKORO_SAMPLE_RATE)


def _synthesize_with_pyttsx3(text, out_path):
    import pyttsx3
    engine = pyttsx3.init()
    try:
        voices = engine.getProperty("voices")
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
    finally:
        engine.stop()


def synthesize_to_file(text: str, out_path: str) -> str:
    """Synthesize `text` to a local WAV file at `out_path`. Returns out_path."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    engine = os.environ.get("TTS_ENGINE", "auto").lower()
    if engine in ("auto", "kokoro"):
        try:
            _synthesize_with_kokoro(text, out_path)
            return out_path
        except Exception as e:
            print(f"[kokoro unavailable ({e}), falling back to pyttsx3]")

    _synthesize_with_pyttsx3(text, out_path)
    return out_path
