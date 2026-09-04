"""Offline text-to-speech: synthesizes text to a local WAV file.

Two engines are supported, both fully local/offline (nothing is sent to a
remote service):

  - pyttsx3 (default): uses the OS's built-in voices. Lightweight, no
    model download, works out of the box everywhere.
  - chatterbox-tts (optional, set TTS_ENGINE=chatterbox): a local neural
    TTS model for higher quality voices. Heavier (pulls in PyTorch and a
    one-time model download) and not installed by default -- see README.
    Falls back to pyttsx3 automatically if it isn't installed or fails.
"""
import os
import threading

_chatterbox_model = None
_chatterbox_lock = threading.Lock()


def _load_chatterbox():
    global _chatterbox_model
    if _chatterbox_model is None:
        with _chatterbox_lock:
            if _chatterbox_model is None:
                from chatterbox.tts import ChatterboxTTS
                device = os.environ.get("TTS_DEVICE", "cpu")
                _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    return _chatterbox_model


def _synthesize_with_chatterbox(text, out_path):
    import torchaudio as ta
    model = _load_chatterbox()
    wav = model.generate(text)
    ta.save(out_path, wav, model.sr)


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

    if os.environ.get("TTS_ENGINE", "pyttsx3").lower() == "chatterbox":
        try:
            _synthesize_with_chatterbox(text, out_path)
            return out_path
        except Exception as e:
            print(f"[chatterbox-tts unavailable ({e}), falling back to pyttsx3]")

    _synthesize_with_pyttsx3(text, out_path)
    return out_path
