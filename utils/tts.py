"""Offline text-to-speech: synthesizes text to a local WAV file.

Two engines are supported, both fully local/offline (nothing is sent to a
remote service):

  - Kokoro (default, if installed): an 82M-parameter neural TTS model
    (StyleTTS 2 based) that sounds substantially more natural than
    classic OS voices, while still being small/fast enough to run on
    CPU. Uses kokoro-onnx (onnxruntime, no PyTorch needed) against the
    model files bundled in models/kokoro/ -- no network access or
    external model hub required at all, since those hosts are blocked
    on some networks. See README for the optional pip install.
  - pyttsx3 (fallback): uses the OS's built-in voices. Robotic but
    lightweight, no model download, and always available, so it's what
    keeps the app working if kokoro-onnx isn't installed or fails to load.

TTS_ENGINE controls which is used:
  - "auto" (default): try Kokoro, fall back to pyttsx3 if unavailable.
  - "kokoro": use Kokoro only (still falls back to pyttsx3 on failure,
    so a missing optional dependency never breaks a chat response).
  - "pyttsx3": skip Kokoro entirely (useful on low-resource machines).
"""
import os
import threading

_kokoro_engine = None
_kokoro_lock = threading.Lock()

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "kokoro")
_DEFAULT_KOKORO_MODEL = os.path.join(_MODELS_DIR, "kokoro-v1.0.int8.onnx")
_DEFAULT_KOKORO_VOICES = os.path.join(_MODELS_DIR, "voices-v1.0.bin")


def _load_kokoro():
    global _kokoro_engine
    if _kokoro_engine is None:
        with _kokoro_lock:
            if _kokoro_engine is None:
                from kokoro_onnx import Kokoro
                model_path = os.environ.get("KOKORO_MODEL_PATH", _DEFAULT_KOKORO_MODEL)
                voices_path = os.environ.get("KOKORO_VOICES_PATH", _DEFAULT_KOKORO_VOICES)
                _kokoro_engine = Kokoro(model_path, voices_path)
    return _kokoro_engine


def _synthesize_with_kokoro(text, out_path):
    import soundfile as sf

    engine = _load_kokoro()
    voice = os.environ.get("TTS_VOICE", "af_heart")
    lang = os.environ.get("TTS_LANG", "en-us")
    samples, sample_rate = engine.create(text, voice=voice, speed=1.0, lang=lang)
    sf.write(out_path, samples, sample_rate)


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


def warm_up():
    """Eagerly load Kokoro and run one throwaway synthesis so any one-time
    cost (reading the ~90MB model file, building the onnxruntime session,
    initializing the espeak-ng phonemizer backend) happens now, in the
    background at startup, instead of during -- and slowing down or
    tripping up -- a user's first chat request."""
    if os.environ.get("TTS_ENGINE", "auto").lower() == "pyttsx3":
        return

    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _synthesize_with_kokoro("Warming up.", tmp_path)
        print("[kokoro warmed up]")
    except Exception as e:
        print(f"[kokoro warmup skipped: {e}]")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
