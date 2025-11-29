# utils/tts.py
import os
import uuid
import tempfile
from threading import Event
from chatterbox_tts import TTS  # offline package: chatterbox-tts

# NOTE: model names/voice names depend on chatterbox-tts version.
# Replace 'default' and 'voices' with actual model/voice IDs if required.

# Load engine once
# If your chatterbox-tts requires different constructor args, adjust here.
tts_engine = TTS(model="default", local=True)  # best-effort init

# Simple cache of available voices (depends on your local models)
def available_voices():
    """
    Return list of available voice names/IDs. Adjust to your package if API differs.
    """
    try:
        # some chatterbox versions expose voices()
        voices = getattr(tts_engine, "voices", None)
        if callable(voices):
            return tts_engine.voices()
        # fallback — common named voices
        return ["default", "female_1", "male_1"]
    except Exception:
        return ["default"]

def synthesize_to_file(text: str, out_path: str, voice: str = "default"):
    """
    Synthesize text to out_path using chosen voice. Returns out_path.
    """
    # ensure folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # The exact API to synthesize depends on chatterbox-tts version.
    # Two common patterns:
    # 1) audio = tts_engine.generate(text, voice=voice); audio.save(out_path)
    # 2) tts_engine.synthesize_to_file(text, out_path, voice=voice)
    try:
        # try common generate + save pattern
        gen = getattr(tts_engine, "generate", None)
        if callable(gen):
            audio = tts_engine.generate(text, voice=voice)
            # audio might be a bytes object or have .save
            if hasattr(audio, "save"):
                audio.save(out_path)
            elif isinstance(audio, (bytes, bytearray)):
                with open(out_path, "wb") as f:
                    f.write(audio)
            else:
                # try to call write out raw audio buffer
                with open(out_path, "wb") as f:
                    f.write(bytes(audio))
            return out_path

        # try direct synthesize_to_file
        synth = getattr(tts_engine, "synthesize_to_file", None)
        if callable(synth):
            synth(text=text, file_path=out_path, voice=voice)
            return out_path

        # last resort - use text_to_speech
        tts_engine.text_to_speech(text=text, file_path=out_path, voice=voice)
        return out_path

    except Exception as e:
        raise RuntimeError(f"TTS generation failed: {e}")

def synthesize_async(text: str, out_path: str, voice: str = "default"):
    """
    Starts async generation in background and returns an Event that is set when finished.
    Writes file to out_path.
    """
    done = Event()

    def _worker():
        try:
            synthesize_to_file(text, out_path, voice=voice)
        finally:
            done.set()

    import threading
    threading.Thread(target=_worker, daemon=True).start()
    return done
