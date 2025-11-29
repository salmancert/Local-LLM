# utils/tts.py
from chatterbox_tts import TTS

# Load the offline model once (fast)
tts = TTS(model="gpt-voice", local=True)

def synthesize_speech(text, output_path):
    """
    Generate local audio using Chatterbox offline TTS.
    """
    try:
        audio = tts.generate(text)
        audio.save(output_path)
        return output_path
    except Exception as e:
        print("TTS error:", e)
        return None
