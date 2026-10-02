from flask import Flask, render_template, request, jsonify
from utils.ollama_client import query_ollama, ollama_embed
from utils.web_search import search_web  # optional for online use
from utils.doc_parser import parse_document
import os
import re
import sys
import uuid
import shutil
import getpass
import datetime
from chromadb import PersistentClient
import whisper
import pyttsx3
import threading
import queue

# Resolve paths from this file so the app works no matter which directory it is started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['AUDIO_FOLDER'] = os.path.join(BASE_DIR, 'audio')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Initialize ChromaDB with persistent storage
chroma_client = PersistentClient(path=os.path.join(BASE_DIR, "chroma_store"))
collection = chroma_client.get_or_create_collection("chat_memory")

# Initialize Whisper model ("tiny", "base", "small", "medium", "large"; bigger = more RAM)
whisper_model = whisper.load_model(os.environ.get("WHISPER_MODEL", "base"))

# Whisper decodes audio with the ffmpeg command-line tool
if shutil.which("ffmpeg") is None:
    print("WARNING: ffmpeg not found on PATH, voice input will fail. "
          "Install it (Ubuntu: sudo apt install ffmpeg, Windows: winget install ffmpeg).")

def save_to_memory(role, text, session_id, response=None):
    embedding = ollama_embed(text)
    doc_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    metadata = {
        "role": role,
        "session_id": session_id,
        "timestamp": timestamp
    }
    if role == "user" and response is not None:
        metadata["response"] = response

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[metadata]
    )

def retrieve_context(query, top_k=5):
    query_embedding = ollama_embed(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n".join(results["documents"][0]) if results["documents"] else ""

def _pick_voice(voices):
    def is_english(voice):
        langs = " ".join(str(lang) for lang in (getattr(voice, 'languages', None) or []))
        return 'english' in voice.name.lower() or re.search(r"\ben(\b|_)", langs.lower()) is not None

    # Select a natural-sounding female voice: Zira on Windows, any English "Female" voice elsewhere
    for voice in voices:
        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
            return voice
    for voice in voices:
        if str(getattr(voice, 'gender', '')).lower() == 'female' and is_english(voice):
            return voice
    # eSpeak NG (Linux) lists every language and starts with Afrikaans, so pick English explicitly
    english = [voice for voice in voices if is_english(voice)]
    for voice in english:
        if 'america' in voice.name.lower() or voice.id.lower().endswith('en-us'):
            return voice
    return english[0] if english else None  # None keeps the engine default

def _init_tts():
    # Offline TTS: SAPI5 on Windows, eSpeak NG on Linux. Set SERVER_TTS=0 to disable it.
    if os.environ.get("SERVER_TTS", "1").lower() in ("0", "false", "no", "off"):
        return None
    try:
        engine = pyttsx3.init()
    except Exception as e:
        hint = " Install it with: sudo apt install espeak-ng" if sys.platform.startswith("linux") else ""
        print(f"WARNING: offline TTS unavailable ({e}), the server will not speak responses.{hint}")
        return None

    try:
        voice = _pick_voice(engine.getProperty('voices') or [])
        if voice is not None:
            voice_id = voice.id
            # eSpeak NG voices are male; its "+f3" variant turns the chosen voice into a female one
            if sys.platform.startswith("linux") and str(voice.gender).lower() != 'female':
                voice_id += "+f3"
            engine.setProperty('voice', voice_id)
    except Exception as e:
        print(f"TTS voice selection failed, using the default voice: {e}")
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)
    return engine

tts_engine = _init_tts()
_tts_queue = queue.Queue()

def _tts_worker():
    # pyttsx3 is not thread-safe, so one worker thread speaks queued responses in order
    while True:
        text = _tts_queue.get()
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

if tts_engine is not None:
    threading.Thread(target=_tts_worker, daemon=True).start()

def speak_offline(text):
    # Queue the text so the request is not blocked while speaking
    if tts_engine is not None:
        _tts_queue.put(text)

def save_upload(storage, folder):
    # Never use the client's filename as a path (it could be "../.." or "C:\..."); keep only a
    # plain extension, which the document parser needs to detect the file type
    ext = os.path.splitext(storage.filename or "")[1].lower()
    if not (ext[1:].isascii() and ext[1:].isalnum()):
        ext = ""
    filepath = os.path.join(folder, uuid.uuid4().hex + ext)
    storage.save(filepath)
    return filepath

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get_username')
def get_username():
    # getpass reads USERNAME on Windows and USER/LOGNAME on Linux
    try:
        return jsonify({"username": getpass.getuser()})
    except Exception:
        return jsonify({"username": None})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']
    session_id = data.get('session_id', str(uuid.uuid4()))

    if user_message.lower().startswith("search:"):
        query = user_message.replace("search:", "").strip()
        response = search_web(query)  # remove if fully offline
    else:
        context = retrieve_context(user_message)
        prompt = f"{context}\n\nUser: {user_message}\nAssistant:"
        response = query_ollama(prompt)

    save_to_memory("user", user_message, session_id, response)
    save_to_memory("assistant", response, session_id)

    speak_offline(response)  # respond with TTS

    return jsonify({"response": response, "session_id": session_id})

@app.route('/upload', methods=['POST'])
def upload_doc():
    file = request.files['file']
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    filepath = save_upload(file, app.config['UPLOAD_FOLDER'])
    try:
        content = parse_document(filepath)
    finally:
        os.remove(filepath)

    chunk_size = 1000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    for chunk in chunks:
        save_to_memory("document", chunk, session_id)

    return jsonify({"message": "Document uploaded and stored in memory successfully", "session_id": session_id})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio = request.files['audio']
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    filepath = save_upload(audio, app.config['AUDIO_FOLDER'])
    try:
        # fp16 only works on a CUDA GPU; on CPU this avoids Whisper's FP32 fallback warning
        result = whisper_model.transcribe(filepath, fp16=whisper_model.device.type == "cuda")
    finally:
        os.remove(filepath)
    text = result['text']

    return jsonify({"message": "Audio transcribed successfully", "text": text, "session_id": session_id})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
