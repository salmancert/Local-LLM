from flask import Flask, render_template, request, jsonify
from utils.ollama_client import query_ollama
from utils.web_search import search_web  # optional for online use
from utils.doc_parser import parse_document
import os
import uuid
import datetime
import requests
from chromadb import PersistentClient
import whisper
import pyttsx3
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'audio/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Initialize ChromaDB with persistent storage
chroma_client = PersistentClient(path="chroma_store")
collection = chroma_client.get_or_create_collection("chat_memory")

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # or "small", "medium", "large"

# Initialize offline TTS
tts_engine = pyttsx3.init()

def ollama_embed(text):
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return response.json()['embedding']

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

def _speak(text):
    try:
        # Select a natural-sounding female voice
        voices = tts_engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                tts_engine.setProperty('voice', voice.id)
                break
        else:
            tts_engine.setProperty('voice', voices[0].id)  # fallback

        tts_engine.setProperty('rate', 170)
        tts_engine.setProperty('volume', 1.0)

        tts_engine.say(text)
        tts_engine.runAndWait()
    except RuntimeError as e:
        print(f"TTS error: {e}")

def speak_offline(text):
    # Start a new thread to prevent blocking and loop issues
    threading.Thread(target=_speak, args=(text,), daemon=True).start()
    
@app.route('/')
def index():
    return render_template('chat.html')

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

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    content = parse_document(filepath)
    chunk_size = 1000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    for chunk in chunks:
        save_to_memory("document", chunk, session_id)

    os.remove(filepath)
    return jsonify({"message": "Document uploaded and stored in memory successfully", "session_id": session_id})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio = request.files['audio']
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    filepath = os.path.join(app.config['AUDIO_FOLDER'], audio.filename)
    audio.save(filepath)

    result = whisper_model.transcribe(filepath)
    text = result['text']

    os.remove(filepath)

    return jsonify({"message": "Audio transcribed successfully", "text": text, "session_id": session_id})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
