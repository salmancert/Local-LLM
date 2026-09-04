from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils.foundry_client import query_foundry, foundry_embed
from utils.tts import synthesize_to_file
from utils.web_search import search_web  # optional for online use
from utils.doc_parser import parse_document
import os
import uuid
import datetime
from chromadb import PersistentClient
from chromadb.config import Settings
import whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'audio/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Initialize ChromaDB with persistent storage. anonymized_telemetry is
# disabled so nothing leaves the machine -- this app is meant to run fully
# offline/locally.
chroma_client = PersistentClient(
    path="chroma_store",
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection("chat_memory")

# Initialize Whisper model
whisper_model = whisper.load_model("base")  # or "small", "medium", "large"

def save_to_memory(role, text, session_id, response=None):
    try:
        embedding = foundry_embed(text)
    except Exception as e:
        print(f"[Foundry embedding error, skipping memory save: {e}]")
        return

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
    try:
        query_embedding = foundry_embed(query)
    except Exception as e:
        print(f"[Foundry embedding error, skipping context retrieval: {e}]")
        return ""

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n".join(results["documents"][0]) if results["documents"] else ""

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
        response = query_foundry(prompt)

    save_to_memory("user", user_message, session_id, response)
    save_to_memory("assistant", response, session_id)

    # Synthesize the reply to a local audio file the browser can play.
    # TTS runs fully offline (see utils/tts.py); if it fails for any
    # reason, the frontend falls back to the browser's own speech synthesis.
    audio_url = None
    try:
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        synthesize_to_file(response, audio_path)
        audio_url = f"/audio/{audio_filename}"
    except Exception as e:
        print(f"[TTS error: {e}]")

    return jsonify({"response": response, "session_id": session_id, "audio_url": audio_url})

@app.route('/audio/<path:filename>')
def get_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename, mimetype="audio/wav")

@app.route('/upload', methods=['POST'])
def upload_doc():
    file = request.files['file']
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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

    filename = secure_filename(audio.filename) or f"{uuid.uuid4()}.webm"
    filepath = os.path.join(app.config['AUDIO_FOLDER'], filename)
    audio.save(filepath)

    result = whisper_model.transcribe(filepath)
    text = result['text']

    os.remove(filepath)

    return jsonify({"message": "Audio transcribed successfully", "text": text, "session_id": session_id})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
