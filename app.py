from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils.foundry_client import query_foundry, foundry_embed
from utils.tts import synthesize_to_file, warm_up as warm_up_tts
from utils.web_search import search_web  # optional for online use
from utils.doc_parser import parse_document
import os
import re
import uuid
import datetime
import threading
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

# Warm up the TTS engine in the background so any one-time model-load /
# backend-init cost happens now, not during the user's first chat request
# (which is what made the first reply's audio fall back to pyttsx3).
threading.Thread(target=warm_up_tts, daemon=True).start()

SYSTEM_PROMPT = (
    "You are Iris, a helpful local AI assistant. Think through non-trivial "
    "questions step by step before answering.\n\n"
    "Respond using exactly this format:\n"
    "<thinking>\n"
    "Your step-by-step reasoning here.\n"
    "</thinking>\n"
    "<answer>\n"
    "Your final answer to the user here, written normally.\n"
    "</answer>\n\n"
    "For simple greetings or small talk, keep the thinking section brief. "
    "Always include both tags."
)

# How many past user/assistant turns to feed back as conversation history.
MAX_HISTORY_MESSAGES = 12

_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def split_thinking(raw_response):
    """Pull the <thinking>/<answer> blocks apart. Falls back gracefully if
    the model didn't follow the format (small local models don't always)."""
    thinking_match = _THINKING_RE.search(raw_response)
    answer_match = _ANSWER_RE.search(raw_response)
    thinking = thinking_match.group(1).strip() if thinking_match else None

    if answer_match:
        answer = answer_match.group(1).strip()
    elif thinking_match:
        # No (or truncated) </answer> tag -- use whatever follows thinking.
        answer = raw_response[thinking_match.end():].strip() or thinking
    else:
        answer = raw_response.strip()

    return answer, thinking

def save_to_memory(role, text, session_id):
    try:
        embedding = foundry_embed(text)
    except Exception as e:
        print(f"[Foundry embedding error, skipping memory save: {e}]")
        return

    doc_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[{"role": role, "session_id": session_id, "timestamp": timestamp}]
    )

def get_conversation_history(session_id, limit=MAX_HISTORY_MESSAGES):
    """The actual chronological back-and-forth for this session, so the
    model can follow up on earlier turns -- not a semantic search, which
    misses follow-ups that don't share vocabulary with earlier messages."""
    try:
        results = collection.get(
            where={"$and": [{"session_id": session_id}, {"role": {"$in": ["user", "assistant"]}}]},
        )
    except Exception as e:
        print(f"[Memory error, starting fresh conversation: {e}]")
        return []

    turns = sorted(zip(results["documents"], results["metadatas"]), key=lambda t: t[1]["timestamp"])
    messages = [{"role": meta["role"], "content": text} for text, meta in turns]
    return messages[-limit:]

def retrieve_document_context(query, session_id, top_k=3):
    """Semantic search over this session's uploaded documents only."""
    try:
        query_embedding = foundry_embed(query)
    except Exception as e:
        print(f"[Foundry embedding error, skipping document context: {e}]")
        return ""

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"$and": [{"session_id": session_id}, {"role": "document"}]},
        )
    except Exception as e:
        print(f"[Memory error, skipping document context: {e}]")
        return ""

    docs = results.get("documents") or []
    return "\n".join(docs[0]) if docs and docs[0] else ""

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']
    session_id = data.get('session_id', str(uuid.uuid4()))

    thinking = None
    if user_message.lower().startswith("search:"):
        query = user_message.replace("search:", "").strip()
        response = search_web(query)  # remove if fully offline
    else:
        history = get_conversation_history(session_id)
        doc_context = retrieve_document_context(user_message, session_id)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if doc_context:
            messages.append({
                "role": "system",
                "content": f"Relevant context from the user's uploaded documents:\n{doc_context}",
            })
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        raw_response = query_foundry(messages, max_tokens=800)
        response, thinking = split_thinking(raw_response)

    save_to_memory("user", user_message, session_id)
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

    return jsonify({"response": response, "thinking": thinking, "session_id": session_id, "audio_url": audio_url})

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
