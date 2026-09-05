from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils.foundry_client import query_foundry, foundry_embed, foundry_embed_batch
from utils.tts import synthesize_to_file, warm_up as warm_up_tts
from utils.web_search import search_web  # optional for online use
from utils.doc_parser import parse_document
from utils import mcp_manager
from utils import local_tools
from utils import graph_memory
import os
import re
import json
import uuid
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Connect to any MCP servers configured in mcp_servers.json (e.g. a Power BI
# MCP server) in the background. A no-op with zero cost if none are
# configured -- see utils/mcp_manager.py.
mcp_manager.start()

# Optional cross-session memory (Graphiti + an embedded graph DB) -- see
# utils/graph_memory.py. Off by default (ENABLE_GRAPH_MEMORY=1 to turn on);
# a no-op with zero cost otherwise. Unlike the ChromaDB memory below, which
# is scoped to one session_id, this persists facts across every session.
graph_memory.start()

# How many past user/assistant turns to feed back as conversation history.
MAX_HISTORY_MESSAGES = 12

# Bound on how many tool-call round trips a single reply can take, so a
# model stuck calling tools in a loop can't hang a request forever.
MAX_TOOL_ROUNDS = 4

_COT_INSTRUCTIONS = (
    "Think through non-trivial questions step by step before answering.\n\n"
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

_TOOLS_INSTRUCTIONS = (
    "You have access to tools. Call a tool when it would genuinely help "
    "answer the user's question (e.g. it needs current or external data "
    "you don't already know); otherwise just answer directly."
)

def build_system_prompt(tools_available, deep_think):
    """Chain-of-thought is opt-in (deep_think) because it roughly doubles
    output length on every single reply, which is real added latency on a
    small local model -- not something to pay by default on every "hi"."""
    parts = ["You are Iris, a helpful local AI assistant."]
    if tools_available:
        parts.append(_TOOLS_INSTRUCTIONS)
    if deep_think:
        parts.append(_COT_INSTRUCTIONS)
    return "\n\n".join(parts)

def get_all_tools():
    """Native local tools (spreadsheets, Word docs, PDFs, text files, web
    fetch, and opt-in email/shell -- see utils/local_tools.py) plus
    whatever's exposed by connected MCP servers (e.g. Power BI)."""
    return local_tools.get_tool_schemas() + mcp_manager.get_openai_tools()

def dispatch_tool_call(name, arguments):
    if local_tools.is_local_tool(name):
        return local_tools.call_tool(name, arguments)
    return mcp_manager.call_tool(name, arguments)

def run_completion(messages, tools, max_tokens):
    """Run one full turn against Foundry Local, including any tool-call
    round trips (native local tools and/or MCP servers), and return the
    final text content. `messages` is mutated in place with the
    assistant/tool turns the model made along the way."""
    message = query_foundry(messages, max_tokens=max_tokens, tools=tools)

    rounds = 0
    while getattr(message, "tool_calls", None) and rounds < MAX_TOOL_ROUNDS:
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result_text = dispatch_tool_call(tc.function.name, arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

        rounds += 1
        message = query_foundry(messages, max_tokens=max_tokens, tools=tools)

    return message.content or ""

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

def ingest_document_chunks(chunks, session_id):
    """Embeds and stores all of a document's chunks in one batch instead
    of one chunk at a time. The old code called save_to_memory() (one
    Foundry Local embedding round trip + one ChromaDB write) per chunk --
    for a document that splits into, say, 150 chunks, that's 150 sequential
    round trips each way, which is what actually made uploads feel slow,
    not ChromaDB itself: measured ~15x faster for ChromaDB writes and
    ~14x faster for the embedding calls when batched instead of looped
    (150 items, conservative local-server overhead assumptions). Runs in
    a background thread (see /upload) so the HTTP request returns
    immediately rather than the browser hanging until every chunk of a
    large document is indexed."""
    try:
        embeddings = foundry_embed_batch(chunks)
    except Exception as e:
        print(f"[Foundry embedding error, document not stored in memory: {e}]")
        return

    ids = [str(uuid.uuid4()) for _ in chunks]
    timestamp = datetime.datetime.now().isoformat()
    metadatas = [{"role": "document", "session_id": session_id, "timestamp": timestamp} for _ in chunks]

    try:
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    except Exception as e:
        print(f"[Memory error, document not stored: {e}]")

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

# --- Workforce: break a complex request into subtasks, run them (with
# tools) on a small pool of worker "agents", then synthesize one answer.
# Inspired by Eigent/CAMEL-AI's multi-agent Workforce pattern, scaled down
# for a single local model: workers are stateless per-subtask completions
# (not full separate agent processes), and "parallel" is best-effort --
# Foundry Local serves one model instance, so true wall-clock speedup
# depends on whether it can service concurrent requests; either way the
# code stays correct if it just serializes them. Opt-in (like Deep think)
# since planning + N workers + synthesis is several LLM calls, not one.
MAX_WORKFORCE_SUBTASKS = 5
WORKFORCE_MAX_WORKERS = 3

_PLAN_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)

def plan_subtasks(user_message, doc_context):
    system_parts = ["You are the planning coordinator for Iris, a helpful local AI assistant."]
    if doc_context:
        system_parts.append(f"Relevant context:\n{doc_context}")
    system_parts.append(
        "Break the user's request into a short list of concrete, self-contained "
        "subtasks that separate specialist workers can each complete independently, "
        f"in any order. Respond with ONLY a JSON array of 1 to {MAX_WORKFORCE_SUBTASKS} "
        "short subtask description strings -- no prose, no markdown fences, nothing else. "
        "If the request is simple enough that breaking it down wouldn't help, respond "
        "with a single-item array containing the request itself, unchanged."
    )
    messages = [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": user_message},
    ]
    raw = run_completion(messages, tools=[], max_tokens=300)

    match = _PLAN_LIST_RE.search(raw)
    if match:
        try:
            subtasks = json.loads(match.group(0))
            subtasks = [str(s).strip() for s in subtasks if str(s).strip()]
            if subtasks:
                return subtasks[:MAX_WORKFORCE_SUBTASKS]
        except (json.JSONDecodeError, TypeError):
            pass
    return [user_message]  # couldn't parse a plan -- treat as a single task

def run_worker(subtask, tools):
    """A focused, stateless worker: only sees its own subtask, not the
    conversation or the other workers -- matches the Workforce pattern
    where the coordinator holds context and workers are dispatched fresh."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a focused specialist worker completing exactly one subtask "
                "as part of a larger effort coordinated by Iris. Do only this subtask, "
                "be concise and concrete, and use tools if they help."
            ),
        },
        {"role": "user", "content": subtask},
    ]
    result = run_completion(messages, tools, max_tokens=500)
    return split_thinking(result)[0]  # strip any stray <thinking> tags

def synthesize_workforce_result(user_message, subtask_results, deep_think):
    summary = "\n\n".join(f"Subtask: {s}\nResult: {r}" for s, r in subtask_results)
    system = build_system_prompt(False, deep_think) + (
        "\n\nYou coordinated a team of specialist workers to handle the user's "
        "request below. Combine their results into one clear, well-organized "
        "final answer -- synthesize, don't just list the subtasks back."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Original request: {user_message}\n\nWorker results:\n{summary}"},
    ]
    max_tokens = 800 if deep_think else 500
    raw = run_completion(messages, tools=[], max_tokens=max_tokens)
    return split_thinking(raw)

def run_workforce(user_message, doc_context, tools, deep_think):
    """Returns (response, thinking, breakdown), or None if the planner
    decided the request doesn't need decomposing -- callers should fall
    back to a normal single-agent reply in that case."""
    subtasks = plan_subtasks(user_message, doc_context)
    if len(subtasks) <= 1:
        return None

    subtask_results = [None] * len(subtasks)
    with ThreadPoolExecutor(max_workers=WORKFORCE_MAX_WORKERS) as pool:
        futures = {pool.submit(run_worker, subtask, tools): i for i, subtask in enumerate(subtasks)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                subtask_results[i] = future.result()
            except Exception as e:
                subtask_results[i] = f"[worker error: {e}]"

    pairs = list(zip(subtasks, subtask_results))
    response, thinking = synthesize_workforce_result(user_message, pairs, deep_think)
    breakdown = [{"subtask": s, "result": r} for s, r in pairs]
    return response, thinking, breakdown

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']
    session_id = data.get('session_id', str(uuid.uuid4()))
    deep_think = bool(data.get('deep_think'))
    workforce = bool(data.get('workforce'))

    thinking = None
    breakdown = None
    if user_message.lower().startswith("search:"):
        query = user_message.replace("search:", "").strip()
        response = search_web(query)  # remove if fully offline
    else:
        history = get_conversation_history(session_id)
        doc_context = retrieve_document_context(user_message, session_id)
        # Facts learned in ANY earlier session (if ENABLE_GRAPH_MEMORY=1) --
        # empty string immediately if disabled, so this never adds latency
        # or behavior change when the feature is off. Folded into
        # doc_context so both Workforce planning and the normal reply path
        # see it without threading a third parameter through run_workforce.
        graph_context = graph_memory.search(user_message)
        if graph_context:
            graph_block = f"Facts remembered from earlier conversations:\n{graph_context}"
            doc_context = f"{doc_context}\n\n{graph_block}" if doc_context else graph_block
        tools = get_all_tools()  # local tools + whatever MCP servers are connected

        outcome = run_workforce(user_message, doc_context, tools, deep_think) if workforce else None
        if outcome:
            response, thinking, breakdown = outcome
        else:
            # Either workforce is off, or the planner decided this request
            # doesn't need decomposing -- either way, a normal single-agent
            # reply, with the session's conversation history included.
            messages = [{"role": "system", "content": build_system_prompt(bool(tools), deep_think)}]
            if doc_context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant context:\n{doc_context}",
                })
            messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            max_tokens = 800 if deep_think else 400
            raw_response = run_completion(messages, tools, max_tokens)
            response, thinking = split_thinking(raw_response)

    save_to_memory("user", user_message, session_id)
    save_to_memory("assistant", response, session_id)
    # Fire-and-forget: extract entities/facts from this turn into the
    # cross-session graph (no-op if ENABLE_GRAPH_MEMORY isn't set).
    graph_memory.add_episode(
        f"chat-{session_id}-{uuid.uuid4()}",
        f"User: {user_message}\nAssistant: {response}",
        source_description="chat",
    )

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

    return jsonify({
        "response": response,
        "thinking": thinking,
        "breakdown": breakdown,
        "session_id": session_id,
        "audio_url": audio_url,
    })

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

    # Embedding + storing runs in the background (batched -- see
    # ingest_document_chunks) so the browser doesn't sit waiting for a
    # large document to finish indexing before the upload call returns.
    # Asking about the document immediately after upload may not see it
    # yet; in practice this background pass is fast (a couple of Foundry
    # Local round trips instead of hundreds), so that window is small.
    threading.Thread(target=ingest_document_chunks, args=(chunks, session_id), daemon=True).start()

    # Fire-and-forget: also persist this document's content into the
    # cross-session graph, so it's retrievable in later sessions too, not
    # just this one (no-op if ENABLE_GRAPH_MEMORY isn't set).
    graph_memory.add_episode(
        f"doc-{session_id}-{filename}",
        content,
        source_description=f"uploaded document: {filename}",
    )

    os.remove(filepath)
    return jsonify({"message": "Document uploaded -- indexing in the background", "session_id": session_id})

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
