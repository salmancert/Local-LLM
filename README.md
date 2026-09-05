# AI-Powered Conversational Assistant with Document Memory and Voice Interaction

This project implements a web-based conversational AI assistant that combines document understanding, voice interaction, and contextual memory. It provides a natural language interface for document querying, web search, and general conversation with both text and voice support.

The assistant leverages local language models through [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local), maintains conversation context using ChromaDB for semantic search, and supports voice interaction through offline text-to-speech and speech recognition capabilities. It can also call tools (native local tools and MCP servers -- spreadsheets, documents, Power BI, browser automation, and more), decompose complex requests across a small team of agents (Workforce mode), remember facts across sessions in an optional knowledge graph (Cross-Session Memory), and hold a fully hands-free spoken conversation (Voice Mode). The system is designed to run entirely offline, with telemetry disabled, making it suitable for environments with limited internet connectivity while still providing optional web search functionality.

## Repository Structure
```
.
├── app.py                 # Main Flask application with routing and core logic
├── offline script.py      # Utility for offline document processing and embedding
├── static/               # Static web assets
│   └── style.css        # CSS styling for the chat interface
├── templates/           # HTML templates
│   └── chat.html       # Main chat interface template
├── models/
│   └── kokoro/           # Bundled Kokoro TTS model weights (see Text-to-Speech below)
├── mcp_servers.example.json  # Template for configuring MCP tool servers (e.g. Power BI, Outlook)
├── workspace/            # Sandbox for local_tools.py's file tools (created at runtime, gitignored)
└── utils/              # Utility modules
    ├── doc_parser.py       # Document parsing functionality
    ├── embedding_store.py  # Standalone ChromaDB + sentence-transformers helper
    ├── foundry_client.py   # Client for local Foundry Local inference/embeddings
    ├── local_tools.py       # Native tools: spreadsheets, Word, PDF, notepad, archives, web fetch, calc, email, shell
    ├── mcp_manager.py       # Generic MCP client -- connects configured tool servers
    ├── graph_memory.py      # Optional cross-session memory (Graphiti + embedded graph DB)
    ├── tts.py               # Offline text-to-speech (Kokoro / pyttsx3)
    └── web_search.py        # Optional web search functionality
```

## Usage Instructions
### Prerequisites
- Python 3.10 or higher
- [Foundry Local](https://github.com/microsoft/Foundry-Local) installed and on your `PATH`
- Flask, ChromaDB, Whisper, pyttsx3, PyMuPDF (installed via `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/salmancert/Local-LLM.git
cd Local-LLM

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install required packages
pip install -r requirements.txt

# Install Foundry Local (if not already installed)
# Follow instructions at: https://github.com/microsoft/Foundry-Local#quickstart
```

### Quick Start

Foundry Local does not need to be started manually -- `utils/foundry_client.py` starts/attaches to the local service automatically the first time a chat or embedding request is made, downloading the model on first use if necessary (this can take a while the first time).

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

### Configuration

All configuration is via environment variables; sensible local defaults are used if they're unset.

| Variable | Default | Purpose |
|---|---|---|
| `FOUNDRY_MODEL` | `phi-4-mini` | Foundry Local catalog alias used for chat completions. 3.8B, a generation newer than the previous default (qwen2.5-1.5b) and built with native tool calling as a first-class capability -- worth it now that tool calling/MCP/Workforce all depend on the model reliably deciding when to call a tool. It's larger, so not literally faster in tokens/sec on identical hardware; "faster" here means staying CPU-practical rather than jumping to a 7B+ model. Override to trade back down for raw speed (`qwen2.5-0.5b`/`qwen2.5-1.5b`) or up for capability (`qwen2.5-7b`, `phi-4`) |
| `FOUNDRY_EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | Foundry Local catalog alias used for embeddings |
| `FOUNDRY_ENDPOINT` | *(auto-discovered)* | Overrides the endpoint instead of discovering it via the SDK (e.g. a remote Foundry Local instance) |
| `FOUNDRY_API_KEY` | *(none)* | API key to send when `FOUNDRY_ENDPOINT` is set |
| `TTS_ENGINE` | `auto` | `auto` (use Kokoro if installed, else pyttsx3), `kokoro` (natural voice), or `pyttsx3` (always available, skips Kokoro) |
| `TTS_VOICE` | `af_heart` | Kokoro voice name (see [voice list](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)) |
| `TTS_LANG` | `en-us` | Kokoro language (e.g. `en-us`, `en-gb`, `fr-fr`, `ja`, `zh` -- must match the chosen voice) |
| `KOKORO_MODEL_PATH` | `models/kokoro/kokoro-v1.0.int8.onnx` | Override the bundled Kokoro model file |
| `KOKORO_VOICES_PATH` | `models/kokoro/voices-v1.0.bin` | Override the bundled Kokoro voices file |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | *(unset)* | Enables the `local__send_email` tool when all of `SMTP_HOST`/`SMTP_USER`/`SMTP_PASSWORD` are set (`SMTP_PORT` defaults to `587`) |
| `ENABLE_SHELL_TOOL` | `false` | Set `true` to enable `local__run_shell_command` -- read the Tool Calling section before turning this on |
| `SHELL_TOOL_TIMEOUT` | `30` | Seconds before a shell command is killed, when the shell tool is enabled |
| `ENABLE_GRAPH_MEMORY` | `false` | Set `true`/`1` to enable cross-session memory (see below). Requires `pip install "graphiti-core[falkordblite]"` and Python 3.12+ |
| `GRAPH_MEMORY_DB_PATH` | `graph_memory.db` | Where the embedded graph database file is stored, when graph memory is enabled |

Make sure the model aliases you configure are actually available in your Foundry Local catalog (`foundry model list`); if an embedding call fails (e.g. the alias isn't available), the app logs a warning and continues without that memory/context lookup rather than crashing the chat request.

### More Detailed Examples

1. Document Upload and Query:
```python
# Upload a document through the web interface
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload

# Query the document
curl -X POST -H "Content-Type: application/json" \
     -d '{"message": "What does the document say about X?"}' \
     http://localhost:8000/chat
```

2. Voice Interaction:
```python
# Upload audio for transcription
curl -X POST -F "audio=@your_recording.wav" http://localhost:8000/upload_audio
```

### Text-to-Speech

Every `/chat` response is synthesized to a local WAV file and returned as `audio_url`, which the browser plays directly -- nothing is sent to a remote TTS service. Two engines are supported (see `utils/tts.py`):

- **[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)** (default) -- an 82M-parameter neural TTS model that sounds substantially more natural than a classic OS voice, while still being small and fast enough to run in real time on CPU. It runs via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (onnxruntime -- no PyTorch) against the model files bundled directly in `models/kokoro/`:
  - `kokoro-v1.0.int8.onnx` (~89 MB, int8-quantized)
  - `voices-v1.0.bin` (~27 MB, all voices)

  These are committed to the repo (both under GitHub's 100 MB per-file limit, so no Git LFS needed) precisely so **no network access is required at runtime** -- not even to GitHub or Hugging Face. That matters because Hugging Face is blocked on some networks/orgs (the original `kokoro` package downloads weights from `huggingface.co` on first use, which fails in that environment); `kokoro-onnx`'s upstream releases are hosted on GitHub instead, which is how these files were fetched, and once they're in your clone nothing needs to be fetched again.

  `TTS_ENGINE=auto` (the default) uses Kokoro automatically -- no extra setup needed beyond `pip install -r requirements.txt`. Pick a different voice/language with `TTS_VOICE` / `TTS_LANG` (see the table above).
- **pyttsx3** (fallback) -- uses the OS's built-in voices. Robotic-sounding but lightweight, no model files, and always available, so it's what keeps voice replies working if Kokoro fails to load for any reason (e.g. the model files are missing or `kokoro-onnx` isn't installed).

If TTS fails entirely, the frontend falls back to the browser's built-in `speechSynthesis` API.

### Conversation Memory

Each `/chat` request rebuilds the actual chronological conversation for that `session_id` (up to the last `MAX_HISTORY_MESSAGES` turns, in `app.py`) from ChromaDB and sends it to Foundry Local as proper multi-turn messages, so follow-up questions ("what about the second one?") work the way they would in any chat app. This is deliberately *not* semantic search -- a follow-up often shares no vocabulary with the turn it's following up on, so similarity search alone misses it. Semantic search is still used, but only for retrieving relevant chunks from documents *you uploaded in that same session* (`retrieve_document_context` in `app.py`), which is a different problem (finding the right needle in a large haystack) than recalling what was just said.

**Document ingestion is batched, not per-chunk.** The original implementation called `save_to_memory()` -- one Foundry Local embedding HTTP call *and* one ChromaDB write -- separately for every ~1000-character chunk of an uploaded document; a document that splits into, say, 150 chunks meant 150 sequential round trips each way, which is what actually made uploads feel slow (ChromaDB itself was never the bottleneck). `ingest_document_chunks()` in `app.py` now embeds all of a document's chunks in one `foundry_embed_batch()` call (`utils/foundry_client.py`) and writes them to ChromaDB in one `collection.add()` call, and runs the whole thing in a background thread so `/upload` returns immediately instead of the browser hanging until indexing finishes. Measured directly (150 chunks, real ChromaDB, a stand-in local embedding server): batching cut ChromaDB write time ~15x and embedding-call time ~14x -- both were tested here since real Foundry Local doesn't run in this project's dev sandbox, but the mechanism (fewer round trips, less fixed per-call overhead paid over and over) is exactly what would apply against it too. One consequence of running in the background: asking about a just-uploaded document in the first second or two may not see it yet, since indexing hasn't finished -- in practice this window is now small (a couple of round trips instead of hundreds).

### Chain-of-Thought Reasoning (opt-in)

Click **Deep think** in the control bar to have the model reason step by step in a `<thinking>...</thinking>` block before giving its `<answer>...</answer>` (sent as `deep_think: true` on `/chat`; see `build_system_prompt` in `app.py`). The backend splits these apart: only the `<answer>` text is spoken (TTS), shown as the main reply, and stored back into conversation memory, while the reasoning is returned separately as `thinking` and rendered as a collapsed "Show reasoning" toggle under the reply. Small local models don't always follow the format perfectly -- if the tags are missing or truncated, the code falls back gracefully to using the raw response as the answer with no reasoning trace.

**Off by default.** Asking the model to write out a full reasoning trace before every answer roughly doubles output length, which is real, noticeable latency on a small model running locally on CPU -- not something worth paying on every "hi". With Deep think off, replies use a short, direct system prompt and a lower token cap (`max_tokens=400` vs `800`), so the fast path stays fast; Deep think is there for when you actually want the model to slow down and work through something carefully.

### Tool Calling

Iris can call tools -- Foundry Local's chat completions API supports standard OpenAI-style `tools`/`tool_calls` for models tagged with the `tools` task in `foundry model list` (the default model, `phi-4-mini`, is one, and was picked partly for this). Tools come from two places, merged into one list by `get_all_tools()` in `app.py`:

- **Native local tools** (`utils/local_tools.py`) -- always available, no setup, no external process. Run in-process against a sandboxed `workspace/` folder in the repo root.
- **MCP servers** (`utils/mcp_manager.py`) -- anything you configure in `mcp_servers.json`, e.g. Power BI.

**How it works end to end:** every tool -- local or MCP -- is exposed to Foundry Local as an OpenAI-format function definition (`local__toolname` for native tools, `servername__toolname` for MCP ones, so names never collide). When the model responds with a tool call, `run_completion()` in `app.py` dispatches it to the right place, feeds the result back as a `tool` message, and asks the model to continue -- repeating up to `MAX_TOOL_ROUNDS` (4) times before giving up, so a model stuck in a call loop can't hang a request forever.

Small local models are not always reliable at deciding when/how to call tools -- expect to iterate on tool names/descriptions if calls aren't happening when you'd expect, and expect the occasional malformed call (handled: JSON-parse failures fall back to `{}` arguments rather than crashing the request).

#### Native tools: spreadsheets, Word, PDF, notepad, archives, web browsing, and more

| Tool | Does |
|---|---|
| `local__read_text_file` / `local__write_text_file` | Plain text files (Notepad-like) |
| `local__copy_file` / `local__move_file` / `local__delete_file` | Basic file management |
| `local__read_pdf` / `local__create_pdf` | Read a PDF's text; create a new PDF from text |
| `local__highlight_spelling_errors` | Proofread a PDF *on its own*: finds likely misspelled words (offline dictionary lookup via `pyspellchecker` -- no network call) and saves a copy with each one highlighted right on the page, using PyMuPDF's per-word coordinates to place the highlight precisely. Dictionary-based, not grammar-aware -- it won't catch real-word errors like "there" for "their", and can occasionally flag a genuine but obscure proper noun/technical term; only offered if `pyspellchecker` is installed |
| `local__read_spreadsheet` / `local__write_spreadsheet` | `.csv`, `.xlsx`, `.xlsm` -- read returns tab-separated rows (optionally from one named sheet), write creates a fresh single-sheet file |
| `local__write_spreadsheet_sheet` / `local__list_spreadsheet_sheets` | Add or replace one sheet in an `.xlsx` workbook without touching its others -- unlike `write_spreadsheet`, calls accumulate instead of overwriting the file each time |
| `local__split_spreadsheet_by_column` | One call: split a spreadsheet into one sheet per unique value in a chosen column (e.g. "one sheet per name") -- e.g. a timesheet with a Name column becomes one workbook with an Alice sheet, a Bob sheet, etc. Groups deterministically in Python rather than asking the model to loop the call above once per name |
| `local__read_word_document` / `local__write_word_document` | `.docx` |
| `local__create_zip_archive` / `local__extract_zip_archive` | `.zip`, files or whole folders |
| `local__create_tar_archive` / `local__extract_tar_archive` | `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2` |
| `local__extract_rar_archive` | `.rar` extraction only -- RAR is proprietary with no free encoder, so there's no create; only offered if the `rarfile` package is installed (also needs a system `unrar`/`unar` binary on `PATH` to actually decompress) |
| `local__calculate` | Arithmetic (`+ - * / // % **`), evaluated by a whitelisted AST walk -- no `eval()`, so it can't be turned into code execution regardless of what expression the model passes |
| `local__get_current_datetime` | The model doesn't otherwise reliably know today's date |
| `local__fetch_webpage` | Fetch a URL, strip scripts/nav/styling, return readable text -- this is "web browsing": research and reading, not clicking/filling forms (see Playwright below for that) |
| `local__list_workspace_files` | List what's in the workspace |

**Everything file-related above is sandboxed to `workspace/`** (created automatically, gitignored). Every path a tool touches is resolved and checked to stay inside it -- no `..` traversal, no absolute paths -- so a tool call can read or overwrite files there and nowhere else on your machine. Archive extraction gets the same treatment against "zip-slip"/"tar-slip": every member's path is checked before extracting, since a crafted archive with a member named e.g. `../../.bashrc` would otherwise happily write outside the destination (confirmed this actually works as a real attack against `zipfile`/`tarfile` if you don't check first -- both extract wherever a member name points, no questions asked). All of this matters because tool calls are LLM-decided, and the model's context can include content from outside the conversation (a fetched web page, an MCP tool's output, an uploaded document, a downloaded archive) -- treat that the same as any other prompt-injection surface. The sandbox is the actual safety boundary here, not a suggestion to the model.

Two more native tools exist but are **off unless you configure them**:

- `local__send_email` -- only offered to the model if `SMTP_HOST`, `SMTP_USER`, and `SMTP_PASSWORD` are set (works with Gmail, Outlook.com, or any SMTP provider's app-password auth). For reading/managing an actual Outlook inbox or calendar, use an MCP server instead (see below) -- that needs Microsoft Graph/OAuth, which is a better fit for a dedicated, already-vetted server than something to hand-roll here.
- `local__run_shell_command` -- **off by default**, and worth actually deciding rather than flipping on reflexively. Set `ENABLE_SHELL_TOOL=true` to enable it; it runs PowerShell on Windows or `/bin/sh` elsewhere, in the `workspace/` directory, with a timeout (`SHELL_TOOL_TIMEOUT`, default 30s). This is arbitrary command execution decided by an LLM that can be steered by anything in its context (a malicious web page it fetched, a crafted document, a compromised MCP tool result) -- there's no reliable way to sandbox "run whatever command the model wants" the way file paths can be sandboxed. Only turn it on if you understand and accept that, and note that `app.py` binds Flask to `0.0.0.0` by default (see Troubleshooting) -- if this machine is reachable on your network, so is anything this tool can do.

#### MCP servers: Power BI, Outlook, browser automation, and anything else

For integrations that need a real running application or an OAuth app registration, point `utils/mcp_manager.py` at a dedicated MCP server instead of reimplementing that integration here. It's generic -- it doesn't know or care what server it's talking to.

**Setup:**
1. Install the server globally with `npm`, e.g. `npm install -g @microsoft/powerbi-modeling-mcp`. **Prefer this over invoking through `npx`** -- confirmed against the real Power BI and Playwright MCP servers: `npx -y <package>` has to check the npm registry to resolve which version to run before it can start, and on a slow or restricted network that check can hang indefinitely with no error, silently breaking the connection. A global install needs that check only once (at install time), and the server then starts in under a second every time after. If you'd rather not install globally, pin an exact version with `npx` (`npx -y <package>@<exact-version>`) instead of leaving it unpinned -- it's still one network check per launch, but at least a predictable one.
2. Copy `mcp_servers.example.json` to `mcp_servers.json` (gitignored -- it may end up holding paths or credentials specific to your machine). Only add the servers you actually want; every entry gets launched and connected to at startup.
3. Add an entry per server, pointing `command`/`args`/`env` at the installed binary -- follow that server's own README for auth/setup:
   - **Power BI**: [microsoft/powerbi-modeling-mcp](https://github.com/microsoft/powerbi-modeling-mcp) (confirmed working end to end against the real server -- see below) or [sulaiman013/powerbi-mcp](https://github.com/sulaiman013/powerbi-mcp)
     ```json
     "powerbi": { "command": "powerbi-modeling-mcp", "args": ["--start"], "env": {} }
     ```
     The `--start` flag matters: without it, the server prints an interactive "Press any key to close" banner and crashes immediately, because stdin isn't a real terminal once it's launched as an MCP subprocess.
   - **Outlook / Microsoft 365** (mail, calendar, Excel, more): [Softeria/ms-365-mcp-server](https://github.com/Softeria/ms-365-mcp-server)
     ```json
     "outlook": { "command": "ms-365-mcp-server", "args": [], "env": {} }
     ```
   - **Browser automation** (clicking, filling forms, JS-rendered pages -- beyond what `local__fetch_webpage` can do): Microsoft's official [Playwright MCP](https://github.com/microsoft/playwright-mcp) (also confirmed working the same way -- global install, no `npx`)
     ```json
     "browser": { "command": "playwright-mcp", "args": [], "env": {} }
     ```
4. Restart the app and check its startup log for `[MCP: connected to '<name>' (N tool(s))]` -- that confirms the handshake actually succeeded (a stuck `npx` connection instead just never prints anything, which is exactly the silent-hang failure mode step 1 avoids). Connecting happens in the background; this never blocks a request -- a chat that arrives before a server finishes connecting just proceeds without tools for that one turn.

**Confirmed working for real** (not just against a scripted stand-in): connected to the actual published `@microsoft/powerbi-modeling-mcp` package and got back its real 21 tools (`measure_operations`, `table_operations`, `dax_query_operations`, `relationship_operations`, and so on, namespaced as `powerbi__measure_operations` etc.), correctly surfaced through this app's own `get_all_tools()`. What isn't tested here: actually querying a real Power BI/Fabric dataset -- that server authenticates interactively via a browser OAuth flow (`AuthenticationMode=InteractiveBrowser`), which needs a real desktop session to complete, and this development environment doesn't have Foundry Local or a Power BI tenant available to drive that flow end-to-end.

**Zero cost when unused.** With no `mcp_servers.json` and none of the opt-in native tools configured, `get_all_tools()` returns `[]` and no `tools` parameter is even sent to Foundry Local -- behavior and latency are identical to not having this feature at all.

### Workforce: Multi-Agent Task Orchestration (opt-in)

Click **Workforce** to have complex requests handled by a small team of agents instead of one reply -- the same idea as [Eigent](https://github.com/eigent-ai/eigent)'s multi-agent Workforce (built on CAMEL-AI), scaled down to fit a single local model. See `run_workforce` and friends in `app.py`:

1. **Plan** -- a coordinator call asks the model to break your request into a short list of concrete, independent subtasks (a JSON array). If it decides the request doesn't need breaking down, Workforce mode gets out of the way and you get a normal reply instead -- no orchestration tax on a simple message just because the toggle is on.
2. **Work** -- each subtask is dispatched to a stateless worker (its own fresh completion, with its own `<thinking>`-free system prompt) that can call MCP tools just like the main chat can -- so, for example, three workers could each pull a different Power BI metric in parallel. Workers run on a small thread pool (`WORKFORCE_MAX_WORKERS`, default 3); if one fails, its error is captured and the rest continue rather than the whole request failing.
3. **Synthesize** -- once every worker has a result, one final call combines them into a single coherent answer (this step respects Deep think if it's also on).

The reply includes a `breakdown` (list of `{subtask, result}`) whenever real decomposition happened, rendered as a collapsed "Show task breakdown" toggle under the reply, so you can see what each worker actually did.

**Honest caveats:** this is several sequential-or-parallel LLM calls per request (1 plan + N workers + 1 synthesis), so it's slower than a normal reply -- that's why it's opt-in, same as Deep think. "Parallel" is best-effort: workers are dispatched concurrently from Python's side, but Foundry Local serves one model instance, so actual wall-clock speedup depends on whether it can service concurrent requests or just queues them -- either way the result is correct, just not necessarily faster than sequential on a single-GPU/CPU box. And a small local model's plans and worker outputs are meaningfully weaker than what you'd get from a frontier-model-backed Workforce -- expect to iterate on subtask quality, same as with tool calling.

### Cross-Session Memory (opt-in)

The "Conversation Memory" and "Document Upload" features above are both scoped to one `session_id` -- ask Iris something in a new session (new browser/localStorage) and it has no idea what was discussed or uploaded before. `utils/graph_memory.py` adds a second, independent memory layer that persists across every session: it's what lets you upload a document or have a conversation today, and ask about it in a completely different session next week.

**What it is and why this package specifically:** the user's ask here was for the same kind of "graphical retention of data for later auditing" that [MiroFish](https://github.com/666ghj/MiroFish) uses -- MiroFish's memory is Zep Cloud, a hosted product. [Graphiti](https://github.com/getzep/graphiti) (`graphiti-core`) is the open-source, Apache-2.0-licensed engine Zep itself is built on, and it works standalone against any OpenAI-compatible endpoint -- including Foundry Local -- so it was used directly instead of pulling in MiroFish (which is a whole standalone multi-agent simulation app built for a different problem, and AGPL-3.0 licensed) just to get at one dependency.

Rather than embeddings-and-cosine-similarity (what the ChromaDB memory above does), Graphiti extracts **entities and relationships** from text via LLM calls and stores them as a temporal knowledge graph -- each fact is a timestamped edge (e.g. "Alice WORKS_AT Acme Corp", learned at time T), so it's not just "what was said" but a point-in-time-queryable record of what was true and when, which is the "auditing" angle: you can ask what was known as of a certain time, and facts that get contradicted later are marked invalid rather than silently overwritten.

**Setup:**
```bash
pip install "graphiti-core[falkordblite]"   # requires Python 3.12+
export ENABLE_GRAPH_MEMORY=1
python app.py
```
That's it -- no separate database server to run. The `[falkordblite]` extra pulls in `redislite`, which runs an embedded FalkorDB (a Redis-module-based graph database) as a file (`graph_memory.db` by default, gitignored) rather than a process you manage yourself, consistent with this project's "no extra services" design. The LLM/embedding calls Graphiti needs for entity extraction are routed through the same Foundry Local endpoint as everything else (`utils/foundry_client.get_endpoint_config`) -- no separate API key, no cloud dependency. Graphiti's own telemetry (PostHog analytics) is disabled unconditionally (`GRAPHITI_TELEMETRY_ENABLED=false`, set at import time in `graph_memory.py`), matching this project's "everything stays local" rule for every other dependency.

**How it's wired into `app.py`:** every `/chat` turn fires an `add_episode` call in the background with that turn's user message + reply, and every `/upload` fires one with the uploaded document's content -- both fire-and-forget (via `asyncio.run_coroutine_threadsafe` on a background event loop, mirroring `utils/mcp_manager.py`'s architecture), so extraction latency never delays a reply. On the read side, `/chat` also calls `graph_memory.search(user_message)` and, if it returns anything, adds it to the prompt as a "Facts remembered from earlier conversations" system message alongside (not replacing) the session-scoped document context.

**Zero cost when unused.** `start()` checks `ENABLE_GRAPH_MEMORY` before importing `graphiti_core`/`redislite` at all -- with the env var unset (the default), no background thread starts, `search()` returns `""` immediately, and `add_episode()` is a no-op; nothing here changes behavior or requires the package to even be installed.

**Honest caveats:**
- Entity/relationship extraction is several LLM calls per episode -- noticeably heavier than the plain ChromaDB save, which is why every write is fire-and-forget and every read is timeout-bounded (`search(..., timeout=5)`, defaulting to no context rather than blocking a reply).
- This was verified end-to-end in this project's dev sandbox using the embedded FalkorDB Lite backend and a scripted local server standing in for Foundry Local's chat/embeddings API (Foundry Local itself doesn't run in that Linux sandbox -- an existing limitation of this whole codebase, not something new here): episode extraction, temporal fact storage, and semantic search retrieval all confirmed working. It has **not** been run against a real Foundry Local instance, since that requires Windows/macOS.
- Graphiti's own stated primary/default backend is Neo4j, not FalkorDB -- that path was **not** tested here at all (no Docker daemon and no reachable Neo4j download server in the dev sandbox this was built in). If you'd rather run real Neo4j, swap the driver construction in `graph_memory._build_graphiti()` for `graphiti_core.driver.neo4j_driver.Neo4jDriver` per [Graphiti's own docs](https://github.com/getzep/graphiti) -- untested here, but it's the officially supported path upstream.

### Voice Mode: Hands-Free Conversation (opt-in)

Click **Voice mode** to talk to Iris hands-free, the way you would with Siri or a similar voice assistant: it listens, transcribes what you said, replies, speaks the reply out loud, and then automatically starts listening again -- no clicking Record for every turn.

**Deliberately not built on the browser's `SpeechRecognition` API.** In Chrome (and most Chromium browsers), that API sends your raw microphone audio to Google's servers to transcribe -- which would quietly break this whole project's offline/local-first design the moment you turned Voice mode on. Instead, Voice mode is built entirely from pieces this app already has: `MediaRecorder` captures audio locally, it's sent to `/upload_audio` and transcribed by the same local Whisper model used for the existing manual Record button, the transcript goes to `/chat` exactly like a typed message, and the reply comes back through the same local TTS pipeline as every other response.

**How turn-taking works without a push-to-talk button:** the only new piece is knowing when you've stopped talking. `templates/chat.html` runs the microphone's audio through the Web Audio API's `AnalyserNode` and computes a rolling volume (RMS); once volume crosses a speaking threshold, a silence timer starts, and if volume stays low for `VOICE_SILENCE_MS` (1.2s), recording stops and that clip is sent off. A hard cap (`VOICE_MAX_RECORD_MS`, 15s) guards against a stuck-open mic recording forever. When the reply's audio finishes playing (`speakResponse`'s `onEnded` callback), Voice mode automatically starts listening again -- that's the whole hands-free loop. Turning Voice mode off releases the microphone (`getUserMedia` tracks are stopped) rather than just muting it.

**Honest caveats:** this was written and syntax-checked, but the dev environment behind this repo has no display or microphone to drive a real browser through an actual spoken conversation, so the full loop (real speech in, real playback out, repeated) has **not** been manually verified end to end here -- please try it locally and adjust `VOICE_SPEECH_RMS` in `chat.html` if it cuts you off mid-sentence or never detects silence (background noise level varies a lot by microphone/room). Browser autoplay policies can also block audio playback that isn't triggered by a direct user gesture in some strict configurations; toggling Voice mode on is itself a user gesture, but this hasn't been checked against every browser.

### Troubleshooting

1. Foundry Local Connection Issues
- Error: "[Foundry connection error: ...]" in the chat response
  - Verify Foundry Local is installed and on your `PATH`
  - Verify the model alias in `FOUNDRY_MODEL` / `FOUNDRY_EMBEDDING_MODEL` exists in your catalog: `foundry model list`
  - The first request for a given model can be slow while it downloads -- check server logs for download progress

2. ChromaDB Issues
- Error: "Collection not found"
  - Check persistence directory permissions
  - Verify ChromaDB path in configuration
  - Clear and reinitialize the database if corrupted

3. Voice Recognition Issues
- Error: "No audio device found"
  - Verify microphone permissions
  - Check audio device settings
  - Ensure Whisper model is properly installed

## Data Flow
The system processes user inputs through multiple stages, from text/voice input to AI response generation, maintaining context through vector embeddings.

```ascii
User Input (Text/Voice) --> Speech Recognition (if voice)
       |
       v
[Conversation History (this session)] + [Document Context (semantic search)]
       |                                          ^
       v                                          |
[Foundry Local Language Model] <----------- [ChromaDB Store]
       |                     \
       |                      \--(tool_calls)--> [MCP Servers, e.g. Power BI] --+
       |                                                                        |
       v <----------------------------------------------------------------------+
[<thinking> / <answer> split]  (Deep think only)
       |
       v
Text-to-Speech Output (local WAV, played by the browser)
```

Key Component Interactions:
1. User input is processed through text or voice channels
2. Speech input is transcribed using Whisper
3. The session's actual chat history is replayed to the model verbatim (not semantic search -- see Conversation Memory above); documents uploaded in that session are searched semantically for relevant chunks
4. If MCP servers are configured, their tools are offered to the model, which may call one or more before producing a final answer (see Tool Calling / MCP above)
5. If Workforce is on and the request is complex enough, steps 3-4 happen per-subtask across a small team of workers instead of once, then a synthesis call combines their results (see Workforce above); simple requests skip this and fall through to the normal single-reply path even with the toggle on
6. With Deep think on, the model (or the workforce's synthesis step) reasons step by step before answering (see Chain-of-Thought Reasoning above); off by default for responsiveness
7. The final answer (reasoning stripped) is stored in ChromaDB for future turns
8. The final answer is synthesized to a local audio file and played by the browser; the reasoning and/or task breakdown, if any, are shown separately as optional, collapsed toggles
9. Web search integration provides additional information (optional)
