# AI-Powered Conversational Assistant with Document Memory and Voice Interaction

This project implements a web-based conversational AI assistant that combines document understanding, voice interaction, and contextual memory. It provides a natural language interface for document querying, web search, and general conversation with both text and voice support.

The assistant leverages local language models through Ollama, maintains conversation context using ChromaDB for semantic search, and supports voice interaction through offline text-to-speech and speech recognition capabilities. The system is designed to work primarily offline, making it suitable for environments with limited internet connectivity while still providing optional web search functionality.

## Repository Structure
```
.
├── app.py                 # Main Flask application with routing and core logic
├── agent.py               # Terminal agent with tool calling for finance files
├── offline script.py      # Utility for offline document processing and embedding
├── setup_linux.sh         # One-time setup script for Ubuntu/Debian
├── requirements.txt       # Python dependencies
├── static/               # Static web assets
│   └── style.css        # CSS styling for the chat interface
├── templates/           # HTML templates
│   └── chat.html       # Main chat interface template
└── utils/              # Utility modules
    ├── doc_parser.py      # Document parsing functionality
    ├── embedding_store.py # ChromaDB interaction for storing embeddings
    ├── ollama_client.py  # Client for local Ollama API interaction
    └── web_search.py     # Optional web search functionality
```

## Usage Instructions
The assistant runs on both **Windows** and **Linux** (Ubuntu/Debian).

### Prerequisites
- Python 3.9 or higher
- [Ollama](https://ollama.com/download) with the `mistral` and `nomic-embed-text` models
- `ffmpeg` (Whisper uses it to decode recorded audio)
- A text-to-speech engine for server-side speech: built in on Windows (SAPI5), `espeak-ng` on Linux
- Python packages from `requirements.txt`: Flask, waitress, ChromaDB, openai-whisper, pyttsx3, PyMuPDF, requests

### Memory requirements (16 GB RAM is enough)
| Component | Approx. RAM |
|-----------|-------------|
| `mistral` 7B (4-bit, via Ollama) | ~5 GB |
| `nomic-embed-text` | ~0.3 GB |
| Whisper `base` + PyTorch | ~1 GB |
| Flask, ChromaDB, browser | ~1-2 GB |

That leaves plenty of headroom on a 16 GB machine. Avoid the Whisper `medium`/`large` models
(5-10 GB) alongside a 7B LLM. Without a GPU, a 7B model can take tens of seconds per reply;
for faster answers use a smaller model such as `llama3.2:3b` (see [Configuration](#configuration)).

### Installation on Linux (Ubuntu/Debian)

```bash
# Clone the repository
git clone https://github.com/salmancert/Local-LLM.git
cd Local-LLM

# Install Ollama (runs as a background systemd service once installed)
curl -fsSL https://ollama.com/install.sh | sh

# Install system packages, create ./venv, install Python packages and pull the models
./setup_linux.sh
```

<details>
<summary>Manual steps (what <code>setup_linux.sh</code> does)</summary>

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg espeak-ng alsa-utils

python3 -m venv venv
source venv/bin/activate

# Without an NVIDIA GPU, install the CPU-only PyTorch first. The default Linux build pulls
# several GB of CUDA libraries that are useless on a CPU-only machine.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

ollama pull mistral
ollama pull nomic-embed-text
```
</details>

> Ubuntu 23.04+ refuses `pip install` outside a virtual environment ("externally-managed-environment"),
> so always activate `venv` first. Also note that Ubuntu has `python3`, not `python`, until the venv is active.

### Installation on Windows

```powershell
git clone https://github.com/salmancert/Local-LLM.git
cd Local-LLM

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Install ffmpeg and Ollama (https://ollama.com/download), then pull the models
winget install ffmpeg
ollama pull mistral
ollama pull nomic-embed-text
```

### Quick Start
1. Make sure Ollama is running:
   - **Linux:** it starts automatically as a service (`systemctl status ollama`). Only run `ollama serve`
     if you did not use the install script; otherwise it fails with "address already in use".
   - **Windows:** start the Ollama app from the Start menu, or run `ollama serve`.

2. Run the Flask application (with the virtual environment activated):
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:8000
```

To add a document to memory without the web UI:
```bash
python "offline script.py" ~/Documents/report.pdf
```

### Configuration
Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Address of the Ollama server |
| `OLLAMA_MODEL` | `mistral` | Chat model (pull it first with `ollama pull <model>`) |
| `WHISPER_MODEL` | `base` | Speech recognition model: `tiny`, `base`, `small`, `medium`, `large` |
| `SERVER_TTS` | `1` | Set to `0` to stop the server from speaking replies (the browser can still read them aloud) |
| `AGENT_MODEL` | `qwen2.5:7b` | Tool-calling model used by `agent.py` |
| `AGENT_CTX` | `16384` | Context size in tokens for `agent.py` |

```bash
# Linux
OLLAMA_MODEL=llama3.2:3b SERVER_TTS=0 python app.py
```
```powershell
# Windows (PowerShell)
$env:OLLAMA_MODEL="llama3.2:3b"; python app.py
```

### More Detailed Examples

1. Document Upload and Query:
```bash
# Upload a document through the web interface
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload

# Query the document
curl -X POST -H "Content-Type: application/json" \
     -d '{"message": "What does the document say about X?"}' \
     http://localhost:8000/chat
```

2. Voice Interaction:
```bash
# Upload audio for transcription
curl -X POST -F "audio=@your_recording.wav" http://localhost:8000/upload_audio
```

### Troubleshooting

1. Ollama Connection Issues
- Error: "Connection refused to localhost:11434"
  - Linux: check the service with `systemctl status ollama`, view logs with `journalctl -u ollama -e`,
    restart with `sudo systemctl restart ollama`
  - Windows: make sure the Ollama app is running in the system tray, or run `ollama serve`
- Error: "model not found"
  - Pull the models: `ollama pull mistral` and `ollama pull nomic-embed-text`

2. ChromaDB Issues
- Error: "Collection not found"
  - Check permissions of the `chroma_store` directory
  - Clear and reinitialize the database if corrupted

3. Voice Recognition Issues
- Error: "No such file or directory: 'ffmpeg'"
  - Install ffmpeg (`sudo apt install ffmpeg` on Ubuntu, `winget install ffmpeg` on Windows)
- Error: "No audio device found" / microphone blocked
  - Verify the browser has microphone permission
  - Browsers only allow the microphone on `http://localhost` or HTTPS, so open the app on the
    same machine that runs it (not via its LAN IP address)

4. Text-to-Speech Issues (Linux)
- "offline TTS unavailable" at startup: install eSpeak NG with `sudo apt install espeak-ng alsa-utils`
- The browser does not read replies aloud: install `speech-dispatcher` (`sudo apt install speech-dispatcher`)
  and restart the browser
- Replies are spoken twice: the server and the browser both speak. Click the Mute button in the
  page, or start the app with `SERVER_TTS=0`

## Terminal Agent for Finance Files
`agent.py` is a Claude Code style assistant that runs in your terminal on a local Ollama model.
The model can call tools to look at your files and run calculations, so it answers from your actual
data instead of guessing. Nothing leaves your machine.

```bash
ollama pull qwen2.5:7b                      # a model that supports tool calling (~4.7 GB)
source venv/bin/activate                    # Windows: venv\Scripts\activate
python agent.py ~/Documents/finance         # the folder with your statements, invoices, budgets
```

Then ask things like:
- "Summarise my spending by category for January and flag duplicate charges"
- "Which invoices in invoices/ are due before the end of the month, and what is the total?"
- "Compare budget.xlsx with the actual spending in statements/ and write the result to reports/budget_vs_actual.csv"

**Tools the model can use**

| Tool | What it does | Asks first? |
|------|--------------|-------------|
| `list_files` | List files with size and date | no |
| `read_file` | Read text, CSV, PDF and Excel files as numbered lines | no |
| `inspect_table` | Columns, types, first rows and numeric summary of a CSV/Excel file | no |
| `search_files` | Regex search across files, including PDFs and spreadsheets | no |
| `run_python` | Run Python with pandas for calculations (totals, grouping, reconciliation) | yes |
| `run_shell` | Run a shell command | yes |
| `write_file` | Create or overwrite a file (shows a diff first) | yes |

File tools are limited to the folder you start the agent in. Before `run_python`, `run_shell` or
`write_file`, the agent shows the code or diff and waits for **y**es, **n**o, **a**lways (for that tool
this session), or your own instruction instead. Start with `--yes` to skip the questions.

**Commands:** `/help`, `/clear` (new conversation), `/model <name>`, `/auto` (toggle asking),
`/tools`, `/exit`. End a line with `\` to continue typing on the next line; Ctrl+C stops an answer.
Use `python agent.py -p "question"` for a single answer without the interactive prompt.

**Models:** any Ollama model with tool support works, set with `--model` or `AGENT_MODEL`.
`qwen2.5:7b` (default) is reliable at tool calling and fits in 16 GB RAM alongside its 16k-token
context (~6 GB in total). Alternatives: `llama3.1:8b`, `qwen3:8b`. Plain `mistral` is weaker at tool
calling. `AGENT_CTX` sets the context size (default 16384 tokens); larger uses more RAM.

## Data Flow
The system processes user inputs through multiple stages, from text/voice input to AI response generation, maintaining context through vector embeddings.

```ascii
User Input (Text/Voice) --> Speech Recognition (if voice)
       |
       v
[Context Retrieval] <--> [ChromaDB Store]
       |
       v
[Ollama Language Model]
       |
       v
[Response Generation]
       |
       v
Text-to-Speech Output
```

Key Component Interactions:
1. User input is processed through text or voice channels
2. Speech input is transcribed using Whisper
3. ChromaDB retrieves relevant context using semantic search
4. Ollama generates contextual responses
5. Responses are stored in ChromaDB for future context
6. Text-to-speech converts responses to audio when needed
7. Web search integration provides additional information (optional)
