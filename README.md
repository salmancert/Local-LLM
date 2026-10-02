# AI-Powered Conversational Assistant with Document Memory and Voice Interaction

This project implements a web-based conversational AI assistant that combines document understanding, voice interaction, and contextual memory. It provides a natural language interface for document querying, web search, and general conversation with both text and voice support.

The assistant leverages local language models through Ollama, maintains conversation context using ChromaDB for semantic search, and supports voice interaction through offline text-to-speech and speech recognition capabilities. The system is designed to work primarily offline, making it suitable for environments with limited internet connectivity while still providing optional web search functionality.

## Repository Structure
```
.
├── app.py                 # Main Flask application with routing and core logic
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
