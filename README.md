# AI-Powered Conversational Assistant with Document Memory and Voice Interaction

This project implements a web-based conversational AI assistant that combines document understanding, voice interaction, and contextual memory. It provides a natural language interface for document querying, web search, and general conversation with both text and voice support.

The assistant leverages local language models through [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local), maintains conversation context using ChromaDB for semantic search, and supports voice interaction through offline text-to-speech and speech recognition capabilities. The system is designed to run entirely offline, with telemetry disabled, making it suitable for environments with limited internet connectivity while still providing optional web search functionality.

## Repository Structure
```
.
├── app.py                 # Main Flask application with routing and core logic
├── offline script.py      # Utility for offline document processing and embedding
├── static/               # Static web assets
│   └── style.css        # CSS styling for the chat interface
├── templates/           # HTML templates
│   └── chat.html       # Main chat interface template
└── utils/              # Utility modules
    ├── doc_parser.py       # Document parsing functionality
    ├── embedding_store.py  # Standalone ChromaDB + sentence-transformers helper
    ├── foundry_client.py   # Client for local Foundry Local inference/embeddings
    ├── tts.py               # Offline text-to-speech (pyttsx3 / chatterbox-tts)
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
| `FOUNDRY_MODEL` | `qwen2.5-1.5b` | Foundry Local catalog alias used for chat completions |
| `FOUNDRY_EMBEDDING_MODEL` | `nomic-embed-text` | Foundry Local catalog alias used for embeddings |
| `FOUNDRY_ENDPOINT` | *(auto-discovered)* | Overrides the endpoint instead of discovering it via the SDK (e.g. a remote Foundry Local instance) |
| `FOUNDRY_API_KEY` | *(none)* | API key to send when `FOUNDRY_ENDPOINT` is set |
| `TTS_ENGINE` | `pyttsx3` | `pyttsx3` (default, lightweight) or `chatterbox` (higher quality, heavier, optional dependency) |
| `TTS_DEVICE` | `cpu` | Device used when `TTS_ENGINE=chatterbox` (e.g. `cpu` or `cuda`) |

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

- **pyttsx3** (default) -- uses the OS's built-in voices, no model download, always available.
- **chatterbox-tts** (optional) -- a local neural TTS model for higher quality voices. Install it explicitly (`pip install chatterbox-tts`, a heavier dependency that pulls in PyTorch) and set `TTS_ENGINE=chatterbox`. If it isn't installed, or synthesis fails, the app automatically falls back to pyttsx3.

If TTS fails entirely, the frontend falls back to the browser's built-in `speechSynthesis` API.

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
[Context Retrieval] <--> [ChromaDB Store]
       |
       v
[Foundry Local Language Model]
       |
       v
[Response Generation]
       |
       v
Text-to-Speech Output (local WAV, played by the browser)
```

Key Component Interactions:
1. User input is processed through text or voice channels
2. Speech input is transcribed using Whisper
3. ChromaDB retrieves relevant context using semantic search (telemetry disabled)
4. Foundry Local generates contextual responses
5. Responses are stored in ChromaDB for future context
6. The response is synthesized to a local audio file and played by the browser
7. Web search integration provides additional information (optional)
