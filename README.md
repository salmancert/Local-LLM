# AI-Powered Conversational Assistant with Document Memory and Voice Interaction

This project implements a web-based conversational AI assistant that combines document understanding, voice interaction, and contextual memory. It provides a natural language interface for document querying, web search, and general conversation with both text and voice support.

The assistant leverages local language models through Ollama, maintains conversation context using ChromaDB for semantic search, and supports voice interaction through offline text-to-speech and speech recognition capabilities. The system is designed to work primarily offline, making it suitable for environments with limited internet connectivity while still providing optional web search functionality.

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
    ├── doc_parser.py      # Document parsing functionality
    ├── embedding_store.py # ChromaDB interaction for storing embeddings
    ├── ollama_client.py  # Client for local Ollama API interaction
    └── web_search.py     # Optional web search functionality
```

## Usage Instructions
### Prerequisites
- Python 3.8 or higher
- Ollama installed locally
- ChromaDB
- Flask
- Whisper (for speech recognition)
- pyttsx3 (for text-to-speech)
- sentence-transformers
- PyMuPDF (for document parsing)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install required packages
pip install flask chromadb whisper pyttsx3 sentence-transformers PyMuPDF requests waitress

# Install Ollama (if not already installed)
# Follow instructions at: https://ollama.ai/download
```

### Quick Start
1. Start the Ollama service:
```bash
ollama serve
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:8000
```

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

### Troubleshooting

1. Ollama Connection Issues
- Error: "Connection refused to localhost:11434"
  - Verify Ollama is running: `ps aux | grep ollama`
  - Check Ollama logs: `ollama logs`
  - Restart Ollama: `ollama restart`

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