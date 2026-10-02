#!/usr/bin/env bash
# One-time setup for Ubuntu / Debian. Run from anywhere:  ./setup_linux.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing system packages (needs sudo)"
# ffmpeg: audio decoding for Whisper, espeak-ng + alsa-utils: offline TTS used by pyttsx3
sudo apt-get update
sudo apt-get install -y python3 python3-venv ffmpeg espeak-ng alsa-utils curl

echo "==> Creating virtual environment in ./venv"
python3 -m venv venv
# shellcheck disable=SC1091
. venv/bin/activate
pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> NVIDIA GPU detected: installing PyTorch with CUDA support"
    pip install torch
else
    # The default Linux wheel bundles several GB of CUDA libraries that are useless without a GPU
    echo "==> No NVIDIA GPU detected: installing the much smaller CPU-only PyTorch"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt

if ! command -v ollama >/dev/null 2>&1; then
    echo
    echo "Ollama is not installed. Install it with:"
    echo "    curl -fsSL https://ollama.com/install.sh | sh"
    echo "then run this script again to download the models."
    exit 0
fi

if ! ollama list >/dev/null 2>&1; then
    echo
    echo "Ollama is installed but not running. Start it with 'sudo systemctl start ollama'"
    echo "(or 'ollama serve' in another terminal), then run this script again."
    exit 0
fi

echo "==> Downloading Ollama models"
ollama pull "${OLLAMA_MODEL:-mistral}"
ollama pull nomic-embed-text
ollama pull "${AGENT_MODEL:-qwen2.5:7b}"   # tool-calling model for agent.py

echo
echo "Setup complete. Start the assistant with:"
echo "    source venv/bin/activate && python app.py"
echo "then open http://localhost:8000"
echo "or the terminal agent with:"
echo "    source venv/bin/activate && python agent.py ~/path/to/finance-files"
