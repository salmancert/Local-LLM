#!/usr/bin/env bash
# One-time setup for Ubuntu / Debian. Run from anywhere:  ./setup_linux.sh
#
# Installs the terminal tools (agent.py and qcr.py) and adds two commands to ~/.local/bin:
#   fin-agent   interactive terminal agent in the current folder (like Claude Code)
#   qcr         quality control review of an audit file
# Add --with-web-app to also install the browser chat app with voice (app.py).
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

WEB_APP=0
for arg in "$@"; do
    case "$arg" in
        --with-web-app) WEB_APP=1 ;;
        *) echo "Unknown option: $arg (use --with-web-app or nothing)"; exit 1 ;;
    esac
done

echo "==> Installing system packages (needs sudo)"
# tesseract-ocr: reads scanned PDFs, libreoffice-*-nogui: converts old .doc and .xls files
PACKAGES=(python3 python3-venv curl tesseract-ocr libreoffice-writer-nogui libreoffice-calc-nogui)
if [ "$WEB_APP" = 1 ]; then
    # ffmpeg: audio decoding for Whisper, espeak-ng + alsa-utils: offline TTS used by pyttsx3
    PACKAGES+=(ffmpeg espeak-ng alsa-utils)
fi
sudo apt-get update
sudo apt-get install -y "${PACKAGES[@]}"

echo "==> Creating virtual environment in ./venv"
python3 -m venv venv
# shellcheck disable=SC1091
. venv/bin/activate
pip install --upgrade pip

if [ "$WEB_APP" = 1 ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "==> NVIDIA GPU detected: installing PyTorch with CUDA support"
        pip install torch
    else
        # The default Linux wheel bundles several GB of CUDA libraries that are useless without a GPU
        echo "==> No NVIDIA GPU detected: installing the much smaller CPU-only PyTorch"
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    pip install -r requirements.txt
else
    pip install -r requirements-agent.txt
fi

echo "==> Adding the fin-agent and qcr commands to ~/.local/bin"
mkdir -p "$HOME/.local/bin"
for pair in "fin-agent:agent.py" "qcr:qcr.py"; do
    name="${pair%%:*}"
    script="${pair#*:}"
    cat > "$HOME/.local/bin/$name" <<EOF
#!/usr/bin/env bash
exec "$REPO/venv/bin/python" "$REPO/$script" "\$@"
EOF
    chmod +x "$HOME/.local/bin/$name"
done
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "Note: ~/.local/bin is not on your PATH yet. Log out and back in (or open a new terminal)." ;;
esac

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
ollama pull "${AGENT_MODEL:-qwen2.5:7b}"   # tool-calling model for fin-agent and qcr
if [ "$WEB_APP" = 1 ]; then
    ollama pull "${OLLAMA_MODEL:-mistral}"
    ollama pull nomic-embed-text
fi

echo
echo "Setup complete. Go to a folder with your files and run:"
echo "    fin-agent                              # chat with tool calling, in the current folder"
echo "    qcr checklist.xlsx ~/Clients/ClientName   # fill your QCR checklist for one client"
echo "    qcr checklist.xlsx ~/Clients --all        # one completed checklist per client"
if [ "$WEB_APP" = 1 ]; then
    echo "Web app: source venv/bin/activate && python app.py, then open http://localhost:8000"
fi
