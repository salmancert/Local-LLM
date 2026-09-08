"""How this app turns text into vectors, with a fallback that keeps memory
working when Foundry Local can't do embeddings.

Two backends:

  - "foundry" (preferred): Foundry Local's embedding model, via
    utils/foundry_client.py. Same local service everything else uses.
  - "local" (fallback): all-MiniLM-L6-v2 run in-process with
    onnxruntime, against the model files bundled in models/minilm/. No
    Foundry Local, no network, no download -- ever.

The fallback exists because embeddings are a comparatively recent Foundry
Local feature, and real-world testing hit exactly this: the configured
embedding model wasn't in that machine's catalog, so every embedding call
failed, which silently disabled conversation memory and document search
(save_to_memory() catches the error and returns). Rather than leave the
app's memory quietly broken on such a setup, we fall back to a small
model that always works.

Backend choice is lazy and sticky: the first embedding call tries Foundry,
and if that fails for any reason the backend switches to "local"
permanently for the life of the process. Nothing is probed at startup, so
an app that never embeds anything pays nothing.

Set EMBEDDING_BACKEND=foundry or =local to skip the automatic choice.

Note the two backends produce different-sized vectors (Foundry's
qwen3-embedding-0.6b is 1024-dim; the bundled fallback is 384-dim) and are
NOT interchangeable within one ChromaDB collection -- see the dimension
mismatch handling in app.py.
"""
import os
import threading

# The fallback model files live in the repo (models/minilm/), committed the
# same way and for the same reason as the Kokoro TTS weights: so this app
# never needs network access to work. The original version of this module
# used sentence-transformers, which downloads from Hugging Face on first
# use -- that broke immediately for a user whose network blocks
# huggingface.co, which is exactly the environment this project targets.
#
# These are the files Chroma publishes for its own default embedder (an
# int8-quantized all-MiniLM-L6-v2 ONNX export, 384 dimensions, ~86MB --
# under GitHub's 100MB per-file limit, so no Git LFS needed). Running them
# needs only onnxruntime and tokenizers, both of which chromadb already
# depends on, so the fallback adds no new package either.
_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm"
)
FALLBACK_MODEL_PATH = os.environ.get("FALLBACK_EMBEDDING_MODEL_PATH", os.path.join(_MODEL_DIR, "model.onnx"))
FALLBACK_TOKENIZER_PATH = os.environ.get("FALLBACK_EMBEDDING_TOKENIZER_PATH", os.path.join(_MODEL_DIR, "tokenizer.json"))

# all-MiniLM-L6-v2's own training/eval length. sentence-transformers uses
# 256 for this model even though the HF config says 128.
_MAX_TOKENS = 256
_BATCH_SIZE = 32

_BACKEND_OVERRIDE = os.environ.get("EMBEDDING_BACKEND", "auto").strip().lower()

_lock = threading.Lock()
_backend = None if _BACKEND_OVERRIDE in ("", "auto") else _BACKEND_OVERRIDE
_session = None
_tokenizer = None


def active_backend():
    """"foundry", "local", or None if nothing has been embedded yet."""
    return _backend


def _set_backend(name):
    global _backend
    with _lock:
        _backend = name


def _get_onnx_session_and_tokenizer():
    global _session, _tokenizer
    with _lock:
        if _session is None:
            import onnxruntime
            from tokenizers import Tokenizer

            for path in (FALLBACK_MODEL_PATH, FALLBACK_TOKENIZER_PATH):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Bundled fallback embedding model missing: {path}. It ships in "
                        "models/minilm/ -- re-clone or restore that directory."
                    )

            print(f"[embeddings: loading bundled fallback model from {_MODEL_DIR}]")
            tokenizer = Tokenizer.from_file(FALLBACK_TOKENIZER_PATH)
            tokenizer.enable_truncation(max_length=_MAX_TOKENS)
            # Pad to the longest item in each batch rather than always to
            # _MAX_TOKENS: padded positions are masked out of both attention
            # and the mean pooling below, so the result is the same, but
            # short texts (most chat turns) don't pay for 256 positions.
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            _tokenizer = tokenizer
            _session = onnxruntime.InferenceSession(
                FALLBACK_MODEL_PATH,
                providers=onnxruntime.get_available_providers(),
            )
        return _session, _tokenizer


def _embed_batch_local(texts):
    """all-MiniLM-L6-v2 forward pass: masked mean pooling over the last
    hidden state, then L2 normalization -- the standard sentence-transformers
    recipe for this model, matching how these exact weights are meant to be
    used."""
    import numpy as np

    session, tokenizer = _get_onnx_session_and_tokenizer()

    vectors = []
    texts = list(texts)
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start:start + _BATCH_SIZE]
        encoded = tokenizer.encode_batch(batch)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        })

        last_hidden_state = outputs[0]
        mask = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
        pooled = np.sum(last_hidden_state * mask, axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)

        norms = np.linalg.norm(pooled, axis=1)
        norms[norms == 0] = 1e-12
        vectors.extend((pooled / norms[:, np.newaxis]).astype(np.float32).tolist())
    return vectors


def embed_batch(texts):
    """Embeddings for `texts`, in the same order. Raises only if BOTH
    backends fail -- a Foundry failure quietly switches to the fallback."""
    if not texts:
        return []

    if _backend == "local":
        return _embed_batch_local(texts)

    from utils.foundry_client import foundry_embed_batch
    try:
        embeddings = foundry_embed_batch(texts)
        if _backend is None:
            _set_backend("foundry")
        return embeddings
    except Exception as e:
        if _backend == "foundry":
            # Explicitly pinned to Foundry -- don't silently change behavior.
            raise
        print(f"[embeddings: Foundry Local embeddings unavailable ({e}); "
              f"falling back to the bundled in-process model in {_MODEL_DIR}]")
        _set_backend("local")
        return _embed_batch_local(texts)


def embed(text):
    """Single-text convenience wrapper. Prefer embed_batch() for more than
    a couple of texts -- see foundry_embed_batch's docstring for why."""
    return embed_batch([text])[0]
