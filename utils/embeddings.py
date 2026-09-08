"""How this app turns text into vectors, with a fallback that keeps memory
working when Foundry Local can't do embeddings.

Two backends:

  - "foundry" (preferred): Foundry Local's embedding model, via
    utils/foundry_client.py. Same local service everything else uses.
  - "local" (fallback): sentence-transformers running in-process. No
    Foundry Local involvement at all.

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
qwen3-embedding-0.6b is 1024-dim; the default fallback is 384-dim) and are
NOT interchangeable within one ChromaDB collection -- see the dimension
mismatch handling in app.py.
"""
import os
import threading

# all-MiniLM-L6-v2: 384 dimensions, ~80MB, CPU-fast, and the most widely
# used small sentence-transformers model there is. sentence-transformers is
# already a dependency of this project, so this adds no new package.
#
# One caveat worth being explicit about: the weights download from Hugging
# Face on first use and are cached afterwards. Everything after that first
# download is fully offline -- but that first run does need network access,
# unlike the rest of this app. Pre-cache it (or point HF_HOME at a cache
# you've copied in) if the machine is permanently offline.
FALLBACK_EMBEDDING_MODEL = os.environ.get("FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_BACKEND_OVERRIDE = os.environ.get("EMBEDDING_BACKEND", "auto").strip().lower()

_lock = threading.Lock()
_backend = None if _BACKEND_OVERRIDE in ("", "auto") else _BACKEND_OVERRIDE
_st_model = None


def active_backend():
    """"foundry", "local", or None if nothing has been embedded yet."""
    return _backend


def _set_backend(name):
    global _backend
    with _lock:
        _backend = name


def _get_sentence_transformer():
    global _st_model
    with _lock:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[embeddings: loading fallback model '{FALLBACK_EMBEDDING_MODEL}' "
                  f"(first run downloads it, then it's cached and offline)]")
            _st_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
        return _st_model


def _embed_batch_local(texts):
    model = _get_sentence_transformer()
    # encode() returns numpy arrays; ChromaDB wants plain lists of floats.
    return [vector.tolist() for vector in model.encode(list(texts))]


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
              f"falling back to in-process '{FALLBACK_EMBEDDING_MODEL}']")
        _set_backend("local")
        return _embed_batch_local(texts)


def embed(text):
    """Single-text convenience wrapper. Prefer embed_batch() for more than
    a couple of texts -- see foundry_embed_batch's docstring for why."""
    return embed_batch([text])[0]
