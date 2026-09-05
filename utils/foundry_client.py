import os
from functools import lru_cache
from types import SimpleNamespace

# Foundry Local does not listen on a fixed, well-known port -- the actual
# service is started on demand and its port is chosen at runtime. The
# previous implementation hardcoded "http://localhost:8000/...", which
# both pointed nowhere (Foundry Local was never listening there) and
# collided with this app's own Flask port (also 8000). Instead we use the
# `foundry-local-sdk` package (module `foundry_local`) to start/attach to
# the local Foundry service and discover its real endpoint at runtime via
# `FoundryLocalManager(alias).endpoint`.
#
# NOTE: foundry-local-sdk is pinned to 0.5.1 in requirements.txt. Versions
# >=1.0.0 replaced this thin OpenAI-compatible REST client with an
# unrelated in-process native binding API (no `.endpoint`/`.api_key`), so
# an unpinned install would silently break this module.

# phi-4-mini (3.8B) over the previous default qwen2.5-1.5b (1.5B): a
# generation newer, meaningfully stronger per Microsoft's own benchmarks
# (reasoning/math/code), and specifically built with native function/tool
# calling as a first-class capability -- which matters a lot here now that
# tool calling, MCP, and Workforce all depend on the model reliably
# deciding when and how to call a tool. It's larger, so it is not faster
# in raw tokens/sec on identical hardware than 1.5B -- "faster" here means
# staying in the small/CPU-practical tier of its generation (Microsoft's
# own "lightweight footprint" positioning) rather than jumping to a 7B+
# model, and Foundry Local still auto-selects the fastest available
# hardware variant (CPU/GPU/NPU) for whichever alias is requested.
# Override with FOUNDRY_MODEL if you'd rather trade capability for raw
# speed (qwen2.5-0.5b/1.5b) or capability for size (qwen2.5-7b, phi-4).
DEFAULT_CHAT_MODEL = os.environ.get("FOUNDRY_MODEL", "phi-4-mini")
# "nomic-embed-text" (the previous default) is an Ollama model name and was
# never in Foundry Local's catalog -- every embedding call failed. The
# correct catalog alias, per Microsoft's own docs, is "qwen3-embedding-0.6b".
DEFAULT_EMBEDDING_MODEL = os.environ.get("FOUNDRY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")


@lru_cache(maxsize=None)
def _get_manager(alias):
    from foundry_local import FoundryLocalManager
    # Starts the Foundry Local service (if needed) and downloads/loads
    # `alias` -- may take a while on first use.
    return FoundryLocalManager(alias)


@lru_cache(maxsize=None)
def _get_client_and_model_id(alias):
    import openai

    # Explicit override for advanced setups (e.g. a remote Foundry Local
    # instance, or a manually chosen port).
    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    if endpoint:
        api_key = os.environ.get("FOUNDRY_API_KEY", "") or "not-needed"
        return openai.OpenAI(base_url=endpoint, api_key=api_key), alias

    manager = _get_manager(alias)
    client = openai.OpenAI(
        base_url=manager.endpoint,
        api_key=manager.api_key or "not-needed",
    )
    # The catalog alias (e.g. "qwen2.5-1.5b") isn't itself a valid model id
    # for the inference API -- resolve it to the concrete loaded model id.
    model_id = manager.get_model_info(alias).id
    return client, model_id


def query_foundry(messages, model=None, max_tokens=None, tools=None, tool_choice=None):
    """messages: a list of {"role": "system"|"user"|"assistant"|"tool", ...},
    e.g. the running conversation so far plus the new user turn -- callers
    are responsible for including whatever history the model should see.

    tools: an optional OpenAI-format `tools` list (see utils/mcp_manager.py);
    omitted/empty means no tool calling is offered to the model at all.

    Returns the raw response message (an object with `.content` and
    `.tool_calls`, matching the OpenAI SDK), not just the text, so callers
    can detect and act on tool calls. On a connection error, returns a
    stand-in object with `.content` set to an error string and
    `.tool_calls` set to None, so callers can handle both cases uniformly."""
    alias = model or DEFAULT_CHAT_MODEL
    try:
        client, model_id = _get_client_and_model_id(alias)
        kwargs = {"model": model_id, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message
    except Exception as e:
        return SimpleNamespace(content=f"[Foundry connection error: {e}]", tool_calls=None)


def foundry_embed(text, model=None):
    alias = model or DEFAULT_EMBEDDING_MODEL
    client, model_id = _get_client_and_model_id(alias)
    response = client.embeddings.create(model=model_id, input=text)
    return response.data[0].embedding


def foundry_embed_batch(texts, model=None):
    """Embed many texts in one HTTP round trip instead of one per text --
    the OpenAI embeddings API (which Foundry Local implements) accepts a
    list for `input` directly. Callers with more than a couple of texts to
    embed (e.g. a chunked document) should always use this instead of
    calling foundry_embed() in a loop: doing it one at a time doesn't just
    add per-call HTTP/dispatch overhead N times over, it's the dominant
    cost -- measured at ~14x slower for 150 short chunks against a local
    server with only 20ms of fixed per-call overhead, which is a
    conservative floor for a real local model server.

    Returns embeddings in the same order as `texts` (explicitly sorted by
    each response item's `.index`, since providers aren't required to
    return results in request order)."""
    if not texts:
        return []
    alias = model or DEFAULT_EMBEDDING_MODEL
    client, model_id = _get_client_and_model_id(alias)
    response = client.embeddings.create(model=model_id, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def get_endpoint_config(chat_model=None, embedding_model=None):
    """Resolve Foundry Local's base_url/api_key and concrete model ids
    without building an `openai.OpenAI` client -- for callers that need to
    hand these to a different SDK wrapper pointed at the same local
    endpoint (e.g. utils/graph_memory.py's Graphiti client, which has its
    own OpenAI-compatible client classes)."""
    chat_alias = chat_model or DEFAULT_CHAT_MODEL
    embed_alias = embedding_model or DEFAULT_EMBEDDING_MODEL

    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    if endpoint:
        api_key = os.environ.get("FOUNDRY_API_KEY", "") or "not-needed"
        return {
            "base_url": endpoint,
            "api_key": api_key,
            "chat_model_id": chat_alias,
            "embedding_model_id": embed_alias,
        }

    chat_manager = _get_manager(chat_alias)
    embed_manager = _get_manager(embed_alias)
    return {
        "base_url": chat_manager.endpoint,
        "api_key": chat_manager.api_key or "not-needed",
        "chat_model_id": chat_manager.get_model_info(chat_alias).id,
        "embedding_model_id": embed_manager.get_model_info(embed_alias).id,
    }
