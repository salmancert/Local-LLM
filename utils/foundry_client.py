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

DEFAULT_CHAT_MODEL = os.environ.get("FOUNDRY_MODEL", "qwen2.5-1.5b")
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
