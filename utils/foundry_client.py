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

# Which chat model to use. This default has moved twice, both times for
# the same reason: tool-calling reliability is what this app lives or dies
# on. phi-4-mini (3.8B) would answer "I've highlighted the spelling errors"
# without ever emitting a tool call, so no file was produced; qwen2.5-7b
# fixed the worst of that, and qwen3-8b is a generation newer again at
# essentially the same size, with notably stronger agentic/tool-use
# behavior. Tools, MCP, Workforce and the PDF grammar pass all depend on
# the model actually deciding to call a tool.
#
# Note qwen3 reasons in <think>...</think> natively, whether or not Deep
# think is on -- app.py's split_thinking() strips those so raw reasoning
# doesn't end up shown, spoken, and written to memory.
#
# Stronger if you have the hardware: qwen3-14b, phi-4, qwen2.5-14b, or
# gpt-oss-20b (Microsoft's own pick for agentic tool-calling work, but it
# effectively needs a capable GPU). Faster/smaller: qwen3-4b, qwen3-1.7b,
# qwen2.5-1.5b.
DEFAULT_CHAT_MODEL = os.environ.get("FOUNDRY_MODEL", "qwen3-8b")

# Tried in order when the configured chat model isn't in this machine's
# catalog. Ordered newest-generation-and-still-locally-practical first,
# with small models at the end so the app still runs on modest hardware
# rather than failing outright. Aliases here were checked against a real
# Foundry Local catalog listing rather than guessed.
CHAT_MODEL_FALLBACKS = [
    "qwen3-8b",
    "qwen2.5-7b",
    "phi-4",
    "qwen3-4b",
    "mistral-7b-v0.2",
    "qwen3-14b",
    "qwen2.5-14b",
    "phi-4-mini",
    "phi-3.5-mini",
    "qwen3-1.7b",
    "qwen2.5-1.5b",
    "qwen2.5-0.5b",
]

# "nomic-embed-text" (the original default here) is an Ollama model name and
# was never in Foundry Local's catalog -- every embedding call failed.
# "qwen3-embedding-0.6b" is the alias in Microsoft's own embedding docs, but
# real-world testing found it missing from at least one user's catalog:
# embeddings are a relatively recent Foundry Local addition, so whether any
# embedding model is present depends on the installed version. Hence the
# fallback list, and -- when the catalog has no embedding model at all --
# utils/embeddings.py falling back to a small in-process model instead.
DEFAULT_EMBEDDING_MODEL = os.environ.get("FOUNDRY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")

EMBEDDING_MODEL_FALLBACKS = [
    "qwen3-embedding-0.6b",
    "all-minilm-l6-v2",
    "bge-small-en-v1.5",
]


@lru_cache(maxsize=None)
def _get_manager(alias):
    from foundry_local import FoundryLocalManager
    # Starts the Foundry Local service (if needed) and downloads/loads
    # `alias` -- may take a while on first use.
    return FoundryLocalManager(alias)


@lru_cache(maxsize=None)
def _get_catalog_manager():
    """A manager bound to no particular model, so the catalog can be read
    without downloading anything -- `alias_or_model_id` is optional in the
    SDK, and passing one triggers a (potentially multi-GB) download."""
    from foundry_local import FoundryLocalManager
    return FoundryLocalManager()


@lru_cache(maxsize=None)
def _catalog_aliases():
    """{alias: FoundryModelInfo} for everything in this machine's catalog.
    Empty dict if the catalog can't be read, so callers degrade to just
    trying the configured alias directly rather than failing here."""
    try:
        models = _get_catalog_manager().list_catalog_models()
    except Exception as e:
        print(f"[Foundry: couldn't read the model catalog: {e}]")
        return {}

    catalog = {}
    for info in models:
        # Several hardware variants can share one alias -- first wins, which
        # matches Foundry Local's own "alias picks the best variant" behavior.
        if info.alias and info.alias not in catalog:
            catalog[info.alias] = info
    return catalog


def _resolve_alias(preferred, fallbacks, kind, require_tool_calling=False):
    """The first of `preferred` + `fallbacks` actually present in the
    catalog. Returns None if none of them are (and, for embeddings, that's
    a normal outcome -- see utils/embeddings.py). Note get_model_info()
    returns None rather than raising for an unknown alias, which is why
    the old `get_model_info(alias).id` blew up with an opaque
    AttributeError when an alias wasn't in the catalog.

    `require_tool_calling` prefers models the catalog marks as supporting
    tool calling (FoundryModelInfo.supports_tool_calling). Falling back to
    a model that can't call tools would quietly disable this app's tools,
    MCP, Workforce and PDF proofreading, so it's worth skipping past one
    that can't -- but it's a preference, not a hard filter: a machine whose
    catalog has only non-tool models should still get a working chatbot."""
    catalog = _catalog_aliases()
    candidates = [preferred] + [a for a in fallbacks if a != preferred]

    if not catalog:
        return preferred  # catalog unreadable -- just try what was asked for

    def _announce(alias):
        if alias != preferred:
            print(f"[Foundry: '{preferred}' isn't usable on this machine, using '{alias}' instead]")
        return alias

    present = [alias for alias in candidates if alias in catalog]

    if require_tool_calling:
        for alias in present:
            if getattr(catalog[alias], "supports_tool_calling", False):
                return _announce(alias)
        if present:
            print(
                f"[Foundry: none of the preferred chat models support tool calling on this "
                f"machine; using '{present[0]}', so tools/MCP/Workforce may not work. "
                f"'foundry model list' shows which models have the 'tools' task.]"
            )
            return _announce(present[0])
    elif present:
        return _announce(present[0])

    available = ", ".join(sorted(catalog)) or "(none)"
    print(f"[Foundry: none of {candidates} are in the catalog for {kind}. Available: {available}]")
    return None


@lru_cache(maxsize=None)
def _get_client_and_model_id(alias):
    import openai

    # Explicit override for advanced setups (e.g. a remote Foundry Local
    # instance, or a manually chosen port). Skips catalog resolution
    # entirely -- the remote end decides what the alias means.
    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    if endpoint:
        api_key = os.environ.get("FOUNDRY_API_KEY", "") or "not-needed"
        return openai.OpenAI(base_url=endpoint, api_key=api_key), alias

    manager = _get_manager(alias)
    client = openai.OpenAI(
        base_url=manager.endpoint,
        api_key=manager.api_key or "not-needed",
    )
    # The catalog alias (e.g. "qwen2.5-7b") isn't itself a valid model id
    # for the inference API -- resolve it to the concrete loaded model id.
    info = manager.get_model_info(alias)
    if info is None:
        raise RuntimeError(
            f"Model '{alias}' is not in this machine's Foundry Local catalog. "
            f"Run 'foundry model list' to see what is available, then set "
            f"FOUNDRY_MODEL (or FOUNDRY_EMBEDDING_MODEL) accordingly."
        )
    return client, info.id


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
    alias = model or resolve_chat_model()
    if alias is None:
        return SimpleNamespace(
            content="[No chat model available in this machine's Foundry Local catalog -- "
                    "run 'foundry model list' and set FOUNDRY_MODEL to one of them.]",
            tool_calls=None,
        )
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


@lru_cache(maxsize=1)
def resolve_chat_model():
    """The chat model alias to actually use on this machine: the configured
    one if the catalog has it, otherwise the best available fallback."""
    if os.environ.get("FOUNDRY_ENDPOINT"):
        return DEFAULT_CHAT_MODEL  # remote endpoint -- no local catalog to check
    return _resolve_alias(DEFAULT_CHAT_MODEL, CHAT_MODEL_FALLBACKS, "chat", require_tool_calling=True)


@lru_cache(maxsize=1)
def resolve_embedding_model():
    """The embedding model alias to use, or None if this machine's catalog
    has no embedding model at all -- a normal outcome on Foundry Local
    versions predating embedding support. utils/embeddings.py treats None
    as "use the in-process fallback embedder instead"."""
    if os.environ.get("FOUNDRY_ENDPOINT"):
        return DEFAULT_EMBEDDING_MODEL
    alias = _resolve_alias(DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_FALLBACKS, "embeddings")
    if alias is not None:
        return alias

    # Nothing from the known-alias list is present. Before giving up, take
    # anything in the catalog whose task looks like an embedding task --
    # the catalog grows over time and may carry a model we've never heard of.
    for candidate, info in sorted(_catalog_aliases().items()):
        if "embed" in (getattr(info, "task", "") or "").lower():
            print(f"[Foundry: using catalog embedding model '{candidate}']")
            return candidate
    return None


def foundry_embed(text, model=None):
    alias = model or resolve_embedding_model()
    if alias is None:
        raise RuntimeError("no embedding model available in the Foundry Local catalog")
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
    alias = model or resolve_embedding_model()
    if alias is None:
        raise RuntimeError("no embedding model available in the Foundry Local catalog")
    client, model_id = _get_client_and_model_id(alias)
    response = client.embeddings.create(model=model_id, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def get_endpoint_config(chat_model=None, embedding_model=None):
    """Resolve Foundry Local's base_url/api_key and concrete model ids
    without building an `openai.OpenAI` client -- for callers that need to
    hand these to a different SDK wrapper pointed at the same local
    endpoint (e.g. utils/graph_memory.py's Graphiti client, which has its
    own OpenAI-compatible client classes)."""
    chat_alias = chat_model or resolve_chat_model()
    embed_alias = embedding_model or resolve_embedding_model()

    endpoint = os.environ.get("FOUNDRY_ENDPOINT")
    if endpoint:
        api_key = os.environ.get("FOUNDRY_API_KEY", "") or "not-needed"
        return {
            "base_url": endpoint,
            "api_key": api_key,
            "chat_model_id": chat_alias,
            "embedding_model_id": embed_alias,
        }

    if chat_alias is None or embed_alias is None:
        # Graphiti needs both an LLM and an embedder against one OpenAI-
        # compatible endpoint; it can't use utils/embeddings.py's in-process
        # fallback, so graph memory simply stays off rather than half-working.
        raise RuntimeError(
            "Foundry Local's catalog is missing a "
            f"{'chat' if chat_alias is None else 'embedding'} model, which graph "
            "memory requires. Run 'foundry model list' to see what's available."
        )

    chat_manager = _get_manager(chat_alias)
    embed_manager = _get_manager(embed_alias)
    return {
        "base_url": chat_manager.endpoint,
        "api_key": chat_manager.api_key or "not-needed",
        "chat_model_id": chat_manager.get_model_info(chat_alias).id,
        "embedding_model_id": embed_manager.get_model_info(embed_alias).id,
    }
