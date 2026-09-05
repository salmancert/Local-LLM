"""Cross-session memory via Graphiti (getzep/graphiti), a temporal
knowledge-graph library -- the open-source engine behind Zep Cloud's
memory feature (MiroFish's own "graphical retention... for later
auditing" is Zep; this is the library it's built on).

Why this exists alongside the ChromaDB memory in app.py: `save_to_memory`/
`retrieve_document_context` there are scoped to a single `session_id` --
open a new session and none of it is visible. This module instead extracts
entities and facts from every conversation turn and every uploaded
document into a graph, as timestamped edges, searchable from ANY later
session ("what did I tell you about the Q3 budget last week?" answered
from a chat two weeks ago in a different browser tab). The "auditing"
angle the user asked about is this graph's temporal edges: each fact
records when it was learned (and, if it's later contradicted, when it
stopped being true), so it's a point-in-time-queryable record, not just a
flat fact store.

Backend: FalkorDB Lite (redislite's embedded FalkorDB build) -- not a
separate Neo4j/FalkorDB server process, consistent with this app's
"no extra services" local-first design. The graph lives in a single file
on disk (see GRAPH_DB_PATH), no Docker or external DB to run. Graphiti's
own default/primary backend is Neo4j; that was NOT tested in this
codebase (no Docker daemon and no way to reach Neo4j's own download
servers in this project's dev sandbox) -- documented here as the
upstream default, not as something verified here. If you'd rather run
real Neo4j, swap `_build_graphiti`'s driver construction for
`graphiti_core.driver.neo4j_driver.Neo4jDriver` per Graphiti's own docs.

Opt-in: set ENABLE_GRAPH_MEMORY=1. Off by default, and `graphiti-core`
need not even be installed unless you turn this on -- every import of it
happens inside functions, not at module load time (same "zero cost when
unused" rule the rest of this app follows for kokoro-onnx, MCP, etc).
Requires Python 3.12+ (the `falkordblite` extra does); on 3.11 this
module can still be imported, it just stays disabled -- see requirements.txt.

Extraction runs several LLM calls per episode (Graphiti calls out to the
model to pull out entities/relationships), so this is meaningfully slower
than the plain ChromaDB memory save. Every write here happens in a
background thread, fire-and-forget -- like utils/mcp_manager.py's
connection setup, it never adds latency to a chat reply. Reads (search)
are synchronous from the caller's point of view but bounded by a timeout,
so a slow/stuck graph never delays a reply either -- it just contributes
no context for that one turn.
"""
import asyncio
import datetime
import os
import threading

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

GRAPH_DB_PATH = os.environ.get("GRAPH_MEMORY_DB_PATH", "graph_memory.db")

_loop = None
_ready = threading.Event()
_graphiti = None
_enabled = False


def _build_graphiti():
    from redislite import AsyncFalkorDB
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    from utils.foundry_client import get_endpoint_config

    config = get_endpoint_config()
    llm_config = LLMConfig(
        api_key=config["api_key"],
        model=config["chat_model_id"],
        small_model=config["chat_model_id"],
        base_url=config["base_url"],
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=config["api_key"],
        embedding_model=config["embedding_model_id"],
        base_url=config["base_url"],
    )

    falkor_db = AsyncFalkorDB(dbfilename=GRAPH_DB_PATH)
    driver = FalkorDriver(falkor_db=falkor_db)

    return Graphiti(
        graph_driver=driver,
        llm_client=OpenAIGenericClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
        # Graphiti's own default cross-encoder needs OPENAI_API_KEY set even
        # with a custom base_url -- pass one explicitly, pointed at the same
        # local endpoint, so this never depends on an OpenAI account existing.
        cross_encoder=OpenAIRerankerClient(config=llm_config),
    )


def _run_loop():
    global _loop, _graphiti
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def setup():
        global _graphiti
        graphiti = _build_graphiti()
        await graphiti.build_indices_and_constraints()
        _graphiti = graphiti
        _ready.set()
        await asyncio.Event().wait()  # park forever; the graph stays open

    try:
        _loop.run_until_complete(setup())
    except Exception as e:
        print(f"[graph_memory: startup failed, disabling: {e}]")
        _graphiti = None
        _ready.set()


def start():
    """Connect/initialize the graph in the background. Opt-in via
    ENABLE_GRAPH_MEMORY=1 -- a no-op with zero cost (no imports, no
    thread) otherwise, same pattern as utils/mcp_manager.py."""
    global _enabled
    _enabled = os.environ.get("ENABLE_GRAPH_MEMORY", "").strip().lower() in ("1", "true", "yes")
    if not _enabled:
        _ready.set()
        return
    threading.Thread(target=_run_loop, daemon=True).start()


def add_episode(name, text, source_description="chat"):
    """Fire-and-forget: extract entities/facts from `text` and store them
    in the graph for retrieval in any future session. Safe to call even
    when disabled or still starting up -- silently does nothing then."""
    if not _enabled or _graphiti is None or _loop is None:
        return

    async def _add():
        try:
            await _graphiti.add_episode(
                name=name,
                episode_body=text,
                source_description=source_description,
                reference_time=datetime.datetime.now(datetime.timezone.utc),
            )
        except Exception as e:
            print(f"[graph_memory: add_episode failed: {e}]")

    asyncio.run_coroutine_threadsafe(_add(), _loop)


def search(query, num_results=5, timeout=5):
    """Matching facts from earlier sessions, as one newline-joined string
    (empty if disabled, not ready yet, no matches, or the search doesn't
    finish within `timeout` seconds) -- callers can drop the result
    straight into a prompt with no special-casing needed."""
    if not _enabled or not _ready.is_set() or _graphiti is None or _loop is None:
        return ""

    future = asyncio.run_coroutine_threadsafe(
        _graphiti.search(query, num_results=num_results), _loop
    )
    try:
        results = future.result(timeout=timeout)
    except Exception as e:
        print(f"[graph_memory: search failed/timed out: {e}]")
        return ""

    facts = [edge.fact for edge in results if getattr(edge, "fact", None)]
    return "\n".join(facts)
