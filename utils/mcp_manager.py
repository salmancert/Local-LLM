"""Generic MCP (Model Context Protocol) client manager.

Connects to whatever MCP servers are configured in mcp_servers.json (repo
root -- see mcp_servers.example.json), exposes their tools to Foundry Local
in OpenAI's `tools` format, and dispatches tool calls back to the right
server. This is intentionally generic: point it at a Power BI MCP server,
a filesystem one, or anything else that speaks MCP -- nothing here is
Power-BI-specific.

Design notes:
  - MCP servers are launched as local subprocesses (stdio transport) and
    kept connected for the life of the process, in a background thread
    running its own asyncio event loop (mirrors how Flask/waitress are
    synchronous but MCP's client is async).
  - Connecting happens in the background at startup and is never waited
    on by a request: get_openai_tools() returns [] until connections are
    ready rather than blocking a chat request, and returns [] forever if
    nothing is configured. So an app with no MCP servers configured pays
    zero cost, and a request that arrives before servers finish
    connecting just proceeds without tools rather than stalling.
"""
import asyncio
import json
import os
import threading

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_servers.json")

_loop = None
_ready = threading.Event()
_servers = {}          # server_name -> {"client": Client, "tools": [Tool, ...]}
_qualified_tools = {}  # "server__tool" -> (server_name, tool_name)


def _load_config():
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH) as f:
            data = json.load(f)
        return data.get("mcpServers", {})
    except Exception as e:
        print(f"[MCP: couldn't read {_CONFIG_PATH}: {e}]")
        return {}


async def _connect_all(exit_stack, server_configs):
    try:
        from mcp import Client, StdioServerParameters
    except ImportError as e:
        print(f"[MCP: 'mcp' package not available, tool calling disabled: {e}]")
        return

    for name, cfg in server_configs.items():
        try:
            params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env") or None,
            )
            client = Client(params)
            await exit_stack.enter_async_context(client)
            listing = await client.list_tools()

            _servers[name] = {"client": client, "tools": listing.tools}
            for tool in listing.tools:
                _qualified_tools[f"{name}__{tool.name}"] = (name, tool.name)

            print(f"[MCP: connected to '{name}' ({len(listing.tools)} tool(s))]")
        except Exception as e:
            print(f"[MCP: failed to connect to '{name}': {e}]")


def _run_loop(server_configs):
    global _loop
    from contextlib import AsyncExitStack

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def setup():
        # Held open for the process's lifetime -- this is what keeps the
        # MCP subprocesses and their stdio pipes alive between calls.
        exit_stack = AsyncExitStack()
        await _connect_all(exit_stack, server_configs)
        _ready.set()
        await asyncio.Event().wait()  # park forever; connections stay open

    try:
        _loop.run_until_complete(setup())
    except Exception as e:
        print(f"[MCP: background loop error: {e}]")
        _ready.set()


def start():
    """Connect to configured MCP servers in the background. Safe to call
    with no config present -- becomes an instant no-op with zero tools."""
    server_configs = _load_config()
    if not server_configs:
        _ready.set()
        return
    threading.Thread(target=_run_loop, args=(server_configs,), daemon=True).start()


def get_openai_tools():
    """OpenAI-format `tools` list aggregated across all connected MCP
    servers. Returns [] if nothing is configured, or if servers are still
    connecting -- callers should skip passing `tools` entirely when empty."""
    if not _ready.is_set():
        return []

    tools = []
    for qualified, (server_name, tool_name) in _qualified_tools.items():
        tool = next(t for t in _servers[server_name]["tools"] if t.name == tool_name)
        tools.append({
            "type": "function",
            "function": {
                "name": qualified,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        })
    return tools


def call_tool(qualified_name, arguments):
    """Execute a tool call by its qualified name (as returned by
    get_openai_tools). Returns a string suitable for a `tool` role message."""
    if qualified_name not in _qualified_tools:
        return f"Error: unknown tool '{qualified_name}'"

    server_name, tool_name = _qualified_tools[qualified_name]
    client = _servers[server_name]["client"]

    future = asyncio.run_coroutine_threadsafe(client.call_tool(tool_name, arguments), _loop)
    try:
        result = future.result(timeout=60)
    except Exception as e:
        return f"Error calling tool '{tool_name}': {e}"

    if result.structured_content is not None:
        return json.dumps(result.structured_content)

    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    return "\n".join(texts) if texts else "(tool returned no content)"
