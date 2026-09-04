"""Native, always-local tools for common productivity tasks: spreadsheets,
Word documents, PDFs, plain text files, web fetching, and (opt-in) email
and shell commands. These run in-process -- no MCP server needed -- and
are merged with any MCP tools (see utils/mcp_manager.py) into one `tools`
list for Foundry Local.

Deeper OS integrations that need a real running application or an OAuth
app registration (Outlook/Excel/Word desktop automation, full browser
automation) aren't implemented here -- they're a better fit for a
dedicated MCP server, which this app already knows how to use. See the
Tools section of README.md for recommended servers.

Safety:
  - File tools are sandboxed to WORKSPACE_DIR (workspace/ in the repo
    root). Every path is resolved and checked to stay inside it before
    any read/write, so a tool call can't read or overwrite files
    elsewhere on the machine, including via `..` traversal.
  - Sending email requires SMTP_HOST/SMTP_USER/SMTP_PASSWORD to be
    configured; the tool isn't even offered to the model otherwise.
  - Shell command execution is off by default. It only becomes available
    if ENABLE_SHELL_TOOL is explicitly set truthy -- see README.md for
    why that's a real decision to make, not a default to accept.

Tool calls are LLM-decided, and the LLM's context can include content
from outside the conversation (fetched web pages, MCP tool output,
uploaded documents) -- treat all of this the same as any other
prompt-injection surface: these tools do what they're told, so the
sandboxing above is the actual safety boundary, not a suggestion.
"""
import csv
import io
import os
import re

WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

MAX_READ_CHARS = 20000
MAX_FETCH_CHARS = 8000

ENABLE_SHELL_TOOL = os.environ.get("ENABLE_SHELL_TOOL", "").strip().lower() in ("1", "true", "yes")
SHELL_TOOL_TIMEOUT = int(os.environ.get("SHELL_TOOL_TIMEOUT", "30"))

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
_EMAIL_CONFIGURED = bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def _resolve_path(relative_path):
    """Resolve `relative_path` inside WORKSPACE_DIR, refusing anything
    that would escape it (absolute paths, `..` traversal, symlinked
    escapes). This is the actual sandbox -- every file tool goes through it."""
    if not relative_path or os.path.isabs(relative_path):
        raise ValueError("path must be relative (no leading '/'), e.g. 'notes.txt'")
    full = os.path.realpath(os.path.join(WORKSPACE_DIR, relative_path))
    workspace_real = os.path.realpath(WORKSPACE_DIR)
    if full != workspace_real and not full.startswith(workspace_real + os.sep):
        raise ValueError(f"'{relative_path}' resolves outside the workspace directory")
    return full


def _truncate(text, limit=MAX_READ_CHARS):
    if len(text) > limit:
        return text[:limit] + f"\n[... truncated, {len(text) - limit} more characters]"
    return text


# --- Plain text files (notepad-like) ---------------------------------

def list_workspace_files(subdirectory=""):
    base = _resolve_path(subdirectory) if subdirectory else WORKSPACE_DIR
    if not os.path.isdir(base):
        return f"'{subdirectory}' is not a directory"
    entries = []
    for root, _dirs, files in os.walk(base):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), WORKSPACE_DIR)
            entries.append(rel)
    return "\n".join(sorted(entries)) if entries else "(workspace is empty)"


def read_text_file(path):
    full = _resolve_path(path)
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return _truncate(f.read())


def write_text_file(path, content):
    full = _resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {path}"


# --- PDF ---------------------------------------------------------------

def read_pdf(path):
    from utils.doc_parser import parse_document
    full = _resolve_path(path)
    return _truncate(parse_document(full))


def create_pdf(path, text):
    import fitz  # PyMuPDF
    full = _resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)

    doc = fitz.open()
    rect = fitz.paper_rect("letter")
    margin = 50
    text_rect = fitz.Rect(margin, margin, rect.width - margin, rect.height - margin)

    # insert_textbox() returns unused/deficit area as a float, not leftover
    # text, so there's no return value to reflow from -- pre-chunk by a
    # conservative character count per page instead (a full US Letter page
    # at 11pt holds roughly 3000-4000 characters; 2500 leaves headroom).
    chunk_size = 2500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]

    for chunk in chunks:
        page = doc.new_page(width=rect.width, height=rect.height)
        page.insert_textbox(text_rect, chunk, fontsize=11)

    page_count = doc.page_count
    doc.save(full)
    doc.close()
    return f"Wrote PDF with {page_count} page(s) to {path}"


# --- Spreadsheets (CSV / XLSX) -----------------------------------------

def read_spreadsheet(path, sheet_name=None):
    full = _resolve_path(path)
    ext = os.path.splitext(full)[1].lower()

    if ext == ".csv":
        with open(full, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
    elif ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(full, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active
        rows = [[("" if c is None else c) for c in row] for row in ws.iter_rows(values_only=True)]
    else:
        raise ValueError("path must end in .csv, .xlsx, or .xlsm")

    lines = ["\t".join(str(cell) for cell in row) for row in rows]
    return _truncate("\n".join(lines))


def write_spreadsheet(path, rows, sheet_name=None):
    """rows: a list of rows, each a list of cell values."""
    full = _resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    ext = os.path.splitext(full)[1].lower()

    if ext == ".csv":
        with open(full, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    elif ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        if sheet_name:
            ws.title = sheet_name
        for row in rows:
            ws.append(row)
        wb.save(full)
    else:
        raise ValueError("path must end in .csv, .xlsx, or .xlsm")

    return f"Wrote {len(rows)} row(s) to {path}"


# --- Word documents (.docx) --------------------------------------------

def read_word_document(path):
    import docx
    full = _resolve_path(path)
    document = docx.Document(full)
    return _truncate("\n".join(p.text for p in document.paragraphs))


def write_word_document(path, content):
    """content: the document text; blank lines start a new paragraph."""
    import docx
    full = _resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)

    document = docx.Document()
    for paragraph in content.split("\n"):
        document.add_paragraph(paragraph)
    document.save(full)
    return f"Wrote Word document to {path}"


# --- Web browsing (fetch + extract readable text) ----------------------

def fetch_webpage(url):
    import requests
    from bs4 import BeautifulSoup

    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("url must start with http:// or https://")

    response = requests.get(url, timeout=15, headers={"User-Agent": "Iris-local-assistant/1.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return _truncate(text, MAX_FETCH_CHARS)


# --- Email (opt-in: only offered if SMTP_* is configured) --------------

def send_email(to, subject, body):
    if not _EMAIL_CONFIGURED:
        raise RuntimeError("Email isn't configured -- set SMTP_HOST, SMTP_USER, and SMTP_PASSWORD.")

    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [to], msg.as_string())
    return f"Email sent to {to}"


# --- Shell / PowerShell (opt-in: off unless ENABLE_SHELL_TOOL is set) --

def run_shell_command(command):
    if not ENABLE_SHELL_TOOL:
        raise RuntimeError("Shell command execution is disabled -- set ENABLE_SHELL_TOOL=true to enable it (see README.md for what that means).")

    import subprocess
    if os.name == "nt":
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        argv = ["/bin/sh", "-c", command]

    result = subprocess.run(
        argv, cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=SHELL_TOOL_TIMEOUT,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return _truncate(f"[exit code {result.returncode}]\n{output}", 4000)


# --- Tool registration ---------------------------------------------------

_STATIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "local__list_workspace_files",
            "description": "List files in the local workspace (the sandboxed folder these tools read/write in).",
            "parameters": {
                "type": "object",
                "properties": {"subdirectory": {"type": "string", "description": "Optional subdirectory to list, relative to the workspace root"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__read_text_file",
            "description": "Read a plain text file (like Notepad) from the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path relative to the workspace root, e.g. 'notes.txt'"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__write_text_file",
            "description": "Write (create or overwrite) a plain text file in the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root, e.g. 'notes.txt'"},
                    "content": {"type": "string", "description": "The full text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__read_pdf",
            "description": "Extract the text content of a PDF file in the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "PDF path relative to the workspace root"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__create_pdf",
            "description": "Create a new PDF file containing the given text, in the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "PDF path to create, relative to the workspace root"},
                    "text": {"type": "string", "description": "The text content of the PDF"},
                },
                "required": ["path", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__read_spreadsheet",
            "description": "Read a spreadsheet (.csv, .xlsx, or .xlsm) from the local workspace and return its rows as tab-separated text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Spreadsheet path relative to the workspace root"},
                    "sheet_name": {"type": "string", "description": "Sheet name for .xlsx files (optional; defaults to the active sheet)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__write_spreadsheet",
            "description": "Write rows of data to a new spreadsheet (.csv, .xlsx, or .xlsm) in the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Spreadsheet path to create, relative to the workspace root"},
                    "rows": {
                        "type": "array",
                        "description": "Rows of data; each row is an array of cell values",
                        "items": {"type": "array", "items": {}},
                    },
                    "sheet_name": {"type": "string", "description": "Sheet name for .xlsx files (optional)"},
                },
                "required": ["path", "rows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__read_word_document",
            "description": "Read the text content of a Word document (.docx) in the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Document path relative to the workspace root"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__write_word_document",
            "description": "Create a new Word document (.docx) with the given text in the local workspace. Blank lines start a new paragraph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Document path to create, relative to the workspace root"},
                    "content": {"type": "string", "description": "The text content of the document"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local__fetch_webpage",
            "description": "Fetch a web page and return its readable text content (for web browsing / research).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch, including http:// or https://"}},
                "required": ["url"],
            },
        },
    },
]

_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "local__send_email",
        "description": "Send an email via the configured SMTP account.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "local__run_shell_command",
        "description": "Run a shell command (PowerShell on Windows, sh elsewhere) in the local workspace directory and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run"}},
            "required": ["command"],
        },
    },
}

_HANDLERS = {
    "local__list_workspace_files": lambda a: list_workspace_files(a.get("subdirectory", "")),
    "local__read_text_file": lambda a: read_text_file(a["path"]),
    "local__write_text_file": lambda a: write_text_file(a["path"], a["content"]),
    "local__read_pdf": lambda a: read_pdf(a["path"]),
    "local__create_pdf": lambda a: create_pdf(a["path"], a["text"]),
    "local__read_spreadsheet": lambda a: read_spreadsheet(a["path"], a.get("sheet_name")),
    "local__write_spreadsheet": lambda a: write_spreadsheet(a["path"], a["rows"], a.get("sheet_name")),
    "local__read_word_document": lambda a: read_word_document(a["path"]),
    "local__write_word_document": lambda a: write_word_document(a["path"], a["content"]),
    "local__fetch_webpage": lambda a: fetch_webpage(a["url"]),
    "local__send_email": lambda a: send_email(a["to"], a["subject"], a["body"]),
    "local__run_shell_command": lambda a: run_shell_command(a["command"]),
}


def get_tool_schemas():
    tools = list(_STATIC_TOOLS)
    if _EMAIL_CONFIGURED:
        tools.append(_EMAIL_TOOL)
    if ENABLE_SHELL_TOOL:
        tools.append(_SHELL_TOOL)
    return tools


def is_local_tool(qualified_name):
    return qualified_name in _HANDLERS


def call_tool(qualified_name, arguments):
    handler = _HANDLERS.get(qualified_name)
    if handler is None:
        return f"Error: unknown tool '{qualified_name}'"
    try:
        return str(handler(arguments))
    except Exception as e:
        return f"Error: {e}"
