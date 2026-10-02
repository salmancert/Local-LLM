"""Terminal agent for working with finance files, powered by a local Ollama model.

Usage:
    python agent.py                      # work on files in the current directory
    python agent.py ~/Documents/finance  # work on files in another directory
    python agent.py --yes                # don't ask before running code or writing files

Everything runs locally: file contents are only sent to the Ollama server at OLLAMA_URL.
"""
import argparse
import datetime
import difflib
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import requests
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from utils.ollama_client import OLLAMA_URL

try:
    import readline  # noqa: F401  (arrow keys and history for input() on Linux/macOS)
except ImportError:
    pass

# Needs a model with tool calling support, e.g. qwen2.5:7b, llama3.1:8b, mistral-nemo, qwen3:8b
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:7b")
# Ollama's default context (2-4k tokens) is too small for file contents and tool results
AGENT_CTX = int(os.environ.get("AGENT_CTX", "16384"))
MAX_STEPS = 20            # tool calls per user message before the agent must stop
MAX_TOOL_OUTPUT = 12000   # characters of tool output sent back to the model
COMMAND_TIMEOUT = 300     # seconds for run_python / run_shell
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", "chroma_store", "__MACOSX"}
EXCEL_EXTS = {".xlsx", ".xlsm", ".xls", ".ods"}
EXTRACT_DIR = "_extracted"             # zip archives are unpacked here inside the workspace
MAX_EXTRACT_BYTES = 5 * 1024 ** 3      # refuse archives that would unpack to more than 5 GB
CONVERTED_DIR = os.path.join(tempfile.gettempdir(), "agent-converted")  # set to the workspace in main()

console = Console()


# ---------------------------------------------------------------- tools

class ToolError(Exception):
    pass


class Workspace:
    def __init__(self, root):
        self.root = os.path.realpath(root)

    def resolve(self, path):
        full = os.path.realpath(os.path.join(self.root, os.path.expanduser(path or ".")))
        try:
            inside = os.path.commonpath([full, self.root]) == self.root
        except ValueError:  # different drives on Windows
            inside = False
        if not inside:
            raise ToolError(f"{path} is outside the workspace {self.root}")
        return full

    def rel(self, full):
        return os.path.relpath(full, self.root)

    def walk(self, path=".", pattern="*"):
        base = self.resolve(path)
        if os.path.isfile(base):
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith((".", "~$")):  # hidden files and Office lock files
                    continue
                full = os.path.join(dirpath, name)
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(self.rel(full), pattern):
                    yield full


def load_table(full, sheet=None):
    import pandas as pd
    ext = os.path.splitext(full)[1].lower()
    if ext in EXCEL_EXTS:
        sheets = pd.ExcelFile(full).sheet_names
        df = pd.read_excel(full, sheet_name=sheet if sheet is not None else sheets[0])
        return df, sheets
    # sep=None sniffs the delimiter (comma, semicolon, tab, pipe) as bank exports vary
    return pd.read_csv(full, sep=None, engine="python"), None


_text_cache = {}


def file_text(full):
    """Return the text of a file as lines, converting PDF, Word and Excel files."""
    key = (full, os.path.getmtime(full), os.path.getsize(full))
    if key not in _text_cache:
        _text_cache[key] = _file_text(full)
    return _text_cache[key]


def _file_text(full):
    ext = os.path.splitext(full)[1].lower()
    if ext == ".pdf":
        return pdf_lines(full)
    if ext in (".docx", ".docm"):
        return docx_lines(full)
    if ext in (".doc", ".rtf", ".odt"):
        return docx_lines(convert_to_docx(full))
    if ext in EXCEL_EXTS:
        import pandas as pd
        lines = []
        for name, df in pd.read_excel(full, sheet_name=None).items():
            lines.append(f"----- sheet {name} -----")
            lines.extend(df.to_csv(index=False).splitlines())
        return lines
    if ext == ".zip":
        raise ToolError(f"{os.path.basename(full)} is a zip archive; its contents are in {EXTRACT_DIR}/")
    with open(full, "rb") as f:
        data = f.read()
    if b"\0" in data[:4096]:
        raise ToolError(f"{os.path.basename(full)} is a binary file and cannot be read as text")
    return data.decode("utf-8", errors="replace").splitlines()


def pdf_lines(full):
    from utils.doc_parser import fitz
    lines = []
    with fitz.open(full) as doc:
        if doc.needs_pass:
            raise ToolError(f"{os.path.basename(full)} is password protected")
        for number, page in enumerate(doc, 1):
            lines.append(f"----- page {number} -----")
            text = page.get_text()
            if not text.strip() and page.get_images():
                text = ocr_page(page)
            lines.extend(text.splitlines())
    return lines


def ocr_page(page):
    """Scanned pages have no text layer; read them with Tesseract OCR when it is installed."""
    if shutil.which("tesseract") is None:
        return "[scanned page without text; install tesseract-ocr to read it]"
    try:
        return "[OCR] " + page.get_textpage_ocr(full=True, dpi=300).extractText()
    except Exception as e:
        return f"[scanned page, OCR failed: {e}]"


def docx_lines(full):
    import docx
    from docx.table import Table
    document = docx.Document(full)
    lines = []
    for block in document.iter_inner_content():  # paragraphs and tables in document order
        if isinstance(block, Table):
            lines.append("[table]")
            for row in block.rows:
                cells = []
                for cell in row.cells:
                    text = " ".join(cell.text.split())
                    if not cells or cells[-1] != text:  # merged cells repeat their text
                        cells.append(text)
                lines.append(" | ".join(cells))
            lines.append("[/table]")
        elif block.text.strip():
            style = block.style.name if block.style is not None else ""
            prefix = "#" * int(style[-1]) + " " if re.fullmatch(r"Heading [1-6]", style) else ""
            lines.append(prefix + block.text)
    return lines


def convert_to_docx(full):
    return convert_document(full, "docx")


def convert_document(full, fmt):
    """Convert legacy Office files (.doc, .xls, ...) to `fmt` with LibreOffice. Converted copies are
    cached inside the workspace (not the shared temp folder) so client data stays in the client's folder."""
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if office is None:
        raise ToolError(f"reading {os.path.basename(full)} needs LibreOffice "
                        f"(Ubuntu: sudo apt install libreoffice-writer-nogui), or save it as .{fmt}")
    stamp = hashlib.sha1(f"{full}:{os.path.getmtime(full)}".encode()).hexdigest()[:16]
    target = os.path.join(CONVERTED_DIR, stamp, os.path.splitext(os.path.basename(full))[0] + "." + fmt)
    if not os.path.exists(target):
        subprocess.run([office, "--headless", "--convert-to", fmt, "--outdir", os.path.dirname(target), full],
                       capture_output=True, timeout=180)
        if not os.path.exists(target):
            raise ToolError(f"LibreOffice could not convert {os.path.basename(full)}")
    return target


def extract_archives(ws):
    """Unpack every .zip in the workspace (and zips inside them) into _extracted/ so the tools can read
    the contents. Archives that were already unpacked and have not changed are skipped."""
    messages = []
    for _ in range(5):  # levels of zips inside zips
        found_new = False
        for full in list(ws.walk(".", "*.zip")):
            rel = ws.rel(full)
            if rel.split(os.sep)[0] == EXTRACT_DIR:
                target = os.path.splitext(full)[0]  # nested zip: unpack next to it
            else:
                target = os.path.join(ws.root, EXTRACT_DIR, os.path.splitext(rel)[0])
            stamp = f"{os.path.getsize(full)}:{os.path.getmtime(full)}"
            marker = os.path.join(target, ".extracted_from_zip")
            if os.path.exists(marker) and open(marker).read().split("\n")[0] == stamp:
                continue
            found_new = True
            try:
                count = unzip(full, target)
                with open(marker, "w") as f:
                    f.write(stamp)
                messages.append(f"unpacked {rel} ({count} files) into {ws.rel(target)}")
            except (zipfile.BadZipFile, RuntimeError, ToolError, OSError) as e:
                os.makedirs(target, exist_ok=True)
                with open(marker, "w") as f:  # don't retry a broken archive on every start
                    f.write(f"{stamp}\nerror: {e}")
                messages.append(f"could not unpack {rel}: {e}")
        if not found_new:
            break
    return messages


def archive_status(ws, full):
    """'unpacked', or the reason the archive could not be unpacked."""
    rel = ws.rel(full)
    if rel.split(os.sep)[0] == EXTRACT_DIR:
        target = os.path.splitext(full)[0]
    else:
        target = os.path.join(ws.root, EXTRACT_DIR, os.path.splitext(rel)[0])
    try:
        with open(os.path.join(target, ".extracted_from_zip")) as f:
            lines = f.read().split("\n")
    except OSError:
        return "not unpacked"
    return lines[1] if len(lines) > 1 else f"unpacked into {ws.rel(target)}"


def unzip(full, target):
    with zipfile.ZipFile(full) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if any(m.flag_bits & 0x1 for m in members):
            raise ToolError("the archive is password protected, unzip it manually")
        if sum(m.file_size for m in members) > MAX_EXTRACT_BYTES:
            raise ToolError("the archive would unpack to more than 5 GB")
        root = os.path.realpath(target)
        os.makedirs(root, exist_ok=True)
        for member in members:
            dest = os.path.realpath(os.path.join(root, member.filename))
            if os.path.commonpath([dest, root]) != root:
                continue  # skip entries like ../../x that would escape the folder
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with archive.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        return len(members)


def tool_list_files(ws, path=".", pattern="*"):
    rows = []
    for full in ws.walk(path, pattern):
        stat = os.stat(full)
        modified = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        rows.append(f"{stat.st_size:>12,}  {modified}  {ws.rel(full)}")
        if len(rows) >= 300:
            rows.append("... (more files, narrow the path or pattern)")
            break
    return "\n".join(rows) or "No files found."


def tool_read_file(ws, path, offset=1, limit=200):
    full = ws.resolve(path)
    if not os.path.isfile(full):
        raise ToolError(f"{path} does not exist or is not a file")
    lines = file_text(full)
    offset, limit = max(int(offset), 1), max(int(limit), 1)
    chunk = lines[offset - 1:offset - 1 + limit]
    out = "\n".join(f"{n:>6}  {line}" for n, line in enumerate(chunk, offset))
    end = offset - 1 + len(chunk)
    if end < len(lines):
        out += f"\n... showing lines {offset}-{end} of {len(lines)}, use offset={end + 1} to read more"
    return out or "(empty file)"


def tool_inspect_table(ws, path, sheet=None):
    import pandas as pd
    full = ws.resolve(path)
    df, sheets = load_table(full, sheet)
    parts = []
    if sheets:
        parts.append(f"Sheets: {sheets} (showing {sheet if sheet is not None else sheets[0]!r})")
    parts.append(f"Rows: {len(df)}, columns: {len(df.columns)}")
    parts.append("Columns (dtype, non-null count, example):")
    for col in df.columns:
        sample = df[col].dropna()
        example = repr(sample.iloc[0]) if len(sample) else "-"
        parts.append(f"  {col!r}: {df[col].dtype}, {df[col].notna().sum()} non-null, e.g. {example}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        parts.append("First rows:\n" + df.head(5).to_string())
        numeric = df.select_dtypes("number")
        if not numeric.empty:
            parts.append("Numeric summary:\n" + numeric.describe().round(2).to_string())
    return "\n".join(parts)


def tool_search_files(ws, pattern, path=".", file_pattern="*"):
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ToolError(f"invalid regex: {e}")
    matches = []
    for full in ws.walk(path, file_pattern):
        try:
            lines = file_text(full)
        except Exception:
            continue  # binary or unreadable
        for n, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{ws.rel(full)}:{n}: {line.strip()[:200]}")
                if len(matches) >= 100:
                    return "\n".join(matches) + "\n... (stopped at 100 matches)"
    return "\n".join(matches) or "No matches."


def _run(args, ws, **kwargs):
    try:
        proc = subprocess.run(args, cwd=ws.root, capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT, **kwargs)
    except subprocess.TimeoutExpired:
        raise ToolError(f"timed out after {COMMAND_TIMEOUT} seconds")
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    return f"{out.strip() or '(no output)'}\n[exit code {proc.returncode}]"


PYTHON_PRELUDE = """\
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)
"""


def tool_run_python(ws, code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(PYTHON_PRELUDE + code)
    try:
        return _run([sys.executable, f.name], ws)
    finally:
        os.remove(f.name)


def tool_run_shell(ws, command):
    return _run(command, ws, shell=True)


def tool_write_file(ws, path, content):
    full = ws.resolve(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return f"Wrote {len(content):,} characters to {ws.rel(full)}"


def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SCHEMAS = [
    _fn("list_files", "List files (size in bytes, modified date, path) under a directory of the workspace.",
        {"path": {"type": "string", "description": "Directory relative to the workspace, default '.'"},
         "pattern": {"type": "string", "description": "Glob such as '*.csv' or '2024/*', default '*'"}}, []),
    _fn("read_file", "Read a text, CSV, PDF, Word or Excel file as numbered lines (converted to text). "
        "Word tables are shown as rows of cells separated by |.",
        {"path": {"type": "string"},
         "offset": {"type": "integer", "description": "First line to read, default 1"},
         "limit": {"type": "integer", "description": "Number of lines, default 200"}}, ["path"]),
    _fn("inspect_table", "Describe a CSV or Excel file: sheets, row count, columns with types, first rows and "
        "numeric summary. Use this before analysing a table.",
        {"path": {"type": "string"},
         "sheet": {"type": "string", "description": "Excel sheet name, default the first sheet"}}, ["path"]),
    _fn("search_files", "Case-insensitive regex search through file contents (including PDF, Word and Excel).",
        {"pattern": {"type": "string", "description": "Regular expression"},
         "path": {"type": "string", "description": "File or directory, default '.'"},
         "file_pattern": {"type": "string", "description": "Glob to filter files, e.g. '*.csv'"}}, ["pattern"]),
    _fn("run_python", "Run a Python script in the workspace directory and return its printed output. pandas is "
        "already imported as pd. Use it for every calculation: totals, grouping, filtering, reconciliation. "
        "Always print() the results.",
        {"code": {"type": "string", "description": "Python source code"}}, ["code"]),
    _fn("run_shell", "Run a shell command in the workspace directory and return its output.",
        {"command": {"type": "string"}}, ["command"]),
    _fn("write_file", "Create or overwrite a text file (e.g. a CSV or Markdown report) in the workspace.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
]

TOOLS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "inspect_table": tool_inspect_table,
    "search_files": tool_search_files,
    "run_python": tool_run_python,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
}
NEEDS_APPROVAL = {"run_python", "run_shell", "write_file"}


# ---------------------------------------------------------------- agent

SYSTEM_PROMPT = """You are a careful finance assistant running in the user's terminal. You work with the \
user's local files (bank statements, invoices, budgets, ledgers, spreadsheets, PDFs) through tools.

Workspace: {root}
Today: {today}. Operating system: {os}.
Zip archives are already unpacked into {extract_dir}/<archive name>/; read the files there.
_qcr/ holds quality control review outputs produced by qcr.py; they are not audit evidence.
Files in the workspace:
{files}

How to work:
- Look at the files with tools before answering. Never guess file contents, figures or column names.
- For CSV/Excel files call inspect_table first, then use run_python with pandas for every calculation \
(sums, averages, grouping by month or category, matching transactions). Do not do arithmetic in your head.
- Report amounts with their currency and sign conventions, and say which file and columns they came from.
- Flag anything that looks wrong: duplicates, missing dates, totals that don't reconcile.
- Never modify the user's original files. Only write new files when the user asks for output.
- If a tool returns an error, fix the call and try again.
- Keep answers short and use Markdown tables for tabular results."""


class Agent:
    def __init__(self, ws, model, auto_approve=False):
        self.ws = ws
        self.model = model
        self.approved = set(NEEDS_APPROVAL) if auto_approve else set()
        self.tokens = 0
        self.reset()

    def reset(self):
        listing = tool_list_files(self.ws).splitlines()
        if len(listing) > 60:
            listing = listing[:60] + [f"... and {len(listing) - 60} more (use list_files)"]
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            root=self.ws.root, today=datetime.date.today().isoformat(), extract_dir=EXTRACT_DIR,
            os=f"{platform.system()} {platform.release()}", files="\n".join(listing))}]
        self.tokens = 0

    # ---- model call
    def chat(self):
        """Stream one assistant message, rendering it live. Returns the message dict."""
        payload = {"model": self.model, "messages": self.messages, "tools": TOOL_SCHEMAS,
                   "stream": True, "options": {"num_ctx": AGENT_CTX}}
        content, tool_calls = "", []
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10, 600)) as r:
            if r.status_code != 200:
                try:
                    error = r.json().get("error", r.text)
                except ValueError:
                    error = r.text
                raise RuntimeError(error)
            spinner = Spinner("dots", text=Text("thinking…", style="dim"))
            with Live(spinner, console=console, refresh_per_second=12, vertical_overflow="visible") as live:
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"])
                    message = chunk.get("message", {})
                    content += message.get("content", "")
                    tool_calls += message.get("tool_calls") or []
                    # Hold back text that may turn out to be a tool call written as JSON
                    if content.strip() and not looks_like_tool_call(content):
                        live.update(Markdown(content))
                    if chunk.get("done"):
                        self.tokens = chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                if tool_calls or parse_text_tool_calls(content) or not content.strip():
                    live.update(Text(""))  # only tool calls: clear the spinner
                else:
                    live.update(Markdown(content))
        message = {"role": "assistant", "content": content}
        if not tool_calls:
            tool_calls = parse_text_tool_calls(content)
            if tool_calls:
                message["content"] = ""
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    # ---- tools
    def approve(self, name, args):
        if name in self.approved:
            return True, None
        if name == "run_python":
            body = Syntax(args.get("code", ""), "python", line_numbers=True, word_wrap=True)
        elif name == "write_file":
            body = write_preview(self.ws, args.get("path", ""), args.get("content", ""))
        else:
            body = Text(str(args.get("command", json.dumps(args))))
        console.print(Panel(body, title=f"[bold]{name}[/]", title_align="left", border_style="yellow"))
        answer = console.input("[yellow]Allow? [bold]y[/]es / [bold]n[/]o / [bold]a[/]lways for this "
                               "tool / or type what to do instead: [/]").strip()
        if answer.lower() in ("y", "yes", ""):
            return True, None
        if answer.lower() in ("a", "always"):
            self.approved.add(name)
            return True, None
        if answer.lower() in ("n", "no"):
            return False, "The user declined this tool call. Ask them how to proceed."
        return False, f"The user declined this tool call and said: {answer}"

    def run_tool(self, call):
        function = call.get("function", {})
        name = function.get("name", "")
        args = function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        console.print(f"[bold cyan]●[/] [bold]{escape(name)}[/]({escape(format_args(args))})")
        if name not in TOOLS:
            result = f"Error: unknown tool {name!r}. Available tools: {', '.join(TOOLS)}"
        else:
            allowed, refusal = (True, None) if name not in NEEDS_APPROVAL else self.approve(name, args)
            if not allowed:
                result = refusal
            else:
                try:
                    result = TOOLS[name](self.ws, **args)
                except ToolError as e:
                    result = f"Error: {e}"
                except TypeError as e:
                    result = f"Error: bad arguments for {name}: {e}"
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
        preview = result.strip("\n").splitlines() or [""]
        more = f" … (+{len(preview) - 4} lines)" if len(preview) > 4 else ""
        console.print(Text("  ⎿  " + "\n     ".join(p[:150] for p in preview[:4]) + more, style="dim"))
        return name, truncate(result)

    # ---- one user turn
    def turn(self, text):
        self.messages.append({"role": "user", "content": text})
        for _ in range(MAX_STEPS):
            message = self.chat()
            self.messages.append(message)
            if not message.get("tool_calls"):
                return
            for call in message["tool_calls"]:
                name, result = self.run_tool(call)
                self.messages.append({"role": "tool", "tool_name": name, "content": result})
        console.print(f"[yellow]Stopped after {MAX_STEPS} tool calls. Say 'continue' to keep going.[/]")


def looks_like_tool_call(content):
    return content.lstrip().startswith(("{", "<tool_call>", "```json"))


def parse_text_tool_calls(content):
    """Some small models write the tool call as JSON text instead of using native tool calls."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    text = re.sub(r"^<tool_call>\s*|\s*</tool_call>$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        return []
    calls = data if isinstance(data, list) else [data]
    result = []
    for call in calls:
        if not isinstance(call, dict):
            return []
        call = call.get("function", call)
        name = call.get("name")
        args = call.get("arguments", call.get("parameters", {}))
        if name not in TOOLS or not isinstance(args, (dict, str)):
            return []
        result.append({"function": {"name": name, "arguments": args}})
    return result


def format_args(args):
    parts = []
    for key, value in args.items():
        value = str(value)
        first = value.splitlines()[0] if value else ""
        if len(first) > 60 or "\n" in value:
            first = first[:60] + "…"
        parts.append(f"{key}={first!r}" if key not in ("code", "content") else f"{key}=…")
    return ", ".join(parts)


def truncate(text):
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    half = MAX_TOOL_OUTPUT // 2
    return (text[:half] + f"\n... [{len(text) - MAX_TOOL_OUTPUT:,} characters truncated] ...\n"
            + text[-half:])


def write_preview(ws, path, content):
    try:
        full = ws.resolve(path)
    except ToolError as e:
        return str(e)
    if not os.path.exists(full):
        lines = content.splitlines()
        shown = "\n".join(lines[:40]) + (f"\n… (+{len(lines) - 40} lines)" if len(lines) > 40 else "")
        return f"New file {path}\n\n{shown}"
    with open(full, encoding="utf-8", errors="replace") as f:
        old = f.read().splitlines()
    diff = list(difflib.unified_diff(old, content.splitlines(), path, path, lineterm=""))
    shown = diff[:60] + ([f"… (+{len(diff) - 60} lines)"] if len(diff) > 60 else [])
    return Syntax("\n".join(shown) or "(no changes)", "diff")


HELP = """[bold]Commands[/]
  /help           show this help
  /clear          start a new conversation (forget the context)
  /model [name]   show or switch the Ollama model
  /auto           toggle asking before run_python / run_shell / write_file
  /tools          list the tools the model can use
  /extract        unpack zip files added since the agent started
  /save           save this conversation (with tool calls) to _agent_logs/ as an audit trail
  /exit           quit (or Ctrl+D)
End a line with \\ to keep typing on the next line. Ctrl+C stops a running answer."""


def save_transcript(agent):
    folder = os.path.join(agent.ws.root, "_agent_logs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".md")
    out = [f"# Agent session, {agent.ws.root}", f"Model: {agent.model}", ""]
    for m in agent.messages[1:]:
        if m["role"] == "user":
            out += ["## User", m["content"], ""]
        elif m["role"] == "assistant":
            if m.get("content"):
                out += ["## Assistant", m["content"], ""]
            for call in m.get("tool_calls", []):
                f = call.get("function", {})
                out += [f"### Tool call: {f.get('name')}", "```json",
                        json.dumps(f.get("arguments"), indent=2, ensure_ascii=False), "```", ""]
        elif m["role"] == "tool":
            out += [f"### Result: {m.get('tool_name')}", "```", m["content"], "```", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return agent.ws.rel(path)


def check_model(model):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=10)
    except requests.RequestException:
        console.print(f"[red]Cannot reach Ollama at {OLLAMA_URL}. Is it running? "
                      "(Linux: systemctl status ollama, Windows: start the Ollama app)[/]")
        return False
    if r.status_code == 404:
        console.print(f"[red]Model {model!r} is not installed. Run: ollama pull {model}[/]")
        return False
    capabilities = r.json().get("capabilities")
    if capabilities is not None and "tools" not in capabilities:
        console.print(f"[yellow]Warning: {model!r} does not support tool calling. "
                      "Try qwen2.5:7b, llama3.1:8b or qwen3:8b.[/]")
    return True


def read_input():
    lines = []
    while True:
        line = console.input("[bold green]›[/] " if not lines else "[bold green]…[/] ")
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Local terminal agent for finance files (Ollama)")
    parser.add_argument("directory", nargs="?", default=".", help="workspace directory (default: current)")
    parser.add_argument("--model", default=AGENT_MODEL, help=f"Ollama model (default: {AGENT_MODEL})")
    parser.add_argument("--yes", action="store_true", help="run code and write files without asking")
    parser.add_argument("-p", "--prompt", help="answer one prompt and exit")
    parser.add_argument("--no-extract", action="store_true", help="don't unpack zip files into _extracted/")
    opts = parser.parse_args()

    root = os.path.expanduser(opts.directory)
    if not os.path.isdir(root):
        sys.exit(f"Not a directory: {opts.directory}")
    ws = Workspace(root)
    global CONVERTED_DIR
    CONVERTED_DIR = os.path.join(ws.root, EXTRACT_DIR, ".converted")
    host = re.sub(r"^\w+://|[:/].*$", "", OLLAMA_URL)
    if host not in ("localhost", "127.0.0.1", "::1", "[::1]"):
        console.print(f"[bold yellow]Note: OLLAMA_URL points to {host}, so file contents are sent to that "
                      "machine, not processed on this one.[/]")
    if not check_model(opts.model):
        sys.exit(1)
    if not opts.no_extract:
        with console.status("[dim]unpacking zip files…[/]"):
            for line in extract_archives(ws):
                console.print(f"[dim]{escape(line)}[/]")
    agent = Agent(ws, opts.model, auto_approve=opts.yes)

    def ask(text):
        try:
            agent.turn(text)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/]")
        except requests.RequestException as e:
            console.print(f"[red]Ollama request failed: {e}[/]")
        except RuntimeError as e:
            console.print(f"[red]Ollama error: {e}[/]")
        if agent.tokens > AGENT_CTX * 0.85:
            console.print(f"[yellow]Context is {agent.tokens:,}/{AGENT_CTX:,} tokens, older messages will be "
                          "forgotten. Use /clear to start fresh.[/]")

    if opts.prompt:
        ask(opts.prompt)
        return

    console.print(Panel(f"[bold]Local finance agent[/]  ·  model [cyan]{opts.model}[/]\n"
                        f"workspace [cyan]{ws.root}[/]\n[dim]/help for commands · everything stays on "
                        "this machine[/]", border_style="cyan"))
    while True:
        try:
            text = read_input()
        except KeyboardInterrupt:
            console.print("[dim](use /exit or Ctrl+D to quit)[/]")
            continue
        except EOFError:
            break
        if not text:
            continue
        if text.startswith("/"):
            command, _, arg = text.partition(" ")
            if command in ("/exit", "/quit"):
                break
            elif command == "/help":
                console.print(HELP)
            elif command == "/clear":
                agent.reset()
                console.print("[dim]Conversation cleared.[/]")
            elif command == "/model":
                if arg.strip() and check_model(arg.strip()):
                    agent.model = arg.strip()
                console.print(f"Model: [cyan]{agent.model}[/]")
            elif command == "/auto":
                agent.approved = set() if agent.approved == NEEDS_APPROVAL else set(NEEDS_APPROVAL)
                state = "on (no confirmation)" if agent.approved == NEEDS_APPROVAL else "off (ask first)"
                console.print(f"Auto-approve: {state}")
            elif command == "/extract":
                for line in extract_archives(ws) or ["no new zip files"]:
                    console.print(f"[dim]{escape(line)}[/]")
                agent.reset()
                console.print("[dim]Conversation restarted with the updated file list.[/]")
            elif command == "/save":
                console.print(f"Saved the conversation to [cyan]{save_transcript(agent)}[/]")
            elif command == "/tools":
                for schema in TOOL_SCHEMAS:
                    f = schema["function"]
                    console.print(f"[bold]{f['name']}[/]: {f['description']}")
            else:
                console.print(f"Unknown command {command}. /help lists commands.")
            continue
        ask(text)
        if agent.tokens:
            console.print(f"[dim]{agent.tokens:,}/{AGENT_CTX:,} tokens of context used[/]")


if __name__ == "__main__":
    main()
