"""Complete your QCR checklist (Excel or Word) from a client's audit files, fully locally.

Reads your own checklist, finds the question / answer / comment / working paper reference columns,
then for every requirement searches all the client's files (zips, Word, Excel, PDF, including
scanned pages) for evidence, asks the local Ollama model for a conclusion, checks the quoted
evidence really is in the files, and writes the answers into a copy of your checklist.

Usage:
    qcr CHECKLIST.xlsx ~/Clients/ClientA              # fill the checklist for one client
    qcr CHECKLIST.docx ~/Clients --all                # one completed checklist per client folder
    qcr CHECKLIST.xlsx ~/Clients/ClientA --dry-run    # only show how the checklist was understood
(or `python qcr.py ...` with the venv active)

The completed copy is saved in <client folder>/_qcr/. Your original checklist is never changed.
AI-filled cells are shaded yellow; every answer must be checked by the reviewer.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter

import requests
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table as RichTable

import agent
from utils.ollama_client import OLLAMA_URL

QCR_MODEL = os.environ.get("QCR_MODEL", agent.AGENT_MODEL)
OUTPUT_DIR = "_qcr"
OWN_DIRS = {OUTPUT_DIR, "_agent_logs"}   # tool output, never part of the audit file
PASSAGE_CHARS = 1200   # size of the text passages the model sees
TOP_PASSAGES = 8       # passages per checklist item
MAX_PER_FILE = 3       # so one long document does not crowd out the others
STATUSES = ["Satisfactory", "Finding", "Not evidenced", "Not applicable"]
SEVERITIES = ["High", "Medium", "Low", "None"]
DEFAULT_LABELS = {"Satisfactory": "Yes", "Finding": "No", "Not applicable": "N/A", "Not evidenced": "Not evidenced"}
HIGHLIGHT = "FFF2CC"   # pale yellow on every cell the tool wrote

console = Console()


# ---------------------------------------------------------------- reading the checklist

# Header text -> column role. Checked in this order, first match wins for each header cell.
ROLE_PATTERNS = [
    ("evidence", r"w/?p\b|working ?paper|evidence|file ref|cross.?ref|location|where (?:documented|filed)"),
    ("answer", r"yes ?/ ?no|\by ?/ ?n\b|answer|response|complied|compliance|status|result|tick|outcome|"
               r"satisf|\bdone\b|✓|^conclusion$"),
    ("comment", r"comment|remark|note|explanation|finding|observation|detail|conclusion|reviewer"),
    ("question", r"question|requirement|procedure|check|item|description|criteri|\btest|matter|point|"
                 r"consideration|step|query|area|standard"),
]
OWN_HEADERS = {"answer": "AI answer", "comment": "AI comment", "evidence": "AI evidence"}


def detect_roles(headers):
    """Map column index -> role from a list of header strings."""
    roles = {}
    for idx, text in enumerate(headers):
        text = " ".join(str(text or "").split()).lower()
        if not text or len(text) > 80:
            continue
        for role, pattern in ROLE_PATTERNS:
            if role not in roles.values() and re.search(pattern, text):
                roles[idx] = role
                break
    return {role: idx for idx, role in roles.items()}


def is_question(text):
    """Checklist rows ask something; short label rows are section headings used as context."""
    words = len(text.split())
    return text.endswith("?") or words >= 5


class Item:
    """One checklist requirement and where its answers go."""

    def __init__(self, key, label, question, context, existing):
        self.key, self.label, self.question, self.context, self.existing = key, label, question, context, existing
        self.write = None   # function(role, text) that writes into the checklist
        self.roles = set()  # roles available for this item: answer / comment / evidence
        self.options = None  # allowed answers (Excel dropdown), if any


class ExcelChecklist:
    def __init__(self, path, sheet=None, overrides=None):
        import openpyxl
        self.keep_vba = path.lower().endswith(".xlsm")
        self.wb = openpyxl.load_workbook(path, keep_vba=self.keep_vba)
        self.ext = ".xlsm" if self.keep_vba else ".xlsx"
        self.overrides = overrides or {}
        self.sheets = [self.wb[sheet]] if sheet else self.wb.worksheets
        self.notes = []

    def items(self):
        items = []
        for ws in self.sheets:
            items += self._sheet_items(ws)
        return items

    def _find_header(self, ws):
        best = None
        for r in range(1, min(ws.max_row, 40) + 1):
            values = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
            roles = detect_roles(values)
            if "question" in roles and (best is None or len(roles) > len(best[1])):
                best = (r, roles)
        return best

    def _sheet_items(self, ws):
        from openpyxl.utils import column_index_from_string, get_column_letter
        header = self._find_header(ws)
        roles = {}
        if header:
            header_row, found = header
            roles = {role: idx + 1 for role, idx in found.items()}
        else:
            header_row = 0
        for role in ("question", "answer", "comment", "evidence"):
            if self.overrides.get(role):
                roles[role] = column_index_from_string(self.overrides[role].upper())
        if "question" not in roles:
            # No recognisable header: use the column with the most text as the questions
            totals = Counter()
            for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 200)):
                for cell in row:
                    if isinstance(cell.value, str):
                        totals[cell.column] += len(cell.value)
            if not totals:
                self.notes.append(f"sheet '{ws.title}': no text, skipped")
                return []
            roles["question"] = totals.most_common(1)[0][0]
        if not ({"answer", "comment"} & roles.keys()):
            # Nowhere to write: add our own columns to the right of the table
            col = ws.max_column + 1
            hdr = header_row or 1
            for role in ("answer", "comment", "evidence"):
                if role not in roles:
                    roles[role] = col
                    ws.cell(hdr, col).value = OWN_HEADERS[role]
                    ws.column_dimensions[get_column_letter(col)].width = 18 if role == "answer" else 50
                    col += 1
            self.notes.append(f"sheet '{ws.title}': no answer/comment column found, added AI columns")
        self.notes.append(f"sheet '{ws.title}': header row {header_row or '-'}, " + ", ".join(
            f"{role} = column {get_column_letter(c)}" for role, c in sorted(roles.items(), key=lambda x: x[1])))

        options = self._dropdown(ws, roles.get("answer"))
        items, context = [], ""
        for r in range(header_row + 1, ws.max_row + 1):
            value = ws.cell(r, roles["question"]).value
            text = " ".join(str(value).split()) if value is not None else ""
            if not text or text.startswith("="):
                continue
            if not is_question(text):
                context = text
                continue
            existing = ws.cell(r, roles["answer"]).value if "answer" in roles else None
            item = Item(f"{ws.title}!{r}", f"{ws.title} row {r}", text, context, existing)
            item.roles = {role for role in ("answer", "comment", "evidence") if role in roles}
            item.options = options
            item.write = self._writer(ws, r, roles)
            items.append(item)
        return items

    def _dropdown(self, ws, col):
        """Allowed values of a dropdown (data validation list) on the answer column, if any."""
        if not col:
            return None
        from openpyxl.utils import get_column_letter
        letter = get_column_letter(col)
        for dv in ws.data_validations.dataValidation:
            if dv.type == "list" and dv.formula1 and dv.formula1.startswith('"') and letter in str(dv.sqref):
                return [o.strip() for o in dv.formula1.strip('"').split(",") if o.strip()]
        return None

    def _writer(self, ws, row, roles):
        from openpyxl.cell.cell import MergedCell
        from openpyxl.styles import Alignment, PatternFill

        def write(role, text, highlight=True):
            if role not in roles or text is None:
                return
            cell = ws.cell(row, roles[role])
            if isinstance(cell, MergedCell) or (isinstance(cell.value, str) and cell.value.startswith("=")):
                return  # never write into merged cells or formulas
            cell.value = text
            if role != "answer":
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            if highlight:
                cell.fill = PatternFill("solid", fgColor=HIGHLIGHT)
        return write

    def save(self, path):
        self.wb.save(path)


class WordChecklist:
    def __init__(self, path, overrides=None):
        import docx
        self.doc = docx.Document(path)
        self.ext = ".docx"
        self.notes = []

    def items(self):
        from docx.shared import Inches
        from docx.table import Table
        items, context, t_idx = [], "", 0
        if not self.doc.tables:
            self.notes.append("no tables found: Word checklists must be laid out as tables")
        for block in self.doc.iter_inner_content():
            if not isinstance(block, Table):
                # A heading or short paragraph above a table names its section
                text = " ".join(block.text.split())
                style = block.style.name if block.style is not None else ""
                if text and (style.startswith("Heading") or len(text.split()) <= 8):
                    context = text
                continue
            table = block
            t_idx += 1
            if not table.rows:
                continue
            header_idx, roles = None, {}
            for h in range(min(2, len(table.rows))):
                found = detect_roles([c.text for c in table.rows[h].cells])
                if "question" in found and len(found) > len(roles):
                    header_idx, roles = h, found
            if header_idx is None:
                self.notes.append(f"table {t_idx}: no question column recognised, skipped")
                continue
            if not ({"answer", "comment"} & roles.keys()):
                table.add_column(Inches(2.0))
                roles["comment"] = len(table.columns) - 1
                table.rows[header_idx].cells[roles["comment"]].text = "AI conclusion"
                self.notes.append(f"table {t_idx}: no answer/comment column, added 'AI conclusion'")
            self.notes.append(f"table {t_idx}: " + ", ".join(
                f"{role} = column {i + 1}" for role, i in sorted(roles.items(), key=lambda x: x[1])))
            for r_idx, row in enumerate(table.rows):
                if r_idx <= header_idx:
                    continue
                cells = row.cells
                if roles["question"] >= len(cells):
                    continue
                qcell = cells[roles["question"]]
                text = " ".join(qcell.text.split())
                targets = {role: cells[i] for role, i in roles.items() if role != "question" and i < len(cells)}
                # A row merged across the table is a section heading
                if not text or all(c._tc is qcell._tc for c in targets.values()) or not is_question(text):
                    if text:
                        context = text
                    continue
                existing = targets["answer"].text.strip() if "answer" in targets else None
                item = Item(f"T{t_idx}R{r_idx}", f"table {t_idx} row {r_idx + 1}", text, context, existing or None)
                item.roles = {role for role, c in targets.items() if c._tc is not qcell._tc}
                item.write = self._writer(targets, qcell)
                items.append(item)
        return items

    def _writer(self, targets, qcell):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        def write(role, text, highlight=True):
            cell = targets.get(role)
            if cell is None or text is None or cell._tc is qcell._tc:
                return
            cell.text = text
            if highlight:
                tc_pr = cell._tc.get_or_add_tcPr()
                for old in tc_pr.findall(qn("w:shd")):
                    tc_pr.remove(old)
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), HIGHLIGHT)
                tc_pr.append(shd)
        return write

    def save(self, path):
        self.doc.save(path)


def open_checklist(path, sheet=None, overrides=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".ods"):
        path = agent.convert_document(path, "xlsx")
        ext = ".xlsx"
    elif ext in (".doc", ".rtf", ".odt"):
        path = agent.convert_document(path, "docx")
        ext = ".docx"
    if ext in (".xlsx", ".xlsm"):
        return ExcelChecklist(path, sheet, overrides)
    if ext == ".docx":
        return WordChecklist(path, overrides)
    sys.exit(f"Unsupported checklist format {ext}: use an Excel or Word file")


# ---------------------------------------------------------------- documents

def display_path(ws, rel):
    """Show unpacked files as the user knows them: 'Audit.zip › planning/memo.docx' instead of
    '_extracted/Audit/planning/memo.docx' (also for zips inside zips)."""
    parts = rel.split(os.sep)
    if parts[0] != agent.EXTRACT_DIR:
        return rel.replace(os.sep, "/")
    shown, start = [], 1
    for i in range(1, len(parts) - 1):
        top_level = os.path.join(ws.root, *parts[1:i + 1]) + ".zip"
        nested = os.path.join(ws.root, *parts[:i + 1]) + ".zip"
        if os.path.isfile(top_level) or os.path.isfile(nested):
            shown.append("/".join(parts[start:i + 1]) + ".zip")
            start = i + 1
    shown.append("/".join(parts[start:]))
    return " › ".join(shown)


def index_documents(ws):
    """Extract text from every document. Returns (passages, inventory rows)."""
    passages, inventory = [], []
    files = [f for f in ws.walk(".") if ws.rel(f).split(os.sep)[0] not in OWN_DIRS]
    with Progress(TextColumn("[dim]reading files"), BarColumn(), MofNCompleteColumn(),
                  console=console, transient=True) as progress:
        for full in progress.track(files):
            rel = display_path(ws, ws.rel(full))
            ext = os.path.splitext(full)[1].lower()
            row = {"file": rel, "type": ext.lstrip(".") or "-", "size_kb": round(os.path.getsize(full) / 1024, 1)}
            if ext == ".zip":
                status = agent.archive_status(ws, full)
                row["status"] = status if status.startswith("unpacked") else f"NOT READ: zip {status}"
                inventory.append(row)
                continue
            try:
                lines = agent.file_text(full)
            except Exception as e:
                row["status"] = f"NOT READ: {e}"
                inventory.append(row)
                continue
            text_chars = sum(len(line) for line in lines)
            notes = []
            if any("[scanned page without text" in line for line in lines):
                notes.append("scanned pages not read (install tesseract-ocr)")
            elif any(line.startswith("[OCR]") for line in lines):
                notes.append("read with OCR, check quotes")
            row["status"] = "read" if text_chars > 20 else "no text found"
            if notes:
                row["status"] += "; " + "; ".join(notes)
            row["text_chars"] = text_chars
            inventory.append(row)
            passages.extend(split_passages(rel, lines))
    return passages, inventory


def split_passages(rel, lines):
    """Split a document into ~PASSAGE_CHARS passages, labelled with page / sheet / line numbers."""
    out, buf, size, where, start = [], [], 0, "", 1

    def flush(end):
        text = "\n".join(buf).strip()
        if text:
            location = f"{where}, " if where else ""
            out.append({"file": rel, "location": f"{location}lines {start}-{end}", "text": text})

    for n, line in enumerate(lines, 1):
        marker = re.match(r"----- (page|sheet) (.+) -----", line)
        if marker:
            flush(n - 1)
            buf, size, where, start = [], 0, f"{marker.group(1)} {marker.group(2)}", n + 1
            continue
        if size + len(line) > PASSAGE_CHARS and buf:
            flush(n - 1)
            buf, size, start = [], 0, n
        buf.append(line[:PASSAGE_CHARS])
        size += len(line) + 1
    flush(len(lines))
    return out


# ---------------------------------------------------------------- retrieval (BM25)

STOPWORDS = set("""the and for are was were has have had been with that this from which their there been not any
all its into such than then them they what when where who will would can could should may might does did
other also only more most some each per our your his her before after over under about between whether
including include included e.g. etc""".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'-]+", text.lower()) if len(w) > 2 and w not in STOPWORDS]


class Retriever:
    def __init__(self, passages):
        self.passages = passages
        self.docs = [Counter(tokens(p["text"] + " " + p["file"])) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(term for d in self.docs for term in d)

    def search(self, item, k=TOP_PASSAGES):
        query = Counter(tokens(item["question"] + " " + item["area"]))
        for term in item["search_terms"]:
            for t in tokens(term):
                query[t] += 2  # the curated search terms matter more than the question wording
        phrases = [t.lower() for t in item["search_terms"] if " " in t]
        n = len(self.docs)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term, weight in query.items():
                tf = doc.get(term)
                if tf:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += weight * idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.lengths[i] / self.avg))
            text = self.passages[i]["text"].lower()
            score += sum(3.0 for phrase in phrases if phrase in text)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        picked, per_file = [], Counter()
        for score, i in scored:
            p = self.passages[i]
            if per_file[p["file"]] < MAX_PER_FILE:
                picked.append(p)
                per_file[p["file"]] += 1
            if len(picked) == k:
                break
        return picked


# ---------------------------------------------------------------- assessment

SYSTEM = """You are an experienced UK audit quality reviewer carrying out a quality control (cold file) \
review of a completed audit engagement file, in the style of an ICAEW Quality Assurance Department review, \
against ISAs (UK), the FRC Ethical Standard and ISQM (UK).

You are given one requirement from the firm's QCR checklist and excerpts retrieved from the client's audit file. Judge ONLY from the \
excerpts. The excerpts are a search result, not the whole file, so be careful about absence of evidence:
- "Satisfactory": the excerpts show the work was performed, documented and concluded on adequately.
- "Finding": the excerpts show the work is missing, incomplete, inconsistent, not linked to the risks, \
not reviewed, or poorly documented. Explain the specific deficiency.
- "Not evidenced": the excerpts do not contain enough to judge either way.
- "Not applicable": the excerpts show the item does not apply to this engagement.
Severity for findings: "High" if it could mean insufficient appropriate evidence for the opinion or a \
breach of ethical/independence requirements; "Medium" for a significant documentation or procedural \
weakness; "Low" for minor matters. Use "None" when there is no finding.
Evidence quotes must be copied word for word from the excerpts (short, under 30 words) with their source \
label such as S2. Never invent facts, figures, names or dates."""

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": STATUSES},
        "severity": {"type": "string", "enum": SEVERITIES},
        "finding": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}}, "required": ["source", "quote"]}},
        "recommendation": {"type": "string"},
    },
    "required": ["status", "severity", "finding", "evidence", "recommendation"],
}


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’…]|\.\.\.", " ", text)).strip().lower()


def normalise(result, passages):
    status = result.get("status") if result.get("status") in STATUSES else "Not evidenced"
    severity = result.get("severity") if result.get("severity") in SEVERITIES else "None"
    if status != "Finding":
        severity = "None"
    elif severity == "None":
        severity = "Medium"
    evidence = []
    for ev in result.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        quote = str(ev.get("quote", "")).strip()
        match = re.search(r"\d+", str(ev.get("source", "")))
        idx = int(match.group()) - 1 if match else -1
        source = passages[idx] if 0 <= idx < len(passages) else None
        # The quote must really be in the cited passage; otherwise the model may have made it up
        verified = bool(source and quote and _norm(quote) in _norm(source["text"]))
        if not verified and quote:
            for p in passages:  # right quote, wrong label
                if _norm(quote) in _norm(p["text"]):
                    source, verified = p, True
                    break
        evidence.append({"quote": quote, "verified": verified,
                         "file": source["file"] if source else "?", "location": source["location"] if source else ""})
    attention = []
    if any(not e["verified"] for e in evidence):
        attention.append("quote not found in file")
    if status == "Satisfactory" and not any(e["verified"] for e in evidence):
        attention.append("satisfactory without verified evidence")
    return {"status": status, "severity": severity, "finding": str(result.get("finding", "")).strip(),
            "recommendation": str(result.get("recommendation", "")).strip(), "evidence": evidence,
            "attention": "; ".join(attention)}


# ---------------------------------------------------------------- filling the checklist

KEYWORD_SCHEMA = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
                  "required": ["keywords"]}


def ollama_json(model, system, prompt, schema):
    payload = {"model": model, "stream": False, "format": schema,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               "options": {"temperature": 0, "num_ctx": agent.AGENT_CTX}}
    last_error = None
    for _ in range(2):  # retry once if the model returns malformed JSON
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(10, 1800))
        if r.status_code != 200:
            try:
                raise RuntimeError(r.json().get("error", r.text))
            except ValueError:
                raise RuntimeError(r.text)
        try:
            return json.loads(r.json()["message"]["content"])
        except (ValueError, KeyError) as e:
            last_error = e
    raise RuntimeError(f"model did not return valid JSON: {last_error}")


def search_terms(item, model):
    """Checklist wording is often terse ('Has the ES been complied with?'), so ask the model for the
    words that would appear in the working papers that evidence it."""
    prompt = (f"Checklist section: {item.context or '-'}\nRequirement: {item.question}\n\n"
              "List 6 to 12 search keywords and short phrases that would appear in UK audit working papers "
              "evidencing this requirement: document names, ISA terms, synonyms and common abbreviations.")
    try:
        result = ollama_json(model, "You help search audit files. Answer in JSON.", prompt, KEYWORD_SCHEMA)
        return [str(k) for k in result.get("keywords", [])][:15]
    except RuntimeError:
        return []


def assess(item, passages, model):
    if not passages:
        return {"status": "Not evidenced", "severity": "None", "evidence": [], "recommendation": "",
                "finding": "Nothing in the client's files matched this requirement.", "attention": ""}
    excerpts = "\n\n".join(f"[S{n}] {p['file']}, {p['location']}\n{p['text']}" for n, p in enumerate(passages, 1))
    prompt = (f"Checklist section: {item.context or '-'}\nRequirement: {item.question}\n\n"
              f"Excerpts from the client's audit file:\n\n{excerpts}\n\n"
              "Assess whether the audit file meets this requirement and answer in JSON.")
    return normalise(ollama_json(model, SYSTEM, prompt, SCHEMA), passages)


def answer_text(status, options, labels):
    """The value for the answer column, respecting a dropdown list if the checklist has one."""
    if options:
        patterns = {"Satisfactory": r"^(y|yes|complied|satisfactory|ok|done|✓|✔)$",
                    "Finding": r"^(n|no|not complied|unsatisfactory|issue|x|✗|✘)$",
                    "Not applicable": r"^(n/?a|not applicable)$"}
        for option in options:
            if re.search(patterns.get(status, "^$"), option.strip().lower()):
                return option
        return None  # e.g. "Not evidenced" is not in the dropdown: leave blank, the comment explains
    return labels[status]


def comment_text(result, include_evidence):
    head = result["status"] if result["status"] != "Finding" else f"Finding ({result['severity']})"
    parts = [f"[AI: {head}] {result['finding']}"]
    if result.get("recommendation") and result["status"] == "Finding":
        parts.append(f"Recommendation: {result['recommendation']}")
    if include_evidence and result["evidence"]:
        parts.append("Evidence: " + evidence_text(result))
    if result.get("attention"):
        parts.append(f"CHECK: {result['attention']}")
    return "\n".join(parts)


def evidence_text(result):
    refs = []
    for e in result["evidence"]:
        flag = "" if e["verified"] else " [UNVERIFIED]"
        refs.append(f"{e['file']} ({e['location']}){flag}: \"{e['quote']}\"")
    return "\n".join(refs)


def fill_client(checklist_path, folder, opts):
    ws = agent.Workspace(folder)
    client = os.path.basename(ws.root.rstrip(os.sep))
    agent.CONVERTED_DIR = os.path.join(ws.root, agent.EXTRACT_DIR, ".converted")
    console.rule(f"[bold]{client}")

    checklist = open_checklist(checklist_path, opts.sheet, {
        "question": opts.question_col, "answer": opts.answer_col,
        "comment": opts.comment_col, "evidence": opts.evidence_col})
    items = checklist.items()
    for note in checklist.notes:
        console.print(f"[dim]{note}[/]")
    todo = [i for i in items if opts.overwrite or i.existing in (None, "")]
    if opts.rows:
        todo = [i for i in todo if any(re.search(rf"\b{re.escape(r)}\b", i.label) for r in opts.rows.split(","))]
    console.print(f"{len(items)} requirements found, {len(items) - len(todo)} already answered, "
                  f"{len(todo)} to check")
    if opts.dry_run:
        table = RichTable("Where", "Section", "Requirement", "Answer", "Writes to", show_lines=False)
        for i in items:
            table.add_row(i.label, i.context[:30], i.question[:70], str(i.existing or ""), ", ".join(sorted(i.roles)))
        console.print(table)
        return None
    if not todo:
        return None

    if not opts.no_extract:
        for line in agent.extract_archives(ws):
            console.print(f"[dim]{line}[/]")
    passages, inventory = index_documents(ws)
    unread = [i for i in inventory if i["status"].startswith(("NOT READ", "no text")) or "not read" in i["status"]]
    console.print(f"{len(inventory)} files, {len(passages)} passages of text")
    for i in unread:
        console.print(f"  [yellow]not read:[/] {i['file']}: {i['status']}")
    if not passages:
        console.print("[red]No readable text in this client's files, skipping.[/]")
        return None

    out_dir = os.path.join(ws.root, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "results.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    labels = dict(DEFAULT_LABELS)
    if opts.labels:
        labels.update(zip(["Satisfactory", "Finding", "Not applicable", "Not evidenced"],
                          [x.strip() for x in opts.labels.split(",")]))
    retriever = Retriever(passages)
    counts = Counter()
    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("checking", total=len(todo))
        for item in todo:
            progress.update(task, description=f"[cyan]{item.label}[/]: {item.question[:50]}…")
            base = hashlib.sha1(json.dumps([item.question, item.context, opts.model]).encode()).hexdigest()
            cached = cache.get(base, {})
            try:
                if "terms" not in cached:
                    cached["terms"] = search_terms(item, opts.model)
                hits = retriever.search({"question": item.question, "area": item.context,
                                         "search_terms": cached["terms"]})
                key = hashlib.sha1(json.dumps(hits).encode()).hexdigest()
                if opts.redo or cached.get("key") != key:
                    cached["result"], cached["key"] = assess(item, hits, opts.model), key
            except (requests.RequestException, RuntimeError) as e:
                progress.stop()
                console.print(f"[red]Ollama error on {item.label}: {e}[/]")
                break
            cache[base] = cached
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1, ensure_ascii=False)

            result = cached["result"]
            counts[result["status"] if result["status"] != "Finding" else f"Finding ({result['severity']})"] += 1
            item.write("answer", answer_text(result["status"], item.options, labels))
            item.write("comment", comment_text(result, include_evidence="evidence" not in item.roles))
            item.write("evidence", evidence_text(result) or None)
            if result["status"] == "Finding":
                progress.console.print(f"  [yellow]●[/] {item.label}: [bold]{result['severity']}[/] "
                                       f"{result['finding'][:100]}")
            progress.advance(task)

    stem = os.path.splitext(os.path.basename(checklist_path))[0]
    out = os.path.join(out_dir, f"{stem} - {client} - {datetime.date.today().isoformat()}{checklist.ext}")
    checklist.save(out)
    with open(os.path.join(out_dir, "files_reviewed.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{i['file']}\t{i['status']}\n" for i in inventory)
    console.print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    console.print(f"Completed checklist: [cyan]{out}[/]")
    return out


def main():
    parser = argparse.ArgumentParser(description="Complete a QCR checklist (Excel/Word) from a client's files")
    parser.add_argument("checklist", help="your checklist: .xlsx, .xlsm, .xls, .docx or .doc")
    parser.add_argument("folder", help="client folder (or a folder of client folders with --all)")
    parser.add_argument("--all", action="store_true", help="fill one checklist per sub-folder (client)")
    parser.add_argument("--dry-run", action="store_true", help="show the requirements and columns found, "
                                                               "without running the model")
    parser.add_argument("--sheet", help="Excel: only this sheet")
    parser.add_argument("--question-col", help="Excel: column letter of the questions (if not detected)")
    parser.add_argument("--answer-col", help="Excel: column letter for the answer (Yes/No/N/A)")
    parser.add_argument("--comment-col", help="Excel: column letter for comments")
    parser.add_argument("--evidence-col", help="Excel: column letter for the working paper reference")
    parser.add_argument("--labels", help="answer words for satisfactory,finding,n/a,not evidenced "
                                         "(default: Yes,No,N/A,Not evidenced)")
    parser.add_argument("--rows", help="only these rows, e.g. '12,15' or 'table 2 row 4'")
    parser.add_argument("--overwrite", action="store_true", help="also answer rows that already have an answer")
    parser.add_argument("--redo", action="store_true", help="ignore cached results and assess again")
    parser.add_argument("--model", default=QCR_MODEL, help=f"Ollama model (default: {QCR_MODEL})")
    parser.add_argument("--no-extract", action="store_true", help="don't unpack zip files")
    opts = parser.parse_args()

    checklist = os.path.realpath(os.path.expanduser(opts.checklist))
    folder = os.path.expanduser(opts.folder)
    if not os.path.isfile(checklist):
        sys.exit(f"Checklist not found: {opts.checklist}")
    if not os.path.isdir(folder):
        sys.exit(f"Not a folder: {opts.folder}")
    if not opts.dry_run and not agent.check_model(opts.model):
        sys.exit(1)

    if not opts.all:
        fill_client(checklist, folder, opts)
        return
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path) and not name.startswith((".", "_")):
            fill_client(checklist, path, opts)


if __name__ == "__main__":
    main()
