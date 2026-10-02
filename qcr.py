"""Quality control review (cold file review) of audit engagement files, run fully locally.

For each client folder (zip, Word, Excel and PDF files from the audit file) this:
  1. unpacks zips and extracts the text of every document (OCR for scanned PDFs),
  2. for each item of the checklist, finds the most relevant passages and asks the local Ollama model
     for a structured assessment with quoted evidence, then checks every quote against the files,
  3. writes an Excel workbook and a draft Word report into <client folder>/_qcr/.

Usage:
    python qcr.py ~/Clients/ClientA                    # review one client
    python qcr.py ~/Clients --all                      # review every sub-folder, plus a portfolio summary
    python qcr.py ~/Clients/ClientA --items 8,29,37    # (re)run selected checklist items only
    python qcr.py ~/Clients/ClientA --checklist my_firm_checklist.xlsx

The output is a draft for a qualified reviewer: every conclusion must be checked against the file.
"""
import argparse
import csv
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

import agent
from utils.ollama_client import OLLAMA_URL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKLIST = os.path.join(BASE_DIR, "qcr_checklist.csv")
QCR_MODEL = os.environ.get("QCR_MODEL", agent.AGENT_MODEL)
OUTPUT_DIR = "_qcr"
OWN_DIRS = {OUTPUT_DIR, "_agent_logs"}   # tool output, never part of the audit file
PASSAGE_CHARS = 1200   # size of the text passages the model sees
TOP_PASSAGES = 8       # passages per checklist item
MAX_PER_FILE = 3       # so one long document does not crowd out the others
STATUSES = ["Satisfactory", "Finding", "Not evidenced", "Not applicable"]
SEVERITIES = ["High", "Medium", "Low", "None"]

console = Console()


# ---------------------------------------------------------------- checklist

def load_checklist(path):
    """Read the checklist from CSV or Excel. Needs a 'question' column; 'id', 'area', 'reference' and
    'search_terms' (separated by ';') are optional, so a firm's own checklist can be dropped in."""
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        import pandas as pd
        rows = pd.read_excel(path).fillna("").astype(str).to_dict("records")
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    items = []
    for n, row in enumerate(rows, 1):
        row = {k.strip().lower().replace(" ", "_"): (v or "").strip() for k, v in row.items() if k}
        if not row.get("question"):
            continue
        items.append({
            "id": row.get("id") or str(n),
            "area": row.get("area", ""),
            "reference": row.get("reference", ""),
            "question": row["question"],
            "search_terms": [t.strip() for t in re.split(r"[;\n]", row.get("search_terms", "")) if t.strip()],
        })
    if not items:
        sys.exit(f"No checklist items with a 'question' column found in {path}")
    return items


def select_items(items, spec):
    if not spec:
        return items
    wanted = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            wanted.add(part.strip())
    return [item for item in items if item["id"] in wanted]


# ---------------------------------------------------------------- documents

def index_documents(ws):
    """Extract text from every document. Returns (passages, inventory rows)."""
    passages, inventory = [], []
    files = [f for f in ws.walk(".") if ws.rel(f).split(os.sep)[0] not in OWN_DIRS]
    with Progress(TextColumn("[dim]reading files"), BarColumn(), MofNCompleteColumn(),
                  console=console, transient=True) as progress:
        for full in progress.track(files):
            rel = ws.rel(full)
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

You are given one checklist item and excerpts retrieved from the client's audit file. Judge ONLY from the \
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


def assess(item, passages, model):
    if not passages:
        return {"status": "Not evidenced", "severity": "None", "evidence": [], "recommendation": "",
                "finding": "No passages in the audit file matched this checklist item.", "attention": ""}
    excerpts = "\n\n".join(f"[S{n}] {p['file']}, {p['location']}\n{p['text']}" for n, p in enumerate(passages, 1))
    prompt = (f"Checklist item {item['id']} ({item['reference']}), area: {item['area']}\n"
              f"Question: {item['question']}\n\nExcerpts from the audit file:\n\n{excerpts}\n\n"
              "Assess this checklist item and answer in JSON.")
    payload = {"model": model, "stream": False, "format": SCHEMA,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
               "options": {"temperature": 0, "num_ctx": agent.AGENT_CTX}}
    last_error = None
    for _ in range(2):  # retry once if the model returns malformed JSON
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=(10, 1800))
        if r.status_code != 200:
            raise RuntimeError(r.json().get("error", r.text) if r.headers.get("content-type", "").startswith(
                "application/json") else r.text)
        try:
            result = json.loads(r.json()["message"]["content"])
            break
        except (ValueError, KeyError) as e:
            last_error = e
    else:
        raise RuntimeError(f"model did not return valid JSON: {last_error}")
    return normalise(result, passages)


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


def indicative_grade(results):
    """A rough indication only, to help prioritise; the reviewer decides the actual grade."""
    sev = Counter(r["severity"] for r in results if r["status"] == "Finding")
    missing = sum(r["status"] == "Not evidenced" for r in results)
    if sev["High"]:  # a high finding stands even if other items could not be assessed
        return "Significant improvement required"
    if missing > len(results) / 4:
        return f"Not graded: {missing} of {len(results)} items not evidenced"
    if sev["Medium"] >= 3:
        return "Improvement required"
    if sev["Medium"] or sev["Low"] >= 3:
        return "Generally acceptable"
    return "Good"


# ---------------------------------------------------------------- outputs

FILLS = {"Finding:High": "FFC7CE", "Finding:Medium": "FFEB9C", "Finding:Low": "FFF2CC",
         "Satisfactory": "C6EFCE", "Not evidenced": "D9D9D9", "Not applicable": "EDEDED"}


def write_workbook(path, client, results, inventory, grade, model):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    bold, wrap = Font(bold=True), Alignment(wrap_text=True, vertical="top")

    def sheet(ws, header, rows, widths):
        ws.append(header)
        for cell in ws[1]:
            cell.font = bold
            cell.fill = PatternFill("solid", fgColor="DDEBF7")
        for row in rows:
            ws.append(row)
        for col, width in zip("ABCDEFGHIJKLMNOP", widths):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = wrap
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    summary = wb.active
    summary.title = "Summary"
    counts = Counter(r["status"] if r["status"] != "Finding" else f"Finding ({r['severity']})" for r in results)
    rows = [("Client", client), ("Review date", datetime.date.today().isoformat()), ("Model", model),
            ("Indicative grade", grade), ("Checklist items", len(results))]
    rows += [(label, counts.get(label, 0)) for label in
             ["Finding (High)", "Finding (Medium)", "Finding (Low)", "Satisfactory", "Not evidenced", "Not applicable"]]
    rows += [("Files not read", sum(1 for i in inventory if i["status"].startswith(("NOT READ", "no text")))),
             ("Items needing attention", sum(1 for r in results if r["attention"])),
             ("", ""),
             ("Important", "Draft produced by a local AI model from text retrieved from the audit file. "
                           "'Not evidenced' means no relevant text was found, not that the work was not done. "
                           "A qualified reviewer must verify every conclusion and decide the grade.")]
    for row in rows:
        summary.append(row)
    for cell in summary["A"]:
        cell.font = bold
    summary.column_dimensions["A"].width = 26
    summary.column_dimensions["B"].width = 100
    for row in summary.iter_rows():
        for cell in row:
            cell.alignment = wrap

    ws = wb.create_sheet("Checklist")
    sheet(ws, ["ID", "Area", "Reference", "Question", "Status", "Severity", "Finding", "Evidence",
               "Recommendation", "Needs attention", "Reviewer conclusion", "Reviewer comments"],
          [[r["id"], r["area"], r["reference"], r["question"], r["status"], r["severity"], r["finding"],
            "\n".join(f"{'' if e['verified'] else '[UNVERIFIED] '}\"{e['quote']}\" ({e['file']}, {e['location']})"
                      for e in r["evidence"]),
            r["recommendation"], r["attention"], "", ""] for r in results],
          [6, 18, 20, 50, 15, 10, 60, 70, 45, 22, 20, 40])
    for row in ws.iter_rows(min_row=2):
        status, severity = row[4].value, row[5].value
        color = FILLS.get(f"{status}:{severity}") or FILLS.get(status)
        if color:
            row[4].fill = row[5].fill = PatternFill("solid", fgColor=color)

    ws = wb.create_sheet("Evidence")
    sheet(ws, ["ID", "Area", "File", "Location", "Quote", "Verified in file"],
          [[r["id"], r["area"], e["file"], e["location"], e["quote"], "Yes" if e["verified"] else "NO"]
           for r in results for e in r["evidence"]], [6, 18, 45, 25, 80, 14])

    ws = wb.create_sheet("Files reviewed")
    sheet(ws, ["File", "Type", "Size (KB)", "Characters of text", "Status"],
          [[i["file"], i["type"], i["size_kb"], i.get("text_chars", ""), i["status"]] for i in inventory],
          [70, 8, 10, 18, 60])
    wb.save(path)


def write_report(path, client, results, inventory, grade, model):
    import docx
    from docx.shared import Pt

    doc = docx.Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading(f"Quality control review: {client}", 0)
    doc.add_paragraph(f"Review date: {datetime.date.today():%d %B %Y}    Model: {model}")
    note = doc.add_paragraph()
    note.add_run("Draft for the reviewer. ").bold = True
    note.add_run("Produced by a local AI model from text retrieved from the audit file. Every finding must be "
                 "verified against the file, and items marked 'Not evidenced' checked manually.")

    doc.add_heading("Summary", 1)
    doc.add_paragraph(f"Indicative grade: {grade}")
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Outcome", "Items"
    counts = Counter(r["status"] if r["status"] != "Finding" else f"Finding ({r['severity']})" for r in results)
    for label in ["Finding (High)", "Finding (Medium)", "Finding (Low)", "Satisfactory", "Not evidenced",
                  "Not applicable"]:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = label, str(counts.get(label, 0))

    order = {"High": 0, "Medium": 1, "Low": 2}
    findings = sorted((r for r in results if r["status"] == "Finding"), key=lambda r: order.get(r["severity"], 3))
    doc.add_heading("Findings", 1)
    if not findings:
        doc.add_paragraph("No findings were raised.")
    for r in findings:
        doc.add_heading(f"{r['id']}. {r['area']}: {r['severity']} ({r['reference']})", 2)
        doc.add_paragraph(r["question"]).italic = True
        doc.add_paragraph(r["finding"])
        for e in r["evidence"]:
            flag = "" if e["verified"] else " [quote not found in file, check]"
            doc.add_paragraph(f"“{e['quote']}” ({e['file']}, {e['location']}){flag}", style="List Bullet")
        if r["recommendation"]:
            p = doc.add_paragraph()
            p.add_run("Recommendation: ").bold = True
            p.add_run(r["recommendation"])

    missing = [r for r in results if r["status"] == "Not evidenced"]
    if missing:
        doc.add_heading("Not evidenced in the documents reviewed", 1)
        doc.add_paragraph("No relevant text was found for these items. Check whether the work is on the file "
                          "(for example in documents that could not be read) before raising a finding.")
        for r in missing:
            doc.add_paragraph(f"{r['id']}. {r['area']} ({r['reference']}): {r['question']}", style="List Bullet")

    unread = [i for i in inventory if i["status"].startswith(("NOT READ", "no text"))
              or "not read" in i["status"]]
    if unread:
        doc.add_heading("Files that could not be fully read", 1)
        for i in unread:
            doc.add_paragraph(f"{i['file']}: {i['status']}", style="List Bullet")
    doc.save(path)


def write_portfolio(path, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "Portfolio"
    header = ["Client", "Indicative grade", "High", "Medium", "Low", "Not evidenced", "Files not read", "Workbook"]
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in rows:
        ws.append([row.get(h, "") for h in header])
    for col, width in zip("ABCDEFGH", [30, 32, 8, 8, 8, 14, 14, 80]):
        ws.column_dimensions[col].width = width
    wb.save(path)


# ---------------------------------------------------------------- driver

def review_client(folder, items, model, redo=False, extract=True, only=None):
    """Review one client. With `only` (a set of item ids), just those items are assessed again and the
    other items are taken from the previous run, so the report stays complete."""
    ws = agent.Workspace(folder)
    client = os.path.basename(ws.root.rstrip(os.sep))
    agent.CONVERTED_DIR = os.path.join(ws.root, agent.EXTRACT_DIR, ".converted")
    out_dir = os.path.join(ws.root, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    console.rule(f"[bold]{client}")

    if extract:
        for line in agent.extract_archives(ws):
            console.print(f"[dim]{line}[/]")
    passages, inventory = index_documents(ws)
    unread = [i for i in inventory if i["status"].startswith(("NOT READ", "no text"))]
    console.print(f"{len(inventory)} files, {len(passages)} passages of text"
                  + (f", [yellow]{len(unread)} files could not be read[/]" if unread else ""))
    if not passages:
        console.print("[red]No readable text found, skipping this client.[/]")
        return None

    # Results are cached per item so an interrupted review resumes where it stopped
    cache_path = os.path.join(out_dir, "results.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    retriever = Retriever(passages)
    results = []
    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("reviewing", total=len(items))
        for item in items:
            progress.update(task, description=f"[cyan]{item['id']}. {item['area']}[/]: {item['question'][:50]}…")
            cached = cache.get(item["id"])
            if only and item["id"] not in only:
                if cached:
                    results.append({**{k: item[k] for k in ("id", "area", "reference", "question")},
                                    **cached["result"]})
                progress.advance(task)
                continue
            hits = retriever.search(item)
            key = hashlib.sha1(json.dumps([item, hits, model], sort_keys=True).encode()).hexdigest()
            if cached and cached.get("key") == key and not (redo or only):
                result = cached["result"]
            else:
                try:
                    result = assess(item, hits, model)
                except (requests.RequestException, RuntimeError) as e:
                    progress.stop()
                    console.print(f"[red]Ollama error on item {item['id']}: {e}[/]")
                    raise SystemExit(1)
                cache[item["id"]] = {"key": key, "result": result}
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=1, ensure_ascii=False)
            results.append({**{k: item[k] for k in ("id", "area", "reference", "question")}, **result})
            if result["status"] == "Finding":
                progress.console.print(f"  [yellow]●[/] {item['id']}. {item['area']}: "
                                       f"[bold]{result['severity']}[/] {result['finding'][:110]}")
            progress.advance(task)

    grade = indicative_grade(results)
    stamp = datetime.date.today().isoformat()
    xlsx = os.path.join(out_dir, f"QCR_{client}_{stamp}.xlsx")
    report = os.path.join(out_dir, f"QCR_{client}_{stamp}.docx")
    write_workbook(xlsx, client, results, inventory, grade, model)
    write_report(report, client, results, inventory, grade, model)
    sev = Counter(r["severity"] for r in results if r["status"] == "Finding")
    console.print(f"Indicative grade: [bold]{grade}[/]  (findings: {sev['High']} high, {sev['Medium']} medium, "
                  f"{sev['Low']} low)\nWrote [cyan]{xlsx}[/]\n  and [cyan]{report}[/]")
    return {"Client": client, "Indicative grade": grade, "High": sev["High"], "Medium": sev["Medium"],
            "Low": sev["Low"], "Not evidenced": sum(r["status"] == "Not evidenced" for r in results),
            "Files not read": len(unread), "Workbook": xlsx}


def main():
    parser = argparse.ArgumentParser(description="Local quality control review of audit engagement files")
    parser.add_argument("folder", help="client folder (or a folder of client folders with --all)")
    parser.add_argument("--all", action="store_true", help="review every sub-folder as a separate client")
    parser.add_argument("--checklist", default=DEFAULT_CHECKLIST, help="checklist CSV or Excel file")
    parser.add_argument("--items", help="re-assess only these checklist ids, e.g. 1-5,8,29 "
                                         "(other items keep their previous results)")
    parser.add_argument("--model", default=QCR_MODEL, help=f"Ollama model (default: {QCR_MODEL})")
    parser.add_argument("--redo", action="store_true", help="ignore cached results and assess again")
    parser.add_argument("--no-extract", action="store_true", help="don't unpack zip files")
    opts = parser.parse_args()

    folder = os.path.expanduser(opts.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Not a folder: {opts.folder}")
    if not agent.check_model(opts.model):
        sys.exit(1)
    items = load_checklist(opts.checklist)
    only = {item["id"] for item in select_items(items, opts.items)} if opts.items else None
    console.print(f"Checklist: {len(items)} items from {opts.checklist}"
                  + (f", re-assessing {len(only)}" if only else ""))

    if not opts.all:
        review_client(folder, items, opts.model, opts.redo, not opts.no_extract, only)
        return
    clients = sorted(d for d in os.listdir(folder)
                     if os.path.isdir(os.path.join(folder, d)) and not d.startswith((".", "_")))
    rows = []
    for name in clients:
        row = review_client(os.path.join(folder, name), items, opts.model, opts.redo, not opts.no_extract, only)
        if row:
            rows.append(row)
    if rows:
        path = os.path.join(folder, f"QCR_portfolio_{datetime.date.today().isoformat()}.xlsx")
        write_portfolio(path, rows)
        console.print(f"\nPortfolio summary: [cyan]{path}[/]")


if __name__ == "__main__":
    main()
