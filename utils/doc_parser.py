try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:
    import fitz  # older PyMuPDF releases

def parse_document(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
