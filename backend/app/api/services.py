from pathlib import Path
import re
import pdfplumber
import pandas as pd

_TYPES = {".pdf": "PDF", ".csv": "CSV", ".txt": "Text", ".text": "Text"}

def detect_file_type(path):
    return _TYPES.get(Path(path).suffix.lower(), "Text")

def parse_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(filter(None, (p.extract_text() for p in pdf.pages)))

def parse_csv(path):
    return pd.read_csv(path)

def parse_txt(path):
    return Path(path).read_text(encoding="utf-8")

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def parse_file(path):
    path, kind = str(path), detect_file_type(path)
    if kind == "PDF": return kind, clean_text(parse_pdf(path))
    if kind == "CSV": return kind, parse_csv(path)
    return kind, clean_text(parse_txt(path))

def is_supported(path):
    return Path(path).suffix.lower() in _TYPES
