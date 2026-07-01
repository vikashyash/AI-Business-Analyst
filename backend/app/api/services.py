import asyncio
from pathlib import Path
import re
import pdfplumber
import pandas as pd

_TYPES = {".pdf": "PDF", ".csv": "CSV", ".txt": "Text", ".text": "Text"}

def detect_file_type(path):
    try:
        return _TYPES.get(Path(path).suffix.lower(), "Text")
    except Exception as e:
        raise RuntimeError(f"detect_file_type failed: {e}") from e

def _parse_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(filter(None, (p.extract_text() for p in pdf.pages)))
    except Exception as e:
        raise RuntimeError(f"parse_pdf failed: {e}") from e

async def parse_pdf(path):
    return await asyncio.to_thread(_parse_pdf, path)

def _parse_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"parse_csv failed: {e}") from e

async def parse_csv(path):
    return await asyncio.to_thread(_parse_csv, path)

def _parse_txt(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"parse_txt failed: {e}") from e

async def parse_txt(path):
    return await asyncio.to_thread(_parse_txt, path)

def clean_text(text):
    try:
        return re.sub(r"\s+", " ", text or "").strip()
    except Exception as e:
        raise RuntimeError(f"clean_text failed: {e}") from e

async def parse_file(path):
    try:
        path, kind = str(path), detect_file_type(path)
        if kind == "PDF":
            return kind, clean_text(await parse_pdf(path))
        if kind == "CSV":
            return kind, await parse_csv(path)
        return kind, clean_text(await parse_txt(path))
    except Exception as e:
        raise RuntimeError(f"parse_file failed: {e}") from e

def is_supported(path):
    try:
        return Path(path).suffix.lower() in _TYPES
    except Exception as e:
        raise RuntimeError(f"is_supported failed: {e}") from e
