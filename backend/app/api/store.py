import json, uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import UploadFile, HTTPException
from ..config import UPLOAD_DIR, MAX_BYTES
from .services import detect_file_type, is_supported

def _manifest():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def _write(data):
    (UPLOAD_DIR / "manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

async def save_upload(file: UploadFile):
    name = file.filename or ""
    if not is_supported(name): raise HTTPException(400, detail="Unsupported file type. Use pdf, csv, or txt.")
    body = await file.read()
    data = _manifest()
    if len(body) > MAX_BYTES: raise HTTPException(413, detail="File too large")
    doc_id, ext = str(uuid.uuid4()), Path(name).suffix.lower()
    path = UPLOAD_DIR / f"{doc_id}{ext}"
    path.write_bytes(body)
    row = {"id": doc_id, "filename": name, "type": detect_file_type(path), "path": str(path), "size": len(body), "created_at": datetime.now(timezone.utc).isoformat()}
    data[doc_id] = row
    _write(data)
    return row

def list_documents():
    return list(_manifest().values())

def get_document(doc_id):
    data = _manifest()
    if doc_id not in data: raise HTTPException(404, detail="Document not found")
    doc = data[doc_id]
    if "path" not in doc:
        hits = list(UPLOAD_DIR.glob(f"{doc_id}.*"))
        if not hits: raise HTTPException(404, detail="File missing on disk")
        doc["path"] = str(hits[0])
    return doc
