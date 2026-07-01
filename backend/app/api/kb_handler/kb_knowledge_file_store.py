import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import UploadFile, HTTPException
from ...config import UPLOAD_DIR, MAX_BYTES
from ..services import detect_file_type, is_supported
from .kb_insertion import process_file_for_insertion

def _manifest():
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        p = UPLOAD_DIR / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception as e:
        raise RuntimeError(f"manifest read failed: {e}") from e

def _write(data):
    try:
        (UPLOAD_DIR / "manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"manifest write failed: {e}") from e

async def save_upload(file: UploadFile,embed_model=None,workspace_id=None):
    try:
        name = file.filename or ""
        if not is_supported(name):
            raise HTTPException(400, detail="Unsupported file type. Use pdf, csv, or txt.")
        body = await file.read()
        if len(body) > MAX_BYTES:
            raise HTTPException(413, detail="File too large")

        doc_id, ext = str(uuid.uuid4()), Path(name).suffix.lower()
        path = UPLOAD_DIR / f"{doc_id}{ext}"

        def _persist():
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            row = {
                "id": doc_id,
                "filename": name,
                "type": detect_file_type(path),
                "path": str(path),
                "size": len(body),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_id":workspace_id
            }
            data = _manifest()
            data[doc_id] = row
            _write(data)
            return row

        row = await asyncio.to_thread(_persist)
        file_type = detect_file_type(path)
        if file_type in ("PDF", "Text"):
            kb_status = await process_file_for_insertion(model=embed_model,file=str(path), file_name=name, doc_id=doc_id)
            row["kb_status"] = kb_status
        return row
    except Exception as e:
        raise HTTPException(500, detail=f"upload failed: {e}") from e

async def list_documents():
    try:
        return await asyncio.to_thread(lambda: list(_manifest().values()))
    except Exception as e:
        raise HTTPException(500, detail=f"list documents failed: {e}") from e

async def get_document(doc_id):
    try:
        def _get():
            data = _manifest()
            if doc_id not in data:
                raise HTTPException(404, detail="Document not found")
            doc = data[doc_id]
            if "path" not in doc:
                hits = list(UPLOAD_DIR.glob(f"{doc_id}.*"))
                if not hits:
                    raise HTTPException(404, detail="File missing on disk")
                doc["path"] = str(hits[0])
            return doc

        return await asyncio.to_thread(_get)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"get document failed: {e}") from e
