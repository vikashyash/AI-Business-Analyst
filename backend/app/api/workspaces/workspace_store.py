import json,uuid,asyncio
from datetime import datetime,timezone
from pathlib import Path
from fastapi import HTTPException
from ...config import WORKSPACE_DIR

def _index_path():
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR / "index.json"

def _read_index():
    p = _index_path()
    return json.loads(p.read_text()) if p.exists() else {}

def _write_index(data):
    _index_path().write_text(json.dumps(data, indent=2))

def _workspace_dir(ws_id):
    d = WORKSPACE_DIR / ws_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "deliverables").mkdir(exist_ok=True)
    return d

async def create_workspace(name: str, description: str = ""):
    def _create():
        ws_id = str(uuid.uuid4())
        ws = {
            "id": ws_id,
            "name": name,
            "description": description,
            "document_ids": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        data = _read_index()
        data[ws_id] = ws
        _write_index(data)
        _workspace_dir(ws_id)
        return ws
    return await asyncio.to_thread(_create)

async def get_workspace(ws_id: str):
    def _get():
        data = _read_index()
        if ws_id not in data:
            raise HTTPException(404, detail="Workspace not found")
        return data[ws_id]
    return await asyncio.to_thread(_get)

async def list_workspaces():
    return await asyncio.to_thread(lambda: list(_read_index().values()))

async def attach_document(ws_id: str, doc_id: str):
    def _attach():
        data = _read_index()
        if ws_id not in data:
            raise HTTPException(404, detail="Workspace not found")
        if doc_id not in data[ws_id]["document_ids"]:
            data[ws_id]["document_ids"].append(doc_id)
        _write_index(data)
        return data[ws_id]
    return await asyncio.to_thread(_attach)

async def save_deliverable(ws_id: str, task_type: str, payload: dict):
    def _save():
        d = _workspace_dir(ws_id) / "deliverables"
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = d / f"{task_type}_{ts}.json"
        path.write_text(json.dumps(payload, indent=2))
        return str(path)
    return await asyncio.to_thread(_save)