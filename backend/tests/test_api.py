import io
from pathlib import Path
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_upload_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.store.UPLOAD_DIR", tmp_path)
    files = {"file": ("sales.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")}
    res = client.post("/upload", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["type"] == "CSV" and "id" in body
    docs = client.get("/documents").json()["documents"]
    assert any(d["id"] == body["id"] for d in docs)

def test_bad_type():
    files = {"file": ("x.doc", io.BytesIO(b"data"), "application/octet-stream")}
    assert client.post("/upload", files=files).status_code == 400
