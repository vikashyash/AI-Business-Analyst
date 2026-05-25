import io
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze_csv(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.store.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("app.api.analyze.analyze_csv", lambda path, q, sid: f"answer:{q}")
    files = {"file": ("sales.csv", io.BytesIO(b"name,amount\nA,10\n"), "text/csv")}
    doc = client.post("/upload", files=files).json()
    body = {"document_id": doc["id"], "session_id": "sess-1", "question": "total amount?"}
    res = client.post("/analyze", json=body)
    assert res.status_code == 200
    assert res.json()["answer"] == "answer:total amount?"

def test_analyze_rejects_txt(monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.store.UPLOAD_DIR", tmp_path)
    files = {"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
    doc = client.post("/upload", files=files).json()
    res = client.post("/analyze", json={"document_id": doc["id"], "session_id": "s1", "question": "sum"})
    assert res.status_code == 400
