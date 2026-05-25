from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.api.store import save_upload, list_documents
from app.api.analyze import run_csv_analysis
import uvicorn

app = FastAPI(title="AI Business Analyst")

class AnalyzeBody(BaseModel):
    document_id: str
    session_id: str
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await save_upload(file)

@app.get("/documents")
def documents():
    return {"documents": list_documents()}

@app.post("/analyze")
def analyze(body: AnalyzeBody):
    return run_csv_analysis(body.document_id, body.session_id, body.question)


if __name__ == "__main__":
    uvicorn.run(app="main:app",host="0.0.0.0",port=8080)

