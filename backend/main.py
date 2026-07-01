from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.api.kb_handler.kb_knowledge_file_store import save_upload, list_documents,get_document
from app.api.analyze import run_csv_analysis, run_rag_analysis
import uvicorn,os
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from app.api.workspaces.workspace_store import(create_workspace,list_workspaces,get_workspace,attach_document)
from app.api.workspaces.models import CreateWorkspaceBody,WorkspaceAnalyzeBody,TaskBody
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

app = FastAPI(title="AI Business Analyst")

def _llm():
    try:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("MODEL_API_KEY")
        return ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"), google_api_key=key, temperature=0)
    except Exception as e:
        raise RuntimeError(f"llm init failed: {e}") from e

class AnalyzeBody(BaseModel):
    document_id: str
    session_id: str
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        return await save_upload(file,embed_model=model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/documents")
async def documents():
    try:
        return {"documents": await list_documents()}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/analyze")
async def analyze(body: AnalyzeBody):
    try:
        doc = await get_document(body.document_id)
        llm = _llm()
        if doc["type"] == "CSV":
            return await run_csv_analysis(llm, doc, body.session_id, body.question)
        if doc["type"] in ("PDF", "Text"):
            return await run_rag_analysis(llm, doc, model, body.question, body.session_id)
        raise HTTPException(400, detail=f"Unsupported document type: {doc['type']}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    


    

    

    



if __name__ == "__main__":
    try:
        uvicorn.run(app="main:app", host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"server failed: {e}")
