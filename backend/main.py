from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.api.kb_handler.kb_knowledge_file_store import save_upload, list_documents,get_document
from app.api.analyze import run_csv_analysis, run_rag_analysis
import uvicorn,os
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from app.api.workspaces.workspace_store import(create_workspace,list_workspaces,get_workspace,attach_document)
from app.api.workspaces.models import CreateWorkspaceBody,WorkspaceAnalyzeBody,TaskBody,RunBody # pydantic class
from app.api.agents.orchestrator import run_orchestrator
from app.api.tasks.run_tasks import run_ba_task

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

app = FastAPI(title="AI Business Analyst")

def _llm():
    try:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("MODEL_API_KEY")
        return ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"), google_api_key=key, temperature=0.2)
    except Exception as e:
        raise RuntimeError(f"llm init failed: {e}") from e

class AnalyzeBody(BaseModel):
    document_id: str
    session_id: str
    question: str

# reusable function for both chat and workspace that retrieves from rag, analyse from csv.
async def _analyze_document(doc: dict, session_id: str, question: str, workspace_id: str = None):
    try:
        llm=_llm()
        if doc["type"]=="CSV":
            return await run_csv_analysis(llm, doc, session_id, question)
        elif doc["type"] in ("PDF","Text"):
            result = await run_rag_analysis(llm, doc, model, question, session_id)
        else:
            raise HTTPException(400, detail=f"Unsupported document type: {doc['type']}")

        if workspace_id:
            result["workspace_id"]=workspace_id
        return result
    except Exception as e:
        print(str(e))
        raise HTTPException(500, detail=str(e))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        return await save_upload(file,embed_model=model)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/documents")
async def documents():
    try:
        return {"documents": await list_documents()}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# chat flow
@app.post("/analyze") 
async def analyze(body: AnalyzeBody):
    try:
        doc = await get_document(body.document_id)
        return await _analyze_document(doc, body.session_id, body.question)
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
# workspace api (upload doc and ask qn)
@app.post("/create_workspace")
async def create_ws(body:CreateWorkspaceBody):
    try:
        return await create_workspace(name=body.name,description=body.description)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/get_all_workspaces")
async def get_all_workspaces():
    try:
        return {"workspaces": await list_workspaces()}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
    
@app.get("/get_workspaces/{ws_id}")
async def get_ws(ws_id:str):
    try:
        return {"workspaces":await get_workspace(ws_id)}
    except Exception as e:
        raise HTTPException(500, detail=str(e)) 

@app.post("/workspaces/{ws_id}/upload")
async def ws_upload(ws_id: str, file: UploadFile = File(...)):
    try:
        await get_workspace(ws_id)
        row = await save_upload(file, embed_model=model, workspace_id=ws_id)
        await attach_document(ws_id, row["id"])
        return row
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
# allows qn and answer flow in workspace.
@app.post("/workspaces/{ws_id}/analyze")
async def ws_analyze(ws_id: str, body: WorkspaceAnalyzeBody):
    """Chat inside a workspace — same agents as /analyze."""
    try:
        ws = await get_workspace(ws_id)
        doc = await get_document(body.document_id)
        if body.document_id not in ws.get("document_ids", []):
            raise HTTPException(400, detail="Document not attached to this workspace")
        return await _analyze_document(
            doc, body.session_id, body.question, workspace_id=ws_id
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
@app.post("/workspaces/{ws_id}/tasks")
async def ws_task(ws_id: str, body: TaskBody):
    """BA mode — run a structured task agent."""
    try:
        llm = _llm()
        return await run_ba_task(llm, model, ws_id, body)
    except HTTPException:
        raise
    except Exception as e:
        print(str(e),"exception..")
        raise HTTPException(500, detail=str(e))


@app.post("/workspaces/{ws_id}/run")
async def ws_run(ws_id: str, body: RunBody):
    """BA orchestrator — plan and run multiple tasks for a goal."""
    try:
        llm = _llm()
        return await run_orchestrator(
            llm=llm,
            embed_model=model,
            ws_id=ws_id,
            goal=body.goal,
            session_id=body.session_id,
            document_ids=body.document_ids,
            params=body.params or {},
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    try:
        uvicorn.run(app="main:app", host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"server failed: {e}")
