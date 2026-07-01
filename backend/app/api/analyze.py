from fastapi import HTTPException
from .agents.csv_agent import analyze_csv
from .agents.agentic_rag_retrieval import analyze_rag


async def run_csv_analysis(llm, doc, session_id: str, question: str):
    try:
        if doc["type"] != "CSV":
            raise HTTPException(400, detail="Only CSV files can be analyzed with the CSV agent")
        answer = await analyze_csv(llm, doc["path"], question, session_id)
        return {"document_id": doc["id"], "session_id": session_id, "type": doc["type"], "question": question, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"analysis failed: {e}") from e


async def run_rag_analysis(llm, doc, embed_model, question: str, session_id: str):
    try:
        if doc["type"] not in ("PDF", "Text"):
            raise HTTPException(400, detail="RAG analysis supports PDF and Text documents only")
        answer = await analyze_rag(llm, doc["id"], embed_model, question, session_id)
        return {"document_id": doc["id"], "session_id": session_id, "type": doc["type"], "question": question, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"rag analysis failed: {e}") from e
