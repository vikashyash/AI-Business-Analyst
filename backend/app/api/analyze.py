from fastapi import HTTPException
from .store import get_document
from .agents.csv_agent import analyze_csv

def run_csv_analysis(doc_id: str, session_id: str, question: str):
    doc = get_document(doc_id)
    if doc["type"] != "CSV": raise HTTPException(400, detail="Only CSV files can be analyzed in this phase")
    answer = analyze_csv(doc["path"], question, session_id)
    return {"document_id": doc_id, "session_id": session_id, "type": doc["type"], "question": question, "answer": answer}
