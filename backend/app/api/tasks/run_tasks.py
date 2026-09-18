from fastapi import HTTPException
from ..workspaces.workspace_store import get_workspace, save_deliverable
from ..kb_handler.kb_knowledge_file_store import get_document
from ..agents import (
    requirements_agent,
    summarize_agent,
    risks_agent,
    profile_agent,
    deliverable_agent
)

HANDLERS = {
    "extract_requirements": requirements_agent.run,
    "summarize": summarize_agent.run,
    "find_risks": risks_agent.run,
    "profile_data": profile_agent.run,
    "generate_brd":deliverable_agent.run
}

# Which document types each task needs at least one of
TASK_DOC_REQUIREMENTS = {
    "extract_requirements": ("PDF", "Text"),
    "summarize": ("PDF", "Text", "CSV"),
    "find_risks": ("PDF", "Text"),
    "profile_data": ("CSV",),
    "generate_brd":("PDF","Text")
}


async def run_ba_task(llm, embed_model, ws_id, body) -> dict:
    try:
        if body.task_type not in HANDLERS:
            raise HTTPException(
                400,
                detail=f"Unknown task: {body.task_type}. Available: {list(HANDLERS.keys())}",
            )

        ws = await get_workspace(ws_id)

        if not body.document_ids:
            raise HTTPException(400, detail="document_ids cannot be empty")

        docs = []
        for doc_id in body.document_ids:
            if doc_id not in ws.get("document_ids", []):
                raise HTTPException(400, detail=f"Document {doc_id} is not in this workspace")
            docs.append(await get_document(doc_id))

        required_types = TASK_DOC_REQUIREMENTS[body.task_type] # file extension verification
        if not any(d["type"] in required_types for d in docs):
            raise HTTPException(
                400,
                detail=f"{body.task_type} needs at least one document of type: {', '.join(required_types)}",
            )

        result = await HANDLERS[body.task_type](
            llm,
            embed_model,
            docs,
            body.params or {},
            body.session_id,
        )

        saved_as = await save_deliverable(ws_id, body.task_type, result)
        return {
            "workspace_id": ws_id,
            "task_type": body.task_type,
            "session_id": body.session_id,
            "document_ids": body.document_ids,
            "result": result,
            "saved_as": saved_as,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))