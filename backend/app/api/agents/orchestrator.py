import asyncio
import json
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,END

from ..tasks.run_tasks import HANDLERS,TASK_DOC_REQUIREMENTS
from ..kb_handler.kb_knowledge_file_store import get_document
from ..workspaces.workspace_store import get_workspace,save_deliverable


AVAILABLE_TASKS=list(HANDLERS.keys())

class PlanOutput(BaseModel):
    tasks:list[str]=Field(
        description=f"Ordered list of BA tasks to run. Must be from: {AVAILABLE_TASKS}"
    )
    reasoning: str = Field(description="Why these tasks were chosen")

"""
plan output structure:

{
  "plan": [
    "summarize",
    "profile_data",
    "find_risks",
    "extract_requirements",
    "generate_brd"
  ],
  "reasoning": "Goal asks for analysis of PDF + CSV and a leadership brief. Start with summarize and profile_data for overview, then find_risks and extract_requirements for BA detail, finish with generate_brd for the packaged deliverable."
}
"""


class FinalBrief(BaseModel):
    title: str
    executive_summary: str
    key_findings: list[str] = []
    requirements_highlights: list[str] = []
    risks_highlights: list[str] = []
    recommendations: list[str] = []
    open_questions: list[str] = []

class OrchestratorState(TypedDict): #final synth_node response structure
    goal: str
    ws_id: str
    session_id: str
    docs: list
    plan: list
    reasoning: str
    task_results: list
    final_deliverable: dict

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a BA orchestrator. Given a user goal and available documents, "
        "choose an ordered list of BA tasks to run. "
        f"Allowed tasks only: {AVAILABLE_TASKS}. "
        "Rules:\n"
        "- Use profile_data only if a CSV document exists.\n"
        "- Use extract_requirements / find_risks / generate_brd only if PDF or Text exists.\n"
        "- Prefer a short plan (2–4 tasks). Include generate_brd when the goal asks for a brief/BRD.\n"
        "- Do not invent task names.",
    ),
    (
        "human",
        "Goal:\n{goal}\n\n"
        "Documents:\n{doc_summary}\n\n"
        "Pick the task plan.",
    ),
])
SYNTH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business analyst. Combine the task results into one leadership brief. "
        "Only use facts present in the results. Do not invent details.",
    ),
    (
        "human",
        "Original goal:\n{goal}\n\n"
        "Task results (JSON):\n{results}\n\n"
        "Produce the final brief.",
    ),
])

def doc_summary(docs:list)->str:
    lines=[]
    for d in docs:
        lines.append(f"-{d.get('filename',d['id'])} (type={d['type']}, id={d['id']})")
    return "\n".join(lines) if lines else "No documents"

def filter_docs_for_task(doc:list,task_type:str)->list:
    allowed = TASK_DOC_REQUIREMENTS.get(task_type, ())
    return [d for d in doc if d["type"] in allowed]

def build_graph(llm,embed_model):

    async def plan_node(state:OrchestratorState)->dict:
        chain = PLAN_PROMPT|llm.with_structured_output(PlanOutput)
        output=await asyncio.to_thread(chain.invoke,{"goal":state["goal"],"doc_summary":doc_summary(state['docs'])})
        valid=[]

        for t in output.tasks:
            if t not in HANDLERS:
                continue
            if filter_docs_for_task(state["docs"],t):
                valid.append(t)
        if not valid:
            if filter_docs_for_task(state["docs"], "summarize"):
                valid = ["summarize"]
            elif filter_docs_for_task(state["docs"], "profile_data"):
                valid = ["profile_data"]
        return {"plan": valid, "reasoning": output.reasoning}

    async def execute_node(state: OrchestratorState) -> dict:
            results = []
            for task_type in state["plan"]:
                docs_for_task = filter_docs_for_task(state["docs"], task_type)
                if not docs_for_task:
                    results.append({
                        "task_type": task_type,
                        "status": "skipped",
                        "reason": "no matching documents",
                    })
                    continue
                try:
                    output = await HANDLERS[task_type](
                        llm,
                        embed_model,
                        docs_for_task,
                        {},
                        f"{state['session_id']}:{task_type}",
                    )
                    results.append({
                        "task_type": task_type,
                        "status": "ok",
                        "result": output,
                    })
                except Exception as e:
                    results.append({
                        "task_type": task_type,
                        "status": "error",
                        "error": str(e),
                    })
            return {"task_results": results}

    async def synthesize_node(state: OrchestratorState) -> dict:
        chain = SYNTH_PROMPT | llm.with_structured_output(FinalBrief)
        payload = json.dumps(state["task_results"], indent=2, default=str)
        brief = await asyncio.to_thread(
            chain.invoke,
            {"goal": state["goal"], "results": payload},
        )
        final = brief.model_dump()
        final["plan"] = state["plan"]
        final["reasoning"] = state.get("reasoning", "")
        return {"final_deliverable": final}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan",plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()

# main entry point.
async def run_orchestrator(llm,embed_model,ws_id: str,goal: str,session_id: str,document_ids: list[str] | None = None,params: dict | None = None,)->dict:
    ws = await get_workspace(ws_id)
    ws_doc_ids = ws.get("document_ids", [])
    if document_ids:
        for doc_id in document_ids:
            if doc_id not in ws_doc_ids:
                raise ValueError(f"Document {doc_id} is not in this workspace")
        selected_ids = document_ids
    else:
        selected_ids = ws_doc_ids
    if not selected_ids:
        raise ValueError("Workspace has no documents to analyze")
    docs = [await get_document(doc_id) for doc_id in selected_ids]
    app = build_graph(llm, embed_model)
    initial: OrchestratorState = {
        "goal": goal,
        "ws_id": ws_id,
        "session_id": session_id,
        "docs": docs,
        "plan": [],
        "reasoning": "",
        "task_results": [],
        "final_deliverable": {},
    }
    final_state = await app.ainvoke(initial)
    deliverable = final_state["final_deliverable"]
    saved_as = await save_deliverable(ws_id, "orchestrator_run", {
        "goal": goal,
        "plan": final_state["plan"],
        "reasoning": final_state.get("reasoning", ""),
        "task_results": final_state["task_results"],
        "final_deliverable": deliverable,
    })
    return {
        "workspace_id": ws_id,
        "goal": goal,
        "session_id": session_id,
        "plan": final_state["plan"],
        "reasoning": final_state.get("reasoning", ""),
        "task_results": final_state["task_results"],
        "final_deliverable": deliverable,
        "saved_as": saved_as,
    }



