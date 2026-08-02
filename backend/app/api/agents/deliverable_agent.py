import asyncio
from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .tools import make_tools
from langchain_core.prompts import ChatPromptTemplate

_checkpointer=InMemorySaver()

class BRDRequirement(BaseModel):
    id: str = Field(description="Requirement id like REQ-1")
    text: str = Field(description="Requirement statement")
    type: str = Field(description="functional or non-functional")

class BRDRisk(BaseModel):
    id: str = Field(description="Risk id like RISK-1")
    description: str
    severity: str = Field(default="medium", description="low, medium, or high")

class BRDOutput(BaseModel):
    title: str = Field(description="BRD / brief title")
    executive_summary: str = Field(description="Short summary for leadership")
    scope: str = Field(description="What is in and out of scope")
    objectives: list[str] = []
    requirements: list[BRDRequirement] = []
    risks: list[BRDRisk] = []
    assumptions: list[str] = []
    open_questions: list[str] = []
    next_steps: list[str] = []

STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business analyst. Convert the research notes into a structured BRD draft. "
        "Only use facts present in the research. Do not invent requirements, risks, or numbers.",
    ),
    ("human", "Research notes:\n{research}\n\nProduce the structured BRD draft."),
])

def _run_sync(llm, embed_model, docs: list, params: dict, session_id: str) -> dict:
    tools = make_tools(llm, docs, embed_model, session_id=session_id)
    focus = params.get(
        "focus",
        "scope, objectives, requirements, risks, assumptions, and recommended next steps",
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a BA writing a Business Requirements Document draft. "
            "Use search_document (and analyze_csv if a dataset is present) to gather facts. "
            "Search multiple times covering: problem/scope, objectives, requirements, "
            "risks, assumptions, and gaps. "
            "When done, write clear research notes covering all BRD sections. "
            "Do not invent facts not found in the documents."
        ),
        checkpointer=_checkpointer,
    )
    user_msg = (
        f"Draft a BRD / leadership brief from the documents. Focus: {focus}. "
        "Research thoroughly across the documents, then summarize everything needed for a BRD."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": f"brd:{session_id}"}},
    )
    research = result["messages"][-1].content
    structured = (
        STRUCTURE_PROMPT | llm.with_structured_output(BRDOutput)
    ).invoke({"research": research})
    return structured.model_dump()

async def run(llm, embed_model, docs: list, params: dict, session_id: str = "ba-task") -> dict:
    return await asyncio.to_thread(_run_sync, llm, embed_model, docs, params, session_id)