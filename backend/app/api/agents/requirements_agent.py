import asyncio
from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .tools import make_tools
from langchain_core.prompts import ChatPromptTemplate

_checkpointer = InMemorySaver()

class Requirement(BaseModel):
    id: str = Field(description="Requirement id like REQ-1")
    text: str = Field(description="Requirement statement")
    type: str = Field(description="functional or non-functional")
    source_hint: str = Field(default="", description="Where this came from in the document")

class RequirementsOutput(BaseModel):
    requirements: list[Requirement]
    open_questions: list[str] = []
    assumptions: list[str] = []


STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business analyst. Convert the research notes into structured requirements. "
        "Only use facts present in the research. Do not invent requirements.",
    ),
    ("human", "Research notes:\n{research}\n\nExtract structured requirements."),
])

def _run_sync(llm, embed_model, docs: list, params: dict, session_id: str) -> dict:
    tools = make_tools(llm, docs, embed_model, session_id=session_id)
    focus = params.get("focus", "functional and non-functional requirements, constraints, acceptance criteria")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a BA requirements extractor. "
            "Always use search_document to find requirement passages. "
            "Search multiple times with different queries if needed "
            "(functional requirements, non-functional, constraints, acceptance criteria). "
            "When done, write a clear research summary of all requirements found, "
            "including source hints. Do not invent requirements not in the documents."
        ),
        checkpointer=_checkpointer,
    )
    user_msg = (
        f"Extract all requirements from the documents. Focus: {focus}. "
        "Search the documents thoroughly, then summarize everything you found."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": f"req:{session_id}"}},
    )
    research = result["messages"][-1].content
    structured = (
        STRUCTURE_PROMPT | llm.with_structured_output(RequirementsOutput)
    ).invoke({"research": research})
    return structured.model_dump()

async def run(llm, embed_model, docs: list, params: dict, session_id: str = "ba-task") -> dict:
    return await asyncio.to_thread(_run_sync, llm, embed_model, docs, params, session_id)

