import asyncio
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .tools import make_tools
from langchain_core.prompts import ChatPromptTemplate

_checkpointer = InMemorySaver()


class Risk(BaseModel):
    id: str = Field(description="Risk id like RISK-1")
    description: str = Field(description="The risk statement")
    severity: str = Field(default="medium", description="low, medium, or high")
    source_hint: str = Field(default="", description="Where this came from in the document")


class RisksOutput(BaseModel):
    risks: list[Risk]
    assumptions: list[str] = []
    open_questions: list[str] = []


STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business analyst. Convert the research notes into a structured risk analysis. "
        "Only use facts present in the research. Do not invent risks.",
    ),
    ("human", "Research notes:\n{research}\n\nExtract risks, assumptions, and open questions."),
])


def _run_sync(llm, embed_model, docs: list, params: dict, session_id: str) -> dict:
    tools = make_tools(llm, docs, embed_model, session_id=session_id)
    focus = params.get("focus", "risks, assumptions, dependencies, and gaps")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a BA risk analyst. "
            "Use search_document to find risks, assumptions, dependencies, and gaps. "
            "Search multiple times with different angles if needed. "
            "When done, write clear research notes of everything found, with source hints. "
            "Do not invent risks not supported by the documents."
        ),
        checkpointer=_checkpointer,
    )
    user_msg = (
        f"Identify risks and assumptions in the documents. Focus: {focus}. "
        "Search thoroughly, then summarize everything you found."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": f"risk:{session_id}"}},
    )
    research = result["messages"][-1].content
    structured = (
        STRUCTURE_PROMPT | llm.with_structured_output(RisksOutput)
    ).invoke({"research": research})
    return structured.model_dump()


async def run(llm, embed_model, docs: list, params: dict, session_id: str = "ba-task") -> dict:
    return await asyncio.to_thread(_run_sync, llm, embed_model, docs, params, session_id)