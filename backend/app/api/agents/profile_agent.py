import asyncio
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .tools import make_tools
from langchain_core.prompts import ChatPromptTemplate

_checkpointer = InMemorySaver()


class ColumnInsight(BaseModel):
    column: str
    observation: str = Field(description="What is notable about this column")


class ProfileOutput(BaseModel):
    dataset_overview: str = Field(description="Rows, columns, and what the dataset is about")
    column_insights: list[ColumnInsight] = []
    key_findings: list[str] = []
    data_quality_notes: list[str] = Field(default=[], description="Missing values, outliers, anomalies")


STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data analyst. Convert the research notes into a structured data profile. "
        "Only use facts present in the research. Do not invent numbers.",
    ),
    ("human", "Research notes:\n{research}\n\nProduce the structured data profile."),
])


def _run_sync(llm, embed_model, docs: list, params: dict, session_id: str) -> dict:
    tools = make_tools(llm, docs, embed_model, session_id=session_id)
    focus = params.get("focus", "row/column counts, column types, distributions, missing values, outliers")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a BA data profiler. "
            "Use analyze_csv to inspect the dataset: shape, columns, dtypes, summary stats, "
            "missing values, and notable patterns. "
            "Ask multiple analysis questions if needed. "
            "When done, write clear research notes of everything you found. "
            "Do not invent numbers not returned by the tool."
        ),
        checkpointer=_checkpointer,
    )
    user_msg = (
        f"Profile the dataset. Focus: {focus}. "
        "Query the CSV thoroughly, then summarize the findings."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": f"profile:{session_id}"}},
    )
    research = result["messages"][-1].content
    structured = (
        STRUCTURE_PROMPT | llm.with_structured_output(ProfileOutput)
    ).invoke({"research": research})
    return structured.model_dump()


async def run(llm, embed_model, docs: list, params: dict, session_id: str = "ba-task") -> dict:
    return await asyncio.to_thread(_run_sync, llm, embed_model, docs, params, session_id)