import asyncio
from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from .tools import make_tools
from langchain_core.prompts import ChatPromptTemplate

_checkpointer = InMemorySaver()

class summary_output(BaseModel):
    title:str = Field(description="Short title for the summary")
    summary: str = Field(description="Executive summary in plain business language")
    key_points: list[str] = Field(default=[], description="Main takeaways as bullet points")
    open_questions: list[str] = []

STRUCTURE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business analyst. Convert the research notes into a structured executive summary. "
        "Only use facts present in the research. Do not invent details.",
    ),
    ("human", "Research notes:\n{research}\n\nWrite the executive summary."),
])

def _run_sync(llm,embed_model,docs,params,session_id):
    try:
        tools=make_tools(llm=llm,docs=docs,embed_model=embed_model,session_id=session_id)
        focus = params.get("focus", "main topics, key findings, decisions, risks, and numbers")
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "You are a BA summarizer. "
                "Use search_document (and analyze_csv if a dataset is present) to gather the important content. "
                "Search with multiple angles if needed. "
                "When done, write clear research notes covering the key points. "
                "Do not invent facts not found in the documents."
            ),
            checkpointer=_checkpointer,
        )
        user_msg = (
            f"Summarize the documents for leadership. Focus: {focus}. "
            "Gather content thoroughly, then summarize everything you found."
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config={"configurable": {"thread_id": f"sum:{session_id}"}},
        )
        research = result["messages"][-1].content
        structured = (
            STRUCTURE_PROMPT | llm.with_structured_output(summary_output)
        ).invoke({"research": research})
        return structured.model_dump()
    except Exception as e:
        raise e
    
async def run(llm, embed_model, docs: list, params: dict, session_id: str = "ba-task") -> dict:
    return await asyncio.to_thread(_run_sync, llm, embed_model, docs, params, session_id)
