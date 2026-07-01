import asyncio
import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.agents import create_pandas_dataframe_agent
from ..services import _parse_csv

_sessions: dict[str, InMemoryChatMessageHistory] = {}



def _session_history(session_id: str):
    try:
        if session_id not in _sessions: 
            _sessions[session_id] = InMemoryChatMessageHistory()
        return _sessions[session_id]
    except Exception as e:
        raise RuntimeError(f"session history failed: {e}") from e

def _question(msgs):
    try:
        if len(msgs) <= 1: 
            return msgs[-1].content
        lines = [f"{'user' if m.type == 'human' else 'assistant'}: {m.content}" for m in msgs[:-1]]
        return "Prior conversation:\n" + "\n".join(lines) + f"\n\nQuestion: {msgs[-1].content}"
    except Exception as e:
        raise RuntimeError(f"question build failed: {e}") from e

def _run(inputs: dict):
    try:
        df = _parse_csv(inputs["path"])
        question = _question(inputs["messages"])
        agent = create_pandas_dataframe_agent(llm=inputs["llm"], df=df, verbose=False, allow_dangerous_code=True)
        out = agent.invoke({"input": question})
        text = out.get("output", str(out)) if isinstance(out, dict) else str(out)
        return AIMessage(content=text)
    except Exception as e:
        raise RuntimeError(f"csv agent run failed: {e}") from e

_chain = RunnableWithMessageHistory(RunnableLambda(_run), _session_history, input_messages_key="messages")

def _analyze_csv_sync(llm,path: str, question: str, session_id: str) -> str:
    out = _chain.invoke(
        {"messages": [HumanMessage(content=question)], "path": path, "llm": llm},
        config={"configurable": {"session_id": session_id}},
    )
    return out.content if hasattr(out, "content") else str(out)

async def analyze_csv(llm,path: str, question: str, session_id: str) -> str:
    try:
        return await asyncio.to_thread(_analyze_csv_sync, llm,path, question, session_id)
    except Exception as e:
        raise RuntimeError(f"analyze_csv failed: {e}") from e
