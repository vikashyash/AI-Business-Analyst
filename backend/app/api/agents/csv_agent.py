import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from ..services import parse_csv

_sessions: dict[str, InMemoryChatMessageHistory] = {}

def _llm():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("MODEL_API_KEY")
    return ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"), google_api_key=key, temperature=0)

def _session_history(session_id: str):
    if session_id not in _sessions: _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]

def _question(msgs):
    if len(msgs) <= 1: return msgs[-1].content
    lines = [f"{'user' if m.type == 'human' else 'assistant'}: {m.content}" for m in msgs[:-1]]
    return "Prior conversation:\n" + "\n".join(lines) + f"\n\nQuestion: {msgs[-1].content}"

def _run(inputs: dict):
    df = parse_csv(inputs["path"])
    question = _question(inputs["messages"])
    agent = create_pandas_dataframe_agent(llm=_llm(), df=df, verbose=False, allow_dangerous_code=True)
    out = agent.invoke({"input": question})
    text = out.get("output", str(out)) if isinstance(out, dict) else str(out)
    return AIMessage(content=text)

_chain = RunnableWithMessageHistory(RunnableLambda(_run), _session_history, input_messages_key="messages")

def analyze_csv(path: str, question: str, session_id: str) -> str:
    out = _chain.invoke({"messages": [HumanMessage(content=question)], "path": path}, config={"configurable": {"session_id": session_id}})
    return out.content if hasattr(out, "content") else str(out)
