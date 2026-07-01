import asyncio
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from ..kb_handler.kb_insertion import create_retrieve_persistent_client
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

_checkpointer = InMemorySaver()
_agents: dict[str, object] = {}

MIN_SCORE = 3
MAX_RETRIES = 3


class RetrievalScore(BaseModel):
    score: int = Field(description="Relevance score from 1 (bad) to 5 (excellent)")
    is_sufficient: bool = Field(description="True if chunks are good enough to answer")
    reason: str = Field(description="Why the context is or is not sufficient")
    missing_info: str = Field(description="What information is still missing, if any")


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade retrieval quality for a business analysis task. "
     "Be strict: score 4-5 only if chunks clearly contain facts needed to answer."),
    ("human",
     "User question:\n{question}\n\n"
     "Retrieved context:\n{context}\n\n"
     "Grade whether this context is sufficient."),
])


def grade_retrieval(llm, question: str, context: str) -> RetrievalScore:
    grader = llm.with_structured_output(RetrievalScore)
    chain = GRADE_PROMPT | grader
    return chain.invoke({"question": question, "context": context})


class RewrittenQuery(BaseModel):
    query: str = Field(description="Improved search query for document retrieval")
    strategy: str = Field(description="What changed in the query and why")


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the search query to retrieve better document passages. "
     "Use business analyst angles: stakeholders, risks, KPIs, requirements, assumptions."),
    ("human",
     "Original user question:\n{question}\n\n"
     "Previous search query:\n{previous_query}\n\n"
     "Why previous retrieval failed:\n{reason}\n\n"
     "Missing info:\n{missing_info}\n\n"
     "Write a better search query."),
])


def rewrite_query(llm, question: str, previous_query: str, grade: RetrievalScore) -> RewrittenQuery:
    rewriter = llm.with_structured_output(RewrittenQuery)
    chain = REWRITE_PROMPT | rewriter
    return chain.invoke({
        "question": question,
        "previous_query": previous_query,
        "reason": grade.reason,
        "missing_info": grade.missing_info,
    })


def _retrieve_chunks(doc_id: str, embed_model, query: str, k: int = 5) -> str:
    client = create_retrieve_persistent_client(doc_id)
    collection = client.get_collection(doc_id)
    if collection.count() == 0:
        raise ValueError(f"No indexed chunks for document {doc_id}")

    embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents"],
    )
    chunks = results["documents"][0]
    return "\n\n---\n\n".join(chunks)


def retrieve_with_quality_loop(llm, question: str, doc_id: str, embed_model, k: int = 5) -> dict:
    search_query = question
    best_result = {
        "context": "",
        "score": 0,
        "attempts": [],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        chunk_k = k if attempt < MAX_RETRIES else max(k, 8)
        retrieved_context = _retrieve_chunks(
            doc_id=doc_id,
            embed_model=embed_model,
            query=search_query,
            k=chunk_k,
        )
        grade = grade_retrieval(llm=llm, question=question, context=retrieved_context)

        best_result["attempts"].append({
            "attempt": attempt,
            "search_query": search_query,
            "score": grade.score,
            "is_sufficient": grade.is_sufficient,
            "reason": grade.reason,
        })

        if grade.score > best_result["score"]:
            best_result["score"] = grade.score
            best_result["context"] = retrieved_context

        if grade.is_sufficient and grade.score >= MIN_SCORE:
            best_result["final_query"] = search_query
            best_result["stopped_reason"] = "quality_threshold_met"
            return best_result

        if attempt < MAX_RETRIES:
            rewritten = rewrite_query(llm, question, search_query, grade)
            search_query = rewritten.query

    best_result["final_query"] = search_query
    best_result["stopped_reason"] = "max_retries_reached"
    return best_result


def _format_retrieval_result(result: dict) -> str:
    header = (
        f"[Retrieval score: {result['score']}/5 | "
        f"attempts: {len(result['attempts'])} | "
        f"stop: {result['stopped_reason']}]"
    )
    if not result["context"]:
        return f"{header}\n\nNo relevant context found in the document."
    if result["score"] < MIN_SCORE:
        return (
            f"{header}\n\n"
            "Retrieved context may be incomplete. Use only what is present.\n\n"
            f"{result['context']}"
        )
    return f"{header}\n\n{result['context']}"


def _get_rag_agent(llm, doc_id: str, embed_model):
    if doc_id not in _agents:
        @tool
        def search_document(query: str) -> str:
            """Search the uploaded document for relevant passages with quality-checked retrieval."""
            result = retrieve_with_quality_loop(
                llm=llm,
                question=query,
                doc_id=doc_id,
                embed_model=embed_model,
            )
            return _format_retrieval_result(result)

        _agents[doc_id] = create_agent(
            model=llm,
            tools=[search_document],
            system_prompt=(
                "You are a business analyst. "
                "Always call search_document first to ground your answer in the document. "
                "If the retrieval header shows a low score or incomplete context, "
                "answer only from what is present and clearly state any gaps. "
                "If the document does not contain the answer, say so clearly."
            ),
            checkpointer=_checkpointer,
            debug=True,
        )
    return _agents[doc_id]


def _thread_id(doc_id: str, session_id: str) -> str:
    return f"{doc_id}:{session_id}"


def _analyze_rag_sync(llm, doc_id: str, embed_model, question: str, session_id: str) -> str:
    agent = _get_rag_agent(llm, doc_id, embed_model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": _thread_id(doc_id, session_id)}},
    )
    return result["messages"][-1].content


async def analyze_rag(llm, doc_id: str, embed_model, question: str, session_id: str) -> str:
    return await asyncio.to_thread(_analyze_rag_sync, llm, doc_id, embed_model, question, session_id)
