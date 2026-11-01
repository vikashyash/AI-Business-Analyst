from langchain_core.tools import StructuredTool
from .agentic_rag_retrieval import retrieve_with_quality_loop,_format_retrieval_result
from .csv_agent import _analyze_csv_sync

def make_doc_search_tool(llm,doc_id,embed_model,doc_label):
    try:
        def search_document(query: str) -> str:
            f"""Search the uploaded document for relevant passages. Input: a search query string."""
            result = retrieve_with_quality_loop(
                llm=llm,
                question=query,
                doc_id=doc_id,
                embed_model=embed_model,
            )
            return _format_retrieval_result(result)
        
        return StructuredTool.from_function(
            func=search_document,
            name="search_document",
            description=f"Search '{doc_label}' for relevant passages. Input: a search query string.",
        )
    except Exception as e:
        return str(e)

def make_csv_tool(llm,csv_path,session_id,doc_label):
    try:
        def analyze_csv(question:str)->str:
            """Analyze a CSV file with pandas. Input: your data question."""
            return _analyze_csv_sync(llm=llm,path=csv_path,question=question,session_id=session_id)
        
        return StructuredTool.from_function(
            func=analyze_csv,
            name="analyze_csv",
            description=f"Analyze CSV '{doc_label}' with pandas. Input: your data question."

        )
    except Exception as e:
        return str(e)

def make_tools(llm, docs: list, embed_model, session_id: str = "ba-tools"):
    try:
        tools=[]
        for doc in docs:
            name = doc.get("filename", doc["id"])
            if doc["type"] in ("PDF", "Text"):
                tools.append(make_doc_search_tool(llm, doc["id"], embed_model, name))
            elif doc["type"] == "CSV":
                tools.append(make_csv_tool(llm, doc["path"], session_id, name))
        return tools
    except Exception as e:
        return str(e)
    
def gather_doc_context(llm, embed_model, docs: list, query: str) -> str:
    """Direct retrieval across all PDF/Text docs (helper for structured output step)."""
    parts = []
    for doc in docs:
        if doc["type"] not in ("PDF", "Text"):
            continue
        result = retrieve_with_quality_loop(
            llm=llm,
            question=query,
            doc_id=doc["id"],
            embed_model=embed_model,
        )
        parts.append(f"=== {doc.get('filename', doc['id'])} ===\n{result['context']}")
    return "\n\n".join(parts) if parts else "No document context found."


