import asyncio
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ...config import KB_DIR
from ..services import detect_file_type, _parse_pdf, _parse_txt, clean_text
from chromadb import PersistentClient



def create_retrieve_persistent_client(doc_id):
    try:
        path = KB_DIR / doc_id
        path.mkdir(parents=True, exist_ok=True)
        return PersistentClient(str(path))
    except Exception as e:
        raise RuntimeError(f"chroma client failed: {e}") from e

def handle_recursion_splitter(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        return splitter.split_text(text)
    except Exception as e:
        raise RuntimeError(f"text split failed: {e}") from e

def prepare_documents(chunks, filename):
    try:
        ids, docs, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i, "type": "document"})
        return ids, docs, metadatas
    except Exception as e:
        raise RuntimeError(f"prepare documents failed: {e}") from e

def generate_embeddings(model,text):
    try:
        return model.encode(text)
    except Exception as e:
        raise RuntimeError(f"embedding failed: {e}") from e

def store_in_chroma(ids, docs, embeddings, metadatas, doc_id):
    try:
        client = create_retrieve_persistent_client(doc_id)
        collection = client.get_or_create_collection(doc_id)
        collection.add(ids=ids, documents=docs, embeddings=[e.tolist() for e in embeddings], metadatas=metadatas)
        return "insertion successful"
    except Exception as e:
        raise RuntimeError(f"chroma store failed: {e}") from e

def _process_file_for_insertion(model,file, file_name, doc_id):
    try:
        file_type = detect_file_type(file)
        if file_type == "PDF":
            text = _parse_pdf(file)
        elif file_type == "Text":
            text = _parse_txt(file)
        else:
            raise ValueError("Only PDF and Text files can be indexed")
        text = clean_text(text)
        if not text:
            raise ValueError("No text could be extracted from the file")
        chunks = handle_recursion_splitter(text)
        if not chunks:
            raise ValueError("No text chunks produced for indexing")
        ids, docs, metadatas = prepare_documents(chunks, file_name)
        embeddings = generate_embeddings(model,docs)
        return store_in_chroma(ids, docs, embeddings, metadatas, doc_id)
    except Exception as e:
        raise RuntimeError(f"file insertion failed: {e}") from e

async def process_file_for_insertion(model,file, file_name, doc_id):
    return await asyncio.to_thread(_process_file_for_insertion,model, file, file_name, doc_id)
