from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..services import detect_file_type,parse_csv,parse_pdf,parse_txt,clean_text
import chromadb,uuid
from chromadb import PersistentClient

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create_persistent_client(file_name):
    try:
        return PersistentClient(f"kb/{file_name}")
    except Exception as e:
        return str(e)

def handle_recursion_splitter(docs):
    try:
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200,  
            add_start_index=True, 
        )
        all_splits = text_splitter.split_documents(docs)
        return all_splits
    except Exception as e:
        return str(e)
    
    
def prepare_documents(chunks, filename):
    docs = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        ids.append(str(uuid.uuid4()))
        
        docs.append(chunk)
        
        metadatas.append({
            "source": filename,
            "chunk_index": i,
            "type": "document"
        })

    return ids, docs, metadatas

def generate_embeddings(text):
    return model.encode(text)

def store_in_chroma(ids, docs, embeddings, metadatas,file_name):
    try:
        collection=create_persistent_client(file_name=file_name)
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return "insertion successful"
    except Exception as e:
        return str(e)

def process_file_for_insertion(file,file_name):   
    file_type=detect_file_type(file=file)
    if file_type=="PDF":
        text=parse_pdf(file=file)
    elif file_type == "Text":
        text = parse_txt(file)

    # clean text
    cleaned_text = clean_text(text)

    # chunk text
    chunks=handle_recursion_splitter(clean_text)

    # doc preparation
    ids, docs, metadatas = prepare_documents(chunks, file_name)

    # Step 5: Embed
    embeddings = generate_embeddings(docs)

    # step 6: store
    store_in_chroma(ids, docs, embeddings, metadatas)





    



    

    
