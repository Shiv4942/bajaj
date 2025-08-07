import os
import time
import tempfile
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

app = FastAPI()

# Globals
vectorstore = None
index = None
embedding_model = None

# Initialize Pinecone
def initialize_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing Pinecone API key")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "retrieval"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
    return pc, pc.Index(index_name)

def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# Clear existing vectors from index
def clear_index():
    global index
    try:
        # Delete all vectors from the index
        index.delete(delete_all=True)
        print("✅ Cleared all existing vectors from index")
        time.sleep(2)  # Wait a moment for deletion to complete
    except Exception as e:
        print(f"⚠️ Warning: Could not clear index: {e}")

# PDF processing and indexing
def process_pdf_files(pdf_files: List[UploadFile]):
    global vectorstore, index, embedding_model

    # Initialize vectorstore if not already done (keeps existing documents)
    if vectorstore is None:
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
    
    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(pdf_file.file, tmp)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Add metadata
        for doc in docs:
            doc.metadata.update({
                "source_file": pdf_file.filename,
                "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add documents to vectorstore
        vectorstore.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"Added {len(chunks)} chunks from {pdf_file.filename}")

        os.unlink(tmp_path)

    print(f"✅ Total chunks processed: {total_chunks}")
    return total_chunks

# Query processing
def process_query(query_text: str):
    if not vectorstore:
        return {"error": "Vectorstore not initialized. Upload and process documents first."}

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return {"error": "Missing GROQ API key"}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
    )

    prompt_template = """Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Provide a concise, to-the-point answer in maximum ONE sentence
- Include specific numbers, percentages, time periods, and conditions mentioned in the documents
- Be precise with numerical values (e.g., "36 months", "1% of Sum Insured", "5% discount")
- If the answer involves conditions, state them briefly in the same sentence
- Do not add explanations or additional context - just the direct answer
- If you cannot find the information, say "Information not available in the provided documents"

Context: {context}

Question: {question}

Answer (one sentence):"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain(query_text)
    return {
        "answer": result.get("result", "No result"),
        "sources": [
            {
                "source": doc.metadata.get("source_file", "Unknown"),
                "excerpt": doc.page_content[:300] + "..."
            } for doc in result.get("source_documents", [])[:3]
        ]
    }

# Get current index stats
def get_index_stats():
    global index
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }
    except Exception as e:
        return {"error": f"Could not get index stats: {e}"}

# API Endpoints

@app.on_event("startup")
def startup_event():
    global index, embedding_model
    pc, index = initialize_pinecone()
    embedding_model = initialize_embeddings()

@app.get("/")
def read_root():
    return {"message": "PDF RAG API is running!", "status": "healthy"}

@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        chunks = process_pdf_files(files)
        stats = get_index_stats()
        return {
            "message": f"{chunks} new document chunks processed successfully",
            "files_processed": [f.filename for f in files],
            "index_stats": stats,
            "note": "Documents added to existing collection - queries will search all uploaded documents"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query/")
async def query_pdf(question: str = Form(...)):
    result = process_query(question)
    return result

@app.get("/status/")
def get_status():
    global vectorstore, index, embedding_model
    stats = get_index_stats()
    return {
        "status": "running",
        "vectorstore_initialized": vectorstore is not None,
        "index_initialized": index is not None,
        "embedding_model_initialized": embedding_model is not None,
        "index_stats": stats
    }

@app.get("/documents/")
def list_documents():
    """List all documents currently in the vector store"""
    if not vectorstore:
        return {"documents": [], "message": "No documents uploaded yet"}
    
    try:
        # Get a sample of documents to see what's in the store
        sample_docs = vectorstore.similarity_search("", k=50)  # Get more docs to see all sources
        
        # Extract unique source files
        sources = set()
        for doc in sample_docs:
            source = doc.metadata.get("source_file", "Unknown")
            upload_time = doc.metadata.get("upload_time", "Unknown")
            sources.add(f"{source} (uploaded: {upload_time})")
        
        return {
            "total_documents": len(sources),
            "documents": sorted(list(sources)),
            "total_chunks": len(sample_docs)
        }
    except Exception as e:
        return {"error": f"Could not retrieve documents: {e}"}

@app.post("/clear/")
def clear_vectorstore():
    """Manually clear all vectors from the index"""
    try:
        clear_index()
        global vectorstore
        vectorstore = None
        return {"message": "Vector store cleared successfully - all documents removed"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  
    uvicorn.run("main:app", host="0.0.0.0", port=port)
