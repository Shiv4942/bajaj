# main.py
import os
import time
import tempfile
import shutil
import requests
import gc
from tempfile import SpooledTemporaryFile
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain / Pinecone imports (these must be installed in your env)
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI(title="HackRX RAG API", version="1.0.0")
security = HTTPBearer()

# Pydantic models
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF document
    questions: List[str]  # Array of questions

class HackRXResponse(BaseModel):
    answers: List[str]  # Array of answers corresponding to questions

# Globals
vectorstore = None
index = None
embedding_model = None
pinecone_client = None

# --------------------------
# Initialization helpers
# --------------------------
def initialize_pinecone():
    """Initialize Pinecone client and index (defensive)."""
    global pinecone_client, index

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY env var.")

    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_client = pc

    index_name = "retrieval"
    # list_indexes() shape may differ between SDK versions - handle generically
    try:
        existing = pc.list_indexes()
        try:
            names = existing.names()
        except Exception:
            names = existing if isinstance(existing, list) else []
    except Exception as e:
        raise RuntimeError(f"Could not list Pinecone indexes: {e}")

    if index_name not in names:
        try:
            pc.create_index(
                name=index_name,
                dimension=384,  # matches MiniLM embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(5)
        except Exception as e:
            raise RuntimeError(f"Failed to create Pinecone index '{index_name}': {e}")

    try:
        idx = pc.Index(index_name)
    except Exception as e:
        raise RuntimeError(f"Failed to get Pinecone index '{index_name}': {e}")

    index = idx
    return pc, idx

def initialize_embeddings():
    """Initialize a memory-friendly sentence-transformers embedding."""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            print("✅ Embedding model initialized (MiniLM)")
        except Exception as e:
            print("❌ Failed to init embeddings:", e)
            raise
    return embedding_model

def clear_index():
    """Clear all vectors from index if available."""
    global index
    if index is None:
        print("⚠️ clear_index called but index is not initialized.")
        return
    try:
        try:
            index.delete(delete_all=True)
        except TypeError:
            index.delete_all()
        print("✅ Cleared all existing vectors from index")
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Warning: Could not clear index: {e}")

# --------------------------
# PDF download & processing (streaming, memory-friendly)
# --------------------------
def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return temporary file path on disk."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        print(f"Downloading PDF from: {url}")
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            print(f"✅ PDF downloaded to: {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        print(f"❌ Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def _process_pages_and_add_to_vectorstore(pages, metadata_base: dict, batch_size: int = 8):
    """
    Helper that accepts an iterable/list of page docs, splits pages into chunks,
    and adds them to vectorstore in batches.
    Returns number of chunks added.
    """
    global vectorstore, embedding_model
    if vectorstore is None:
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    batch_docs = []
    total_added = 0

    for page_doc in pages:
        # merge metadata
        page_doc.metadata.update(metadata_base)
        page_chunks = splitter.split_documents([page_doc])
        for chunk in page_chunks:
            batch_docs.append(chunk)
            if len(batch_docs) >= batch_size:
                vectorstore.add_documents(batch_docs)
                total_added += len(batch_docs)
                print(f"✅ Added batch of {len(batch_docs)} chunks (total={total_added})")
                batch_docs = []
                gc.collect()

    # flush leftover
    if batch_docs:
        vectorstore.add_documents(batch_docs)
        total_added += len(batch_docs)
        print(f"✅ Added final batch of {len(batch_docs)} chunks (total={total_added})")
        batch_docs = []
        gc.collect()

    return total_added

def process_pdf_from_url(pdf_url: str, batch_size: int = 8):
    """
    Download PDF to disk, process page-by-page and upload small batches to Pinecone.
    Returns number of chunks added.
    """
    global vectorstore, index, embedding_model

    if index is None:
        raise RuntimeError("Index not initialized. Call initialize_pinecone() first.")
    initialize_embeddings()
    if vectorstore is None:
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

    tmp_path = download_pdf_from_url(pdf_url)
    total_added = 0
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        print(f"✅ PDF loaded: {len(pages)} pages (processing streaming)")

        metadata_base = {
            "source_url": pdf_url,
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        total_added = _process_pages_and_add_to_vectorstore(pages, metadata_base, batch_size=batch_size)
        return total_added

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                print("🧹 Temp PDF removed:", tmp_path)
        except Exception:
            pass
        gc.collect()

def process_pdf_files(pdf_files: List[UploadFile], batch_size: int = 8):
    """
    Process a list of UploadFile objects one-by-one using SpooledTemporaryFile to limit memory usage.
    """
    global vectorstore, index, embedding_model
    if index is None:
        raise RuntimeError("Index not initialized. Call initialize_pinecone() first.")
    initialize_embeddings()
    if vectorstore is None:
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

    total_chunks = 0

    for upload in pdf_files:
        print(f"📥 Processing upload: {upload.filename}")

        # Use SpooledTemporaryFile to keep small uploads in memory but larger ones on disk
        tmp_path = None
        with SpooledTemporaryFile(max_size=10 * 1024 * 1024, suffix=".pdf") as spooled:
            try:
                shutil.copyfileobj(upload.file, spooled)
                spooled.seek(0)

                # Write to real disk file for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as disk_tmp:
                    shutil.copyfileobj(spooled, disk_tmp)
                    tmp_path = disk_tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                print(f"  -> loaded {len(pages)} pages from {upload.filename}")

                metadata_base = {
                    "source_file": upload.filename,
                    "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                added = _process_pages_and_add_to_vectorstore(pages, metadata_base, batch_size=batch_size)
                total_chunks += added

            except Exception as e:
                print(f"❌ Error processing uploaded file {upload.filename}: {e}")
                raise
            finally:
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass
                gc.collect()

    print(f"✅ Total chunks processed from uploads: {total_chunks}")
    return total_chunks

# --------------------------
# Question processing
# --------------------------
def process_single_question(question: str) -> str:
    """Run RetrievalQA with ChatGroq using the vectorstore retriever."""
    global vectorstore

    if vectorstore is None:
        return "Error: No document processed"

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: GROQ API key not configured"

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.1
        )

        prompt_template = """Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Provide a precise, comprehensive answer based strictly on the document content
- If you cannot find the specific information, say "Information not available in the provided document"

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa_chain(question)
        answer = result.get("result", "No answer generated")
        print(f"✅ Question processed: {question[:50]}... -> {answer[:100]}...")
        return answer

    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# --------------------------
# Auth helper (fixed)
# --------------------------
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify Bearer token - accepts any non-empty Bearer token for now.
    HTTPAuthorizationCredentials has `.scheme` and `.credentials`.
    """
    token = getattr(credentials, "credentials", None)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    print(f"✅ Token verified: {token[:10]}...")
    return token

# --------------------------
# Endpoints
# --------------------------
@app.api_route("/", methods=["GET", "POST", "HEAD", "OPTIONS"])
def read_root():
    return {"message": "HackRX RAG API is running!", "status": "healthy"}

@app.api_route("/health", methods=["GET", "POST", "HEAD", "OPTIONS"])
def health_check():
    return {"status": "healthy", "service": "HackRX RAG API"}

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    start_time = time.time()
    try:
        print(f"🚀 HackRX request received with {len(request.questions)} questions")

        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")

        # Process document (streaming)
        chunks_processed = process_pdf_from_url(request.documents, batch_size=8)
        print(f"✅ Document processed: {chunks_processed} chunks")

        answers = []
        for i, question in enumerate(request.questions):
            if question.strip():
                print(f"🔍 Processing question {i+1}/{len(request.questions)}")
                answer = process_single_question(question.strip())
                answers.append(answer)
            else:
                answers.append("Empty question provided")

        processing_time = time.time() - start_time
        print(f"✅ All questions processed in {processing_time:.2f}s. Returning {len(answers)} answers.")
        return HackRXResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Internal server error after {processing_time:.2f}s: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        if len(files) > 3:
            raise HTTPException(status_code=400, detail="Too many files. Max 3 at a time.")
        chunks = process_pdf_files(files, batch_size=8)
        stats = get_index_stats() if index else {}
        return {
            "message": f"{chunks} document chunks processed successfully",
            "files_processed": [f.filename for f in files],
            "index_stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        print("❌ Upload endpoint error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_pdf(question: str = Form(...)):
    result = process_query(question)
    return result

@app.get("/status")
def get_status():
    global vectorstore, index, embedding_model
    try:
        stats = None
        if index:
            stats = index.describe_index_stats()
    except Exception as e:
        print("⚠️ Could not fetch index stats:", e)
        stats = None

    return {
        "status": "running",
        "service": "HackRX RAG API",
        "vectorstore_initialized": vectorstore is not None,
        "index_initialized": index is not None,
        "embedding_model_initialized": embedding_model is not None,
        "index_stats": {
            "total_vectors": getattr(stats, "total_vector_count", 0) if stats else 0,
            "dimension": getattr(stats, "dimension", None) if stats else None,
            "index_fullness": getattr(stats, "index_fullness", None) if stats else None
        } if stats else None,
        "required_env_vars": {
            "PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
            "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY"))
        },
        "endpoints": {
            "hackrx_main": "POST /hackrx/run (requires Bearer token)",
            "development_upload": "POST /upload/",
            "development_query": "POST /query/",
            "health": "GET /health",
            "status": "GET /status"
        }
    }

@app.post("/test")
async def test_endpoint(request: HackRXRequest):
    try:
        print("🧪 Test endpoint called")
        chunks_processed = process_pdf_from_url(request.documents)
        if request.questions:
            sample_answer = process_single_question(request.questions[0])
            return {
                "test": True,
                "chunks_processed": chunks_processed,
                "sample_question": request.questions[0],
                "sample_answer": sample_answer,
                "total_questions": len(request.questions)
            }
        else:
            return {"test": True, "chunks_processed": chunks_processed}
    except Exception as e:
        return {"test": True, "error": str(e)}

# --------------------------
# Utility helpers (kept)
# --------------------------
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

def process_query(query_text: str):
    if vectorstore is None:
        return {"error": "Vectorstore not initialized. Upload and process documents first."}

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return {"error": "Missing GROQ API key"}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

    prompt_template = """Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Provide a concise, to-the-point answer in maximum ONE sentence
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

# --------------------------
# Startup event
# --------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup (defensive)."""
    global index, embedding_model
    try:
        pc, idx = initialize_pinecone()
        index = idx
        embedding_model = initialize_embeddings()
        print("✅ Services initialized successfully")
        try:
            stats = index.describe_index_stats()
            print("✅ Pinecone index stats:", getattr(stats, "total_vector_count", "unknown"))
        except Exception:
            print("⚠️ Could not describe index stats at startup (index may be empty).")
    except Exception as e:
        print(f"❌ Failed to initialize services on startup: {e}")

# --------------------------
# Run if executed directly
# --------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="debug"
    )
