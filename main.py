import os
import time
import tempfile
import shutil
import requests
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI(title="HackRX RAG API", version="1.0.0")
security = HTTPBearer()

# Pydantic models for request/response (matching HackRX spec exactly)
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF document
    questions: List[str]  # Array of questions

class HackRXResponse(BaseModel):
    answers: List[str]  # Array of answers corresponding to questions

# Root endpoint - handle all HTTP methods for health checks
@app.api_route("/", methods=["GET", "POST", "HEAD", "OPTIONS"])
def read_root():
    return {"message": "HackRX RAG API is running!", "status": "healthy"}

# Health check endpoint
@app.api_route("/health", methods=["GET", "POST", "HEAD", "OPTIONS"])
def health_check():
    return {"status": "healthy", "service": "HackRX RAG API"}

# Globals
vectorstore = None
index = None
embedding_model = None

def initialize_pinecone():
    """Initialize Pinecone with the exact same settings as your original code"""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing Pinecone API key")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "retrieval"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,  # Matching your bge-large-en-v1.5 dimensions
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
    return pc, pc.Index(index_name)

def initialize_embeddings():
    """Initialize embeddings with the exact same model as your original code"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def clear_index():
    """Clear all existing vectors from index - same as your original function"""
    global index
    try:
        index.delete(delete_all=True)
        print("✅ Cleared all existing vectors from index")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Warning: Could not clear index: {e}")

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return temporary file path"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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

def process_pdf_from_url(pdf_url: str):
    """Process PDF from URL using your existing logic"""
    global vectorstore, index, embedding_model
    
    # Download PDF
    pdf_path = download_pdf_from_url(pdf_url)
    
    try:
        # Clear existing vectors (same as your upload function)
        clear_index()
        
        # Initialize fresh vectorstore
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✅ PDF loaded: {len(docs)} pages")
        
        # Add metadata (same format as your original)
        for doc in docs:
            doc.metadata.update({
                "source_url": pdf_url,
                "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Split text (exact same settings as your original)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"✅ Text split into {len(chunks)} chunks")
        
        # Add to vectorstore
        vectorstore.add_documents(chunks)
        print(f"✅ Added {len(chunks)} chunks to vectorstore")
        
        return len(chunks)
        
    finally:
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)

def process_single_question(question: str) -> str:
    """Process a single question using your existing logic but optimized for HackRX"""
    if not vectorstore:
        return "Error: No document processed"
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: GROQ API key not configured"
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Same as your original
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.1  # Lower temperature for more consistent answers
        )
        
        # Updated prompt for better HackRX-style answers (more comprehensive than one sentence)
        prompt_template = """Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Provide a precise, comprehensive answer based strictly on the document content
- Include specific numbers, percentages, time periods, conditions, and exact details mentioned in the documents
- Be accurate with numerical values (e.g., "36 months", "1% of Sum Insured", "5% discount")
- If the answer involves conditions or limitations, include them in the response
- Provide complete information while being concise and direct
- If you cannot find the specific information, say "Information not available in the provided document"
- Do not make assumptions or add information not present in the context

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

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token - accepting any non-empty token for HackRX"""
    if not credentials.token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    print(f"✅ Token verified: {credentials.token[:10]}...")
    return credentials.token

# Main HackRX endpoint - EXACTLY as specified
@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main HackRX endpoint that processes documents and answers questions
    Matches the exact specification provided in requirements
    """
    start_time = time.time()
    
    try:
        print(f"🚀 HackRX request received with {len(request.questions)} questions")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Process the document
        print(f"📄 Processing document: {request.documents}")
        chunks_processed = process_pdf_from_url(request.documents)
        print(f"✅ Document processed: {chunks_processed} chunks")
        
        # Process all questions
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

# Keep your original endpoints for development/testing
@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        chunks = process_pdf_files(files)  # assumes you handle multiple UploadFile instances
        stats = get_index_stats()
        return {
            "message": f"{chunks} document chunks processed successfully",
            "files_processed": [f.filename for f in files],
            "index_stats": stats
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/query/")
async def query_pdf(question: str = Form(...)):
    """Your original query endpoint for development"""
    result = process_query(question)
    return result

# Status endpoint
@app.get("/status")
def get_status():
    """Enhanced status endpoint with all necessary info"""
    global vectorstore, index, embedding_model
    
    # Get index stats
    try:
        stats = index.describe_index_stats() if index else None
    except:
        stats = None
    
    return {
        "status": "running",
        "service": "HackRX RAG API",
        "vectorstore_initialized": vectorstore is not None,
        "index_initialized": index is not None,
        "embedding_model_initialized": embedding_model is not None,
        "index_stats": {
            "total_vectors": stats.total_vector_count if stats else 0,
            "dimension": stats.dimension if stats else None,
            "index_fullness": stats.index_fullness if stats else None
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

# Test endpoint for development (no auth required)
@app.post("/test")
async def test_endpoint(request: HackRXRequest):
    """Test endpoint without authentication for development"""
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

# Include your original helper functions
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
    """Your original process_query function for backward compatibility"""
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

def process_pdf_files(pdf_files: List[UploadFile]):
    """Your original process_pdf_files function for backward compatibility"""
    global vectorstore, index, embedding_model

    clear_index()
    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(pdf_file.file, tmp)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

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
        
        vectorstore.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"Added {len(chunks)} chunks from {pdf_file.filename}")

        os.unlink(tmp_path)

    print(f"✅ Total chunks processed: {total_chunks}")
    return total_chunks

@app.post("/clear/")
def clear_vectorstore():
    """Your original clear function"""
    try:
        clear_index()
        global vectorstore
        vectorstore = None
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global index, embedding_model
    try:
        pc, index = initialize_pinecone()
        embedding_model = initialize_embeddings()
        print("✅ Services initialized successfully")
        print(f"✅ Pinecone index ready: {index.describe_index_stats()}")
        print("✅ HackRX API ready to process requests")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  
    uvicorn.run("main:app", host="0.0.0.0", port=port)


