import os
import gc
import time
import tempfile
import shutil
import requests
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# =============================================================================
# MEMORY MANAGEMENT UTILITIES
# =============================================================================

class MemoryManager:
    """Memory management for low-memory environments"""
    
    def __init__(self):
        self.cleanup_threshold = 400  # MB
    
    def force_cleanup(self):
        """Aggressive memory cleanup"""
        for _ in range(3):
            gc.collect()
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0  # psutil not available
    
    def check_and_cleanup(self):
        """Check memory and cleanup if needed"""
        memory_mb = self.get_memory_usage()
        if memory_mb > self.cleanup_threshold:
            print(f"⚠️ Memory high: {memory_mb:.1f}MB - cleaning up")
            self.force_cleanup()

# Global memory manager
memory_manager = MemoryManager()

def memory_cleanup_decorator(func):
    """Decorator to ensure memory cleanup after function execution"""
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            memory_manager.check_and_cleanup()
    return wrapper

# =============================================================================
# OPTIMIZED FASTAPI APP WITH MEMORY MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle with memory optimization"""
    # Startup
    global index, embedding_model
    try:
        print("🚀 Initializing services with memory optimization...")
        pc, index = initialize_pinecone()
        embedding_model = initialize_embeddings_lazy()  # Lazy initialization
        print("✅ Services initialized successfully")
        memory_manager.force_cleanup()
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
    
    yield
    
    # Shutdown - cleanup
    cleanup_globals()
    memory_manager.force_cleanup()
    print("🧹 App shutdown - memory cleaned")

app = FastAPI(
    title="HackRX RAG API - Memory Optimized", 
    version="1.0.0",
    lifespan=lifespan
)
security = HTTPBearer()

# Pydantic models
class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# =============================================================================
# OPTIMIZED GLOBAL VARIABLES (LAZY LOADING)
# =============================================================================

vectorstore = None
index = None
embedding_model = None
_embedding_model_loaded = False

def cleanup_globals():
    """Clean up global variables"""
    global vectorstore, index, embedding_model, _embedding_model_loaded
    vectorstore = None
    index = None
    embedding_model = None
    _embedding_model_loaded = False
    gc.collect()

# =============================================================================
# MEMORY-OPTIMIZED INITIALIZATION FUNCTIONS
# =============================================================================

def initialize_pinecone():
    """Initialize Pinecone with minimal memory footprint"""
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

def initialize_embeddings_lazy():
    """Initialize embeddings only when needed (lazy loading)"""
    return None  # Will be initialized on first use

def get_embedding_model():
    """Get embedding model with lazy initialization"""
    global embedding_model, _embedding_model_loaded
    
    if not _embedding_model_loaded:
        print("🔄 Loading embedding model (one-time initialization)...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=os.getenv("HF_HOME", "/tmp/huggingface")  # Use temp folder
        )
        _embedding_model_loaded = True
        memory_manager.force_cleanup()
        print("✅ Embedding model loaded")
    
    return embedding_model

# =============================================================================
# MEMORY-OPTIMIZED CORE FUNCTIONS
# =============================================================================

def clear_index():
    """Clear index with memory cleanup"""
    global index
    try:
        if index:
            index.delete(delete_all=True)
            print("✅ Cleared all existing vectors from index")
            time.sleep(2)
            memory_manager.force_cleanup()
    except Exception as e:
        print(f"⚠️ Warning: Could not clear index: {e}")

def download_pdf_streaming(url: str) -> str:
    """Download PDF with streaming to minimize memory usage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print(f"📥 Downloading PDF from: {url}")
        
        with requests.get(url, headers=headers, timeout=60, stream=True) as response:
            response.raise_for_status()
            
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                # Stream in chunks to avoid loading entire file in memory
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                
                print(f"✅ PDF downloaded to: {tmp_file.name}")
                return tmp_file.name
                
    except Exception as e:
        print(f"❌ Failed to download PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def process_pdf_memory_optimized(pdf_url: str):
    """Process PDF with aggressive memory optimization"""
    global vectorstore, index
    
    pdf_path = None
    try:
        # Download PDF with streaming
        pdf_path = download_pdf_streaming(pdf_url)
        
        # Clear existing vectors
        clear_index()
        
        # Get embedding model (lazy loaded)
        embedding_model = get_embedding_model()
        
        # Initialize fresh vectorstore
        vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
        
        # Load PDF pages one by one (memory efficient)
        loader = PyPDFLoader(pdf_path)
        
        # Process in smaller batches to avoid memory spikes
        all_chunks = []
        batch_size = 5  # Process 5 pages at a time
        page_count = 0
        
        for page in loader.lazy_load():  # Lazy loading
            page.metadata.update({
                "source_url": pdf_url,
                "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "page_number": page_count
            })
            
            # Split this page into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks to save memory
                chunk_overlap=80,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
            
            page_chunks = text_splitter.split_documents([page])
            all_chunks.extend(page_chunks)
            page_count += 1
            
            # Process in batches and cleanup frequently
            if len(all_chunks) >= batch_size * 3:  # ~15 chunks
                vectorstore.add_documents(all_chunks[:batch_size * 3])
                print(f"✅ Added batch of {batch_size * 3} chunks (page {page_count})")
                
                # Clear processed chunks from memory
                del all_chunks[:batch_size * 3]
                memory_manager.force_cleanup()
        
        # Process remaining chunks
        if all_chunks:
            vectorstore.add_documents(all_chunks)
            total_chunks = len(all_chunks)
        else:
            total_chunks = page_count * 3  # Estimate
            
        print(f"✅ PDF processed: {page_count} pages, ~{total_chunks} chunks")
        
        # Final cleanup
        del all_chunks
        memory_manager.force_cleanup()
        
        return total_chunks
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise
    finally:
        # Always clean up temp file
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except Exception:
                pass
        memory_manager.force_cleanup()

def process_single_question_optimized(question: str) -> str:
    """Process question with memory optimization"""
    if not vectorstore:
        return "Error: No document processed"
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "Error: GROQ API key not configured"
    
    try:
        # Use fewer retrieved documents to save memory
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Reduced from 8
        
        # Create LLM with memory optimization
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.1,
            max_tokens=512,  # Limit response length to save memory
        )
        
        # Optimized prompt (shorter to save memory)
        prompt_template = """Answer based on the context provided. Be precise and include specific details like numbers, percentages, and conditions.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create QA chain with memory optimization
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,  # Don't return source docs to save memory
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Process question
        result = qa_chain(question)
        answer = result.get("result", "No answer generated")
        
        # Clean up chain from memory
        del qa_chain, llm, retriever
        memory_manager.force_cleanup()
        
        print(f"✅ Question processed: {question[:30]}...")
        return answer
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"❌ {error_msg}")
        memory_manager.force_cleanup()
        return error_msg

# =============================================================================
# OPTIMIZED API ENDPOINTS
# =============================================================================

@app.api_route("/", methods=["GET", "POST", "HEAD", "OPTIONS"])
def read_root():
    return {"message": "HackRX RAG API - Memory Optimized", "status": "healthy"}

@app.api_route("/health", methods=["GET", "POST", "HEAD", "OPTIONS"])
def health_check():
    memory_mb = memory_manager.get_memory_usage()
    return {
        "status": "healthy", 
        "service": "HackRX RAG API", 
        "memory_mb": f"{memory_mb:.1f}",
        "memory_optimized": True
    }

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token"""
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return token

@app.post("/hackrx/run", response_model=HackRXResponse)
@memory_cleanup_decorator
async def hackrx_run(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """Main HackRX endpoint with memory optimization"""
    start_time = time.time()
    
    try:
        print(f"🚀 HackRX request: {len(request.questions)} questions")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Limit questions to prevent memory overload
        if len(request.questions) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 questions allowed")
        
        # Process document with memory optimization
        print(f"📄 Processing document: {request.documents}")
        chunks_processed = process_pdf_memory_optimized(request.documents)
        print(f"✅ Document processed: {chunks_processed} chunks")
        
        # Process questions with memory cleanup between each
        answers = []
        for i, question in enumerate(request.questions):
            if question.strip():
                print(f"🔍 Processing question {i+1}/{len(request.questions)}")
                answer = process_single_question_optimized(question.strip())
                answers.append(answer)
                
                # Force cleanup between questions
                memory_manager.force_cleanup()
            else:
                answers.append("Empty question provided")
        
        processing_time = time.time() - start_time
        print(f"✅ All questions processed in {processing_time:.2f}s")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Internal server error after {processing_time:.2f}s: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# =============================================================================
# MEMORY STATUS AND MONITORING
# =============================================================================

@app.get("/memory-status")
def get_memory_status():
    """Get detailed memory status"""
    memory_mb = memory_manager.get_memory_usage()
    
    return {
        "memory_usage_mb": f"{memory_mb:.1f}",
        "memory_threshold_mb": memory_manager.cleanup_threshold,
        "embedding_model_loaded": _embedding_model_loaded,
        "vectorstore_active": vectorstore is not None,
        "python_objects": len(gc.get_objects()),
        "gc_counts": gc.get_count()
    }

@app.post("/force-cleanup")
def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    initial_memory = memory_manager.get_memory_usage()
    memory_manager.force_cleanup()
    final_memory = memory_manager.get_memory_usage()
    
    return {
        "initial_memory_mb": f"{initial_memory:.1f}",
        "final_memory_mb": f"{final_memory:.1f}",
        "memory_freed_mb": f"{initial_memory - final_memory:.1f}"
    }

# =============================================================================
# SIMPLIFIED ENDPOINTS (MEMORY OPTIMIZED)
# =============================================================================

@app.get("/status")
def get_status():
    """Simplified status endpoint"""
    global vectorstore, index, _embedding_model_loaded
    
    return {
        "status": "running",
        "service": "HackRX RAG API - Memory Optimized",
        "memory_mb": f"{memory_manager.get_memory_usage():.1f}",
        "vectorstore_active": vectorstore is not None,
        "index_active": index is not None,
        "embedding_model_loaded": _embedding_model_loaded,
        "memory_optimized": True
    }

@app.post("/test")
@memory_cleanup_decorator
async def test_endpoint(request: HackRXRequest):
    """Test endpoint with memory optimization"""
    try:
        print("🧪 Test endpoint called")
        chunks_processed = process_pdf_memory_optimized(request.documents)
        
        if request.questions:
            sample_answer = process_single_question_optimized(request.questions[0])
            return {
                "test": True,
                "chunks_processed": chunks_processed,
                "sample_question": request.questions[0],
                "sample_answer": sample_answer,
                "memory_mb": f"{memory_manager.get_memory_usage():.1f}"
            }
        else:
            return {
                "test": True, 
                "chunks_processed": chunks_processed,
                "memory_mb": f"{memory_manager.get_memory_usage():.1f}"
            }
            
    except Exception as e:
        return {"test": True, "error": str(e)}

# =============================================================================
# MINIMAL LEGACY ENDPOINTS (FOR COMPATIBILITY)
# =============================================================================

@app.post("/clear/")
def clear_vectorstore():
    """Clear vectorstore with memory cleanup"""
    try:
        clear_index()
        global vectorstore
        vectorstore = None
        memory_manager.force_cleanup()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============================================================================
# OPTIMIZED STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Memory-optimized uvicorn settings
    port = int(os.environ.get("PORT", 10000))
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker to minimize memory
        loop="asyncio",
        log_level="warning",  # Reduce logging to save memory
        access_log=False,  # Disable access logs
        timeout_keep_alive=5,
        limit_concurrency=50  # Limit concurrent requests
    )
