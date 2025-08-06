import os
import time
import tempfile
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

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

# Initialize embeddings
def initialize_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# PDF processing and indexing
def process_pdf_files(pdf_files: List[UploadFile]):
    global vectorstore, index, embedding_model

    vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)
    total_chunks = 0

    for pdf_file in pdf_files:
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
        vectorstore.add_documents(chunks)
        total_chunks += len(chunks)

        os.unlink(tmp_path)

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
                "excerpt": doc.page_content[:300]
            } for doc in result.get("source_documents", [])[:3]
        ]
    }

# API Endpoints

@app.on_event("startup")
def startup_event():
    global index, embedding_model
    pc, index = initialize_pinecone()
    embedding_model = initialize_embeddings()

@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        chunks = process_pdf_files(files)
        return {"message": f"{chunks} document chunks processed successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query/")
async def query_pdf(question: str = Form(...)):
    result = process_query(question)
    return result
