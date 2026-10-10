
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Directory Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTS_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "documents")
)
VECTORSTORE_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "vectorstore")
)


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# ChromaDB Vector Store
# --------------------------------------------------

vectorstore = Chroma(
    collection_name="documentation",
    embedding_function=embeddings,
    persist_directory=VECTORSTORE_DIR
)


# --------------------------------------------------
# Local Language Model
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# --------------------------------------------------
# Text Chunking
# --------------------------------------------------

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="RAG Documentation Chatbot",
    description="A local RAG chatbot for technical documentation.",
    version="1.0.0"
)


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str


# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "RAG Documentation Chatbot API is running!"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# Read and Process PDF
# --------------------------------------------------

@app.get("/read-document")
def read_document():
    pdf_path = os.path.join(DOCUMENTS_DIR, "sample.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="sample.pdf was not found in the documents folder."
        )

    try:
        with fitz.open(pdf_path) as document:
            text = "\n".join(
                page.get_text() for page in document
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the PDF document."
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the PDF."
        )

    chunks = split_text(text)

    if chunks:
        chunk_ids = [
            f"sample_pdf_chunk_{i}"
            for i in range(len(chunks))
        ]

        vectorstore.add_texts(
            texts=chunks,
            ids=chunk_ids
        )

    return {
        "filename": "sample.pdf",
        "total_chunks": len(chunks),
        "message": "Document processed successfully."
    }


# --------------------------------------------------
# Search Endpoint
# --------------------------------------------------

@app.get("/search")
def search_documents(query: str):
    results = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=3
    )

    return {
        "query": query,
        "results": [
            {
                "text": document.page_content,
                "relevance_score": round(score, 4)
            }
            for document, score in results
        ]
    }


# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        # Retrieve document chunks with relevance scores
        results = vectorstore.similarity_search_with_relevance_scores(
            query=question,
            k=3
        )

        fallback_answer = (
            "I could not find enough information "
            "in the provided document."
        )

        if not results:
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": []
            }

        # Initial threshold; validate against relevant and
        # unrelated questions before adjusting it.
        relevance_threshold = 0.0

        relevant_results = [
            (document, score)
            for document, score in results
            if score >= relevance_threshold
        ]

        # Stop before generation if all results are below threshold
        if not relevant_results:
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": []
            }

        # Build context from relevant chunks only
        context = "\n\n".join(
            document.page_content
            for document, score in relevant_results
        )

        if not context.strip():
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": []
            }

        # Prompt for grounded answers
        prompt = ChatPromptTemplate.from_template(
            """You are a technical documentation assistant.

Answer the question using only information in the context
that directly supports the answer.

Rules:
1. If the context does not answer the question, reply exactly:
   "I could not find enough information in the provided document."
2. Never use general knowledge or memory to fill missing information.
3. Do not invent facts, filenames, page numbers, scores, or examples.
4. Answer clearly and avoid repetition.
5. If the context is unrelated to the question, use the fallback answer.

Context:
{context}

Question:
{question}

Answer:"""
        )

        response = (prompt | llm).invoke({
            "context": context,
            "question": question
        })

        return {
            "question": question,
            "answer": response.content,
            "sources": [
                document.page_content
                for document, score in relevant_results
            ]
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The chatbot could not complete the request. "
                "Check that ChromaDB is accessible and Ollama "
                "is running with llama3.2:1b installed."
            )
        ) from exc
