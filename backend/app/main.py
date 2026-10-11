
import os
import re

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. DIRECTORIES
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DOCUMENTS_DIR = os.path.join(PROJECT_DIR, "documents")
VECTORSTORE_DIR = os.path.join(PROJECT_DIR, "vectorstore")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# --------------------------------------------------
# 2. TEXT SPLITTER
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


# --------------------------------------------------
# 3. EMBEDDINGS AND VECTOR STORE
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="documentation",
    embedding_function=embeddings,
    persist_directory=VECTORSTORE_DIR,
)


# --------------------------------------------------
# 4. LOCAL MODELS
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)

# Downloads this model the first time it is used.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --------------------------------------------------
# 5. FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="RAG Documentation Chatbot",
    description="A local RAG chatbot for technical documentation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


# --------------------------------------------------
# 6. PDF EXTRACTION
# --------------------------------------------------

def extract_pdf_chunks(pdf_bytes: bytes, filename: str):
    chunks = []
    metadatas = []
    chunk_ids = []

    safe_filename = os.path.basename(filename)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_filename)

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The PDF contains no pages.",
                )

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text().strip()

                if not page_text:
                    continue

                for chunk_index, chunk in enumerate(
                    text_splitter.split_text(page_text)
                ):
                    chunks.append(chunk)
                    metadatas.append({
                        "source": safe_filename,
                        "page": page_number,
                    })
                    chunk_ids.append(
                        f"{safe_id}_page_{page_number}_chunk_{chunk_index}"
                    )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid readable PDF.",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text found. Scanned PDFs may require OCR.",
        )

    return chunks, metadatas, chunk_ids


# --------------------------------------------------
# 7. SHARED RETRIEVAL AND RERANKING
# --------------------------------------------------

def retrieve_relevant_chunks(question: str, candidate_count: int = 12):
    # Retrieve a wider group of candidates using embeddings.
    candidates = vectorstore.similarity_search(
        question,
        k=candidate_count,
    )

    if not candidates:
        return []

    # Rerank candidates using a cross-encoder that evaluates
    # the question and each document chunk together.
    pairs = [
        (question, document.page_content)
        for document in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # Keep the best five chunks for the final answer.
    return [
        (document, float(score))
        for document, score in ranked[:5]
    ]


# --------------------------------------------------
# 8. HOME AND HEALTH
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "RAG Documentation Chatbot API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# --------------------------------------------------
# 9. INDEX DEFAULT SAMPLE PDF
# --------------------------------------------------

@app.get("/read-document")
def read_document():
    pdf_path = os.path.join(DOCUMENTS_DIR, "sample.pdf")

    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="sample.pdf was not found in the documents folder.",
        )

    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    chunks, metadatas, chunk_ids = extract_pdf_chunks(
        pdf_bytes,
        "sample.pdf",
    )

    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=chunk_ids,
    )

    return {
        "filename": "sample.pdf",
        "total_chunks": len(chunks),
        "message": "Document processed successfully.",
    }


# --------------------------------------------------
# 10. UPLOAD PDF
# --------------------------------------------------

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        filename = os.path.basename(file.filename)

        chunks, metadatas, chunk_ids = extract_pdf_chunks(
            pdf_bytes,
            filename,
        )

        vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
        )

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "message": "PDF uploaded and indexed successfully.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process the PDF: {str(exc)}",
        ) from exc
    finally:
        await file.close()


# --------------------------------------------------
# 11. SEARCH
# --------------------------------------------------

@app.get("/search")
def search_documents(query: str):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        results = retrieve_relevant_chunks(query, candidate_count=12)

        return {
            "query": query,
            "results": [
                {
                    "text": document.page_content,
                    "source": document.metadata.get("source", "Unknown"),
                    "page": document.metadata.get("page"),
                    "reranker_score": round(score, 4),
                }
                for document, score in results[:3]
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Document search failed: {str(exc)}",
        ) from exc


# --------------------------------------------------
# 12. CHAT
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    fallback_answer = (
        "I could not find enough information in the provided document."
    )

    try:
        results = retrieve_relevant_chunks(
            question,
            candidate_count=12,
        )

        if not results:
            return {
                "question": question,
                "answer": fallback_answer,
                "sources": [],
            }

        context = "\n\n".join(
            (
                f"[Source: {document.metadata.get('source', 'Unknown')}, "
                f"Page: {document.metadata.get('page', 'Unknown')}]\n"
                f"{document.page_content}"
            )
            for document, score in results
        )

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful documentation question-answering assistant.

Answer the user's question using the supplied document excerpts.

Rules:
1. Answer the exact question directly.
2. Use only facts supported by the excerpts.
3. For questions about the document's main topic, title, objectives,
   or proposed method, use relevant title, abstract, introduction,
   and methodology information when present in the excerpts.
4. Do not mistake acknowledgements, declarations, or references
   for the document's main technical content.
5. If the excerpts do not contain enough evidence, say:
   "I could not find enough information in the provided document."
6. Do not invent details or citations.
7. Keep the answer clear and concise.

Document excerpts:
{context}

Question:
{question}

Answer:"""
        )

        response = (prompt | llm).invoke({
            "context": context,
            "question": question,
        })

        sources = [
            {
                "source": document.metadata.get("source", "Unknown"),
                "page": document.metadata.get("page"),
                "text": document.page_content,
            }
            for document, score in results
        ]

        return {
            "question": question,
            "answer": response.content,
            "sources": sources,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Chat failed. Check that the embedding model, "
                "reranker model, ChromaDB, and Ollama are available. "
                f"Details: {str(exc)}"
            ),
        ) from exc
