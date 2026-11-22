import json
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

import fitz
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate


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
# Embeddings turn text into numbers so similar passages sit close
# together. We use Google's hosted embedding model, so no model
# needs to be downloaded or run on this server.

class FreeTierEmbeddings(Embeddings):
    """Pace Gemini embedding calls to stay inside the free tier.

    Google's free embedding quota is 100 requests per minute and every
    embedded text counts toward it. Indexing a long PDF used to fire
    hundreds of requests in a few seconds, so the upload failed with a
    429 RESOURCE_EXHAUSTED error. This wrapper keeps a rolling count
    of the last minute, waits when the quota window is full, and
    waits-and-retries (using Google's own retry delay) when a call is
    still rejected.
    """

    REQUESTS_PER_MINUTE = 90  # stay safely under Google's 100
    BATCH_SIZE = 25           # texts per embed call
    MAX_RETRIES = 4           # retries per call after the first try

    def __init__(self, inner):
        self._inner = inner
        self._recent_calls = []  # one timestamp per embedded text

    def _wait_for_room(self, count: int) -> None:
        while True:
            now = time.monotonic()
            self._recent_calls = [
                t for t in self._recent_calls if now - t < 60
            ]
            if len(self._recent_calls) + count <= self.REQUESTS_PER_MINUTE:
                self._recent_calls.extend([time.monotonic()] * count)
                return
            time.sleep(max(60 - (now - self._recent_calls[0]), 1))

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        text = str(exc)
        return "429" in text or "RESOURCE_EXHAUSTED" in text

    @staticmethod
    def _retry_delay(exc: Exception, fallback: float) -> float:
        match = re.search(
            r"retry(?:Delay['\": ]+| in )([0-9.]+)\s*s",
            str(exc),
            re.IGNORECASE,
        )
        return float(match.group(1)) + 2 if match else fallback

    def _call_with_retry(self, func, count: int, first_backoff: float):
        backoff = first_backoff
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                self._wait_for_room(count)
                return func()
            except Exception as exc:
                if attempt >= self.MAX_RETRIES or not self._is_quota_error(exc):
                    raise
                time.sleep(self._retry_delay(exc, backoff))
                backoff = min(backoff * 2, 90)
        raise HTTPException(  # unreachable, keeps type checkers happy
            status_code=503,
            detail="Embedding service is busy. Please try again.",
        )

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            vectors.extend(
                self._call_with_retry(
                    lambda: self._inner.embed_documents(batch),
                    count=len(batch),
                    first_backoff=30,
                )
            )
        return vectors

    def embed_query(self, text):
        return self._call_with_retry(
            lambda: self._inner.embed_query(text),
            count=1,
            first_backoff=20,
        )


embeddings = FreeTierEmbeddings(
    GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
)

# Every visitor's browser sends an X-Session-Id header. Each session gets
# its own Chroma collection, so one visitor's uploaded documents are never
# visible to anyone else. A shared "public_sample" collection holds the
# bundled sample PDF so a brand-new visitor can try a question right away.
SAMPLE_COLLECTION = "public_sample"

# --------------------------------------------------
# SESSIONS
# --------------------------------------------------
# The backend issues unguessable session tokens. Every upload, search
# and chat call must send one in the X-Session-Id header, and each
# token maps to its own private Chroma collection. Tokens expire after
# seven days; the visitor's browser simply asks for a new one.

SESSIONS_FILE = os.path.join(VECTORSTORE_DIR, "sessions.json")
SESSION_LIFETIME = timedelta(days=7)
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_CHUNKS_PER_PDF = 3000
MAX_DOCS_PER_SESSION = 10


def load_sessions() -> dict:
    try:
        with open(SESSIONS_FILE) as sessions_file:
            return json.load(sessions_file)
    except Exception:
        return {}


def save_sessions(sessions: dict) -> None:
    with open(SESSIONS_FILE, "w") as sessions_file:
        json.dump(sessions, sessions_file)


def prune_sessions(sessions: dict) -> dict:
    cutoff = datetime.now(timezone.utc) - SESSION_LIFETIME
    return {
        token: created
        for token, created in sessions.items()
        if datetime.fromisoformat(created) > cutoff
    }


def valid_session(session_id: str | None) -> bool:
    if not session_id:
        return False
    sessions = prune_sessions(load_sessions())
    save_sessions(sessions)
    return session_id in sessions


def require_session(session_id: str | None) -> str:
    if not valid_session(session_id):
        raise HTTPException(
            status_code=401,
            detail="Missing or expired session. Reload the page to start a new one.",
        )
    return session_id


def session_collection_name(session_id: str | None) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", session_id or "")
    if not safe:
        return SAMPLE_COLLECTION
    return f"session_{safe[:48]}"


# Cosine distance: 0 means identical meaning, 2 means opposite.
# Chunks farther than RELEVANCE_LIMIT from the question are treated as
# "not in the document" so the app abstains instead of guessing.
RELEVANCE_LIMIT = float(os.environ.get("RELEVANCE_LIMIT", "0.6"))


def get_store(session_id: str | None) -> Chroma:
    return Chroma(
        collection_name=session_collection_name(session_id),
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )


def ensure_sample_indexed() -> Chroma | None:
    """Index the bundled sample PDF once. Returns the sample store."""
    sample_store = Chroma(
        collection_name=SAMPLE_COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
    if sample_store._collection.count() > 0:
        return sample_store

    pdf_path = os.path.join(DOCUMENTS_DIR, "sample.pdf")
    if not os.path.exists(pdf_path):
        return None

    try:
        with open(pdf_path, "rb") as pdf_file:
            chunks, metadatas, chunk_ids = extract_pdf_chunks(
                pdf_file.read(),
                "sample.pdf",
            )
        sample_store.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=chunk_ids,
        )
    except Exception:
        return None

    return sample_store


# --------------------------------------------------
# 4. ANSWER MODEL
# --------------------------------------------------
# The language model writes the final answer from the retrieved
# chunks. Gemini 3.5 Flash Lite has one of the largest free quotas
# on Google's API (the newer Gemini 3.6 Flash allows only about 20
# free requests per day, which a demo app exhausts in minutes).

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
)


prompt_template = ChatPromptTemplate.from_template(
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


def invoke_llm_with_retry(chain_input):
    """Call the answer model, retrying brief Google-side hiccups.

    Free-tier models sometimes return a short-lived 503 (high demand)
    or a per-minute 429. Wait a moment and retry instead of failing
    the visitor's question. A per-day quota error is not retried:
    it would only succeed after midnight Pacific time.
    """
    last_error = None
    for attempt in range(3):
        try:
            return (prompt_template | llm).invoke(chain_input)
        except Exception as exc:
            last_error = exc
            text = str(exc)
            if "per day" in text.lower():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The free daily limit for answers is reached. "
                        "It resets at midnight Pacific time - please "
                        "try again tomorrow."
                    ),
                ) from exc
            if attempt == 2 or not (
                "503" in text
                or "UNAVAILABLE" in text
                or "429" in text
                or "RESOURCE_EXHAUSTED" in text
            ):
                raise
            time.sleep(5 * (attempt + 1))
    raise last_error


# --------------------------------------------------
# 5. FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="RAG Documentation Chatbot",
    description="A RAG chatbot for technical documentation.",
    version="3.0.0",
)

# Allowed websites that may call this API. The default list covers
# local development and the live Vercel frontend; extra origins can
# be added with the ALLOWED_ORIGINS environment variable
# (comma-separated).
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://rag-documentation-chatbot.vercel.app",
]
extra_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins + extra_origins,
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
# 7. RETRIEVAL
# --------------------------------------------------

def extract_text(content) -> str:
    """LangChain message content can be a plain string or a list of
    content blocks (Gemini 3 returns blocks with signatures). Keep
    only the human-readable text parts."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(part for part in parts if part)


def retrieve_relevant_chunks(
    question: str,
    session_id: str | None,
    result_count: int = 5,
):
    # Find the chunks whose embeddings are closest to the question.
    # Looks in the visitor's own session collection first; if they have
    # not uploaded anything yet, falls back to the shared sample document.
    session_store = get_store(session_id)
    session_has_docs = session_store._collection.count() > 0

    if session_has_docs:
        # The visitor uploaded a PDF: answer only from their documents,
        # and only when a chunk is close enough to the question.
        results = session_store.similarity_search_with_score(
            question,
            k=result_count,
        )
        return [
            (document, score)
            for document, score in results
            if score <= RELEVANCE_LIMIT
        ]

    # Nothing uploaded yet: let them try the bundled sample document.
    sample_store = ensure_sample_indexed()
    if sample_store is None:
        return []
    results = sample_store.similarity_search_with_score(
        question,
        k=result_count,
    )
    return [
        (document, score)
        for document, score in results
        if score <= RELEVANCE_LIMIT
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


@app.post("/session/new")
def new_session():
    sessions = prune_sessions(load_sessions())
    token = "rag_" + secrets.token_urlsafe(32)
    sessions[token] = datetime.now(timezone.utc).isoformat()
    save_sessions(sessions)
    return {"session_id": token, "expires_in_days": SESSION_LIFETIME.days}


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

    get_store(None).add_texts(
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
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
):
    session_id = require_session(x_session_id)

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

        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF is too large. The limit is 10 MB.",
            )

        filename = os.path.basename(file.filename)

        chunks, metadatas, chunk_ids = extract_pdf_chunks(
            pdf_bytes,
            filename,
        )

        if len(chunks) > MAX_CHUNKS_PER_PDF:
            raise HTTPException(
                status_code=413,
                detail="This PDF produced too many text chunks.",
            )

        store = get_store(session_id)
        existing_sources = {
            metadata.get("source")
            for metadata in store.get(include=["metadatas"])["metadatas"]
        }
        if (
            filename not in existing_sources
            and len(existing_sources) >= MAX_DOCS_PER_SESSION
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Session limit reached ({MAX_DOCS_PER_SESSION} documents).",
            )

        store.add_texts(
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
        if FreeTierEmbeddings._is_quota_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Google's free embedding quota is busy right now. "
                    "Please wait a minute and upload again."
                ),
            ) from exc
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
def search_documents(
    query: str,
    x_session_id: str | None = Header(default=None),
):
    session_id = require_session(x_session_id)

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    try:
        results = retrieve_relevant_chunks(
            query,
            session_id,
            result_count=3,
        )

        return {
            "query": query,
            "results": [
                {
                    "text": document.page_content,
                    "source": document.metadata.get("source", "Unknown"),
                    "page": document.metadata.get("page"),
                    "similarity_score": round(float(score), 4),
                }
                for document, score in results
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
def chat(
    request: ChatRequest,
    x_session_id: str | None = Header(default=None),
):
    session_id = require_session(x_session_id)

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
            session_id,
            result_count=5,
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

        response = invoke_llm_with_retry({
            "context": context,
            "question": question,
        })
        answer = extract_text(response.content).strip()

        if answer == fallback_answer:
            # The model found no support in the excerpts: do not show
            # sources for an answer that cites nothing.
            return {
                "question": question,
                "answer": answer,
                "sources": [],
            }

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
            "answer": answer,
            "sources": sources,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Chat failed. Check that the GOOGLE_API_KEY environment "
                "variable is set and the Gemini API is reachable. "
                f"Details: {str(exc)}"
            ),
        ) from exc
