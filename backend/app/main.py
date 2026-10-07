from fastapi import FastAPI

app = FastAPI(
    title="RAG Documentation Chatbot",
    description="A Retrieval-Augmented Generation chatbot for documentation.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "RAG Documentation Chatbot API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }