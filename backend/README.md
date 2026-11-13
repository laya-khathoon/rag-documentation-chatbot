# PageSage backend

FastAPI service behind PageSage. It reads uploaded PDFs, splits them into chunks, stores them in a Chroma vector store and answers questions using Google's Gemini API. The React app in `../frontend` talks to it.

## How it works

1. A visitor's browser asks for a session (`POST /session/new`) and sends the returned id in an `X-Session-Id` header on later calls.
2. Uploaded PDFs are split into chunks (1000 characters, 200 overlap) and embedded with Google's `gemini-embedding-001` model.
3. Each session has its own Chroma collection, so one visitor's documents are never visible to another. A shared `public_sample` collection holds the bundled sample PDF.
4. A question is matched against the chunks, and the answer is built only from the matching passages, with page sources. If nothing relevant is found, the API says it could not find enough information in the document.

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/` | Simple "API is running" message |
| GET | `/health` | Health check (`{"status": "healthy"}`) |
| POST | `/session/new` | Creates a session id (valid for 7 days) |
| POST | `/upload-document` | Uploads a PDF (needs `X-Session-Id`) |
| GET | `/search?query=...` | Returns the most relevant chunks (needs `X-Session-Id`) |
| POST | `/chat` | Answers a question from the PDF (needs `X-Session-Id`) |

## Limits

- PDF files only, up to 10 MB each
- Up to 3000 chunks per PDF
- Up to 10 documents per session

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
cp .env.example .env             # then put your own GOOGLE_API_KEY in .env
uvicorn app.main:app --reload
```

The API starts on http://127.0.0.1:8000. FastAPI also serves interactive docs at http://127.0.0.1:8000/docs.

## Environment variables

See `.env.example`.

- `GOOGLE_API_KEY` (required): free key from Google AI Studio
- `RELEVANCE_LIMIT` (optional): minimum similarity score for a chunk, default `0.6`
- `ALLOWED_ORIGINS` (optional): comma-separated frontend URLs allowed by CORS

Never commit a real `.env` file or API key.
