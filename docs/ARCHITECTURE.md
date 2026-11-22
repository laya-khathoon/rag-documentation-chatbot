# Architecture

PageSage answers questions from uploaded PDFs with retrieval-augmented
generation (RAG): relevant passages are found first, and the answer is built
only from those passages, with their page numbers.

## Pieces

- **Frontend** (`frontend/`) - React + Vite single-page chat. Deployed on
  Vercel. It finds the backend through `VITE_API_URL` (set in the Vercel
  project settings) and falls back to `http://127.0.0.1:8000` locally.
- **Backend** (`backend/app/main.py`) - the whole FastAPI app in one file.
  Deployed on Render from `render.yaml`. The free tier sleeps when idle, so
  the first request after a quiet spell can take about a minute; the
  frontend shows a wake-up notice while it waits.
- **Vector store** - Chroma, one collection per session plus a shared
  `public_sample` collection for the bundled sample PDF.

## Request flow

1. The browser asks for a session (`POST /session/new`) and sends the id in
   an `X-Session-Id` header on every later call. Sessions are valid for
   7 days.
2. On upload (`POST /upload-document`), the PDF is split into chunks of
   1000 characters with 200 overlap and embedded with Google's
   `gemini-embedding-001` model into the session's own Chroma collection.
   One visitor's documents are never visible to another visitor.
3. On a question (`POST /chat` or `GET /search`), the question is matched
   against the session's chunks, and Gemini answers using only the matching
   passages. If nothing relevant is found, the API says it could not find
   enough information in the document instead of guessing.

## Limits

- PDF files only, up to 10 MB each
- Up to 3000 chunks per PDF
- Up to 10 documents per session

## CORS

The backend allows the local dev origin (`http://localhost:5173`) and the
live Vercel frontend; extra origins can be added in `main.py`.
