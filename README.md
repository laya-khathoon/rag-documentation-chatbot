# RAG Documentation Chatbot

An AI-powered chatbot that answers user questions based on information retrieved from uploaded documents using Retrieval-Augmented Generation (RAG).

## Features

- Retrieve relevant information from PDF documents.
- Answer questions using document content.
- FastAPI backend for API operations.
- React frontend for user interaction.
- Vector storage for document retrieval.

## Technologies Used

- **Language:** Python, JavaScript
- **Backend:** FastAPI
- **Frontend:** React, Vite
- **Libraries:** LangChain, ChromaDB, Sentence Transformers, PyMuPDF

## Project Structure

```text
rag-documentation-chatbot/
├── backend/
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── api/
│       ├── rag/
│       └── services/
├── frontend/
├── documents/
├── vectorstore/
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/laya-khathoon/rag-documentation-chatbot.git
cd rag-documentation-chatbot
```

### 2. Start the Backend

```bash
cd backend
pip install fastapi uvicorn langchain chromadb sentence-transformers pymupdf
uvicorn app.main:app --reload
```

Backend URL: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

### 3. Start the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed in the terminal.

## Current Progress

- FastAPI backend initialized.
- React frontend created using Vite.
- Required RAG libraries installed.
- Sample PDF added for testing.
- Document retrieval endpoint under testing.

## Future Improvements

- Improve document retrieval accuracy.
- Generate contextual answers from retrieved content.
- Build a user-friendly chatbot interface.
- Support multiple PDF documents.

## Author

Laya Khathoon V
