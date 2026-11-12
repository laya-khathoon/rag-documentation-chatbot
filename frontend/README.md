# PageSage frontend

React + Vite single-page app for PageSage. It lets a visitor upload a PDF,
ask questions about it, and shows answers with the page numbers they came
from. It talks to the FastAPI backend in `../backend`.

## Run locally

```bash
npm install
npm run dev
```

The dev server starts on http://localhost:5173 and talks to the FastAPI
backend at http://127.0.0.1:8000 by default. To point it elsewhere, set
`VITE_API_URL` in a `.env` file (in production this is set in the Vercel
project settings).

## Build

```bash
npm run build
```

The production build in `dist/` is what Vercel deploys.
