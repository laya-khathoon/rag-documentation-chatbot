
import { useRef, useState } from 'react'
import './App.css'

// Backend address. In production set VITE_API_URL in the Vercel project
// settings; locally it falls back to the FastAPI dev server.
const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// The backend issues each browser an unguessable session token, kept in
// localStorage. Every request sends it in the X-Session-Id header, and
// the token maps to a private document collection for this browser only.
async function ensureSessionId() {
  const stored = window.localStorage.getItem('ragSessionId')
  if (stored) return stored

  const response = await fetch(`${API}/session/new`, { method: 'POST' })
  if (!response.ok) throw new Error('Could not start a session.')

  const data = await response.json()
  window.localStorage.setItem('ragSessionId', data.session_id)
  return data.session_id
}

// One automatic retry with a fresh token when the old one expired.
async function apiFetch(path, options = {}) {
  let sessionId = await ensureSessionId()
  let response = await fetch(`${API}${path}`, {
    ...options,
    headers: { ...(options.headers || {}), 'X-Session-Id': sessionId },
  })

  if (response.status === 401) {
    window.localStorage.removeItem('ragSessionId')
    sessionId = await ensureSessionId()
    response = await fetch(`${API}${path}`, {
      ...options,
      headers: { ...(options.headers || {}), 'X-Session-Id': sessionId },
    })
  }

  return response
}

function renderMessageContent(content) {
  // Render **bold** markers from the model as real bold text.
  // Plain text stays escaped by React, so this is safe.
  const parts = String(content).split(/\*\*([^*]+)\*\*/g)
  return parts.map((part, index) =>
    index % 2 === 1 ? <strong key={index}>{part}</strong> : part
  )
}

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [serverWaking, setServerWaking] = useState(false)
  const fileInput = useRef(null)

  // The free Render backend sleeps when idle and can take about a
  // minute to wake up. If a request takes more than 8 seconds, tell
  // the visitor what is happening instead of looking stuck.
  async function apiFetchWithWakeNotice(path, options = {}) {
    const wakeTimer = setTimeout(() => setServerWaking(true), 8000)
    try {
      return await apiFetch(path, options)
    } finally {
      clearTimeout(wakeTimer)
      setServerWaking(false)
    }
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0]
    event.target.value = ''

    if (!file) return

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Please select a PDF file.')
      setUploadStatus('')
      return
    }

    setUploading(true)
    setUploadError('')
    setUploadStatus(`Uploading ${file.name}...`)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await apiFetchWithWakeNotice('/upload-document', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || `Upload failed (${response.status})`)
      }

      setUploadedFiles((previous) => [
        ...previous.filter((item) => item.filename !== data.filename),
        {
          filename: data.filename,
          chunks: data.total_chunks,
        },
      ])

      setUploadStatus(
        `${data.filename} uploaded successfully · ${data.total_chunks} chunks indexed`
      )
    } catch (error) {
      setUploadStatus('')
      setUploadError(error.message || 'Could not upload the PDF.')
    } finally {
      setUploading(false)
    }
  }

  async function sendMessage(event) {
    event.preventDefault()

    const userQuestion = question.trim()
    if (!userQuestion || loading || uploading) return

    setMessages((previous) => [
      ...previous,
      { role: 'user', content: userQuestion },
    ])

    setQuestion('')
    setLoading(true)

    try {
      const response = await apiFetchWithWakeNotice('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userQuestion }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || `Request failed (${response.status})`)
      }

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: data.answer || 'No answer was returned.',
          sources: Array.isArray(data.sources) ? data.sources : [],
        },
      ])
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: `Could not get an answer: ${error.message}. The backend may be waking up - wait a few seconds and try again.`,
          sources: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function sendSuggestion(text) {
    setQuestion(text)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand-icon">P</div>

        <div className="brand-text">
          <h1>PageSage</h1>
          <p>Ask your PDFs. Find answers with sources.</p>
        </div>

        <span className="status">
          <span className="status-dot" />
          Cloud AI
        </span>
      </header>

      <main className="chat-container">
        {serverWaking && (
          <p
            role="status"
            style={{
              margin: 0,
              padding: '10px 14px',
              borderRadius: '10px',
              background: '#20294a',
              color: '#a5b4fc',
              fontSize: '14px',
              textAlign: 'center',
            }}
          >
            Waking up the server - this can take about a minute on the
            first request. Hang tight!
          </p>
        )}

        <section className="document-panel">
          <div className="document-panel-heading">
            <div>
              <h2>Your documents</h2>
              <p>Upload a PDF to add it to the searchable knowledge base.</p>
            </div>

            <button
              type="button"
              className="upload-button"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
            >
              {uploading ? 'Uploading...' : '+ Upload PDF'}
            </button>

            <input
              ref={fileInput}
              type="file"
              accept=".pdf,application/pdf"
              onChange={uploadDocument}
              hidden
            />
          </div>

          {uploadStatus && (
            <p className="upload-status" role="status">
              {uploadStatus}
            </p>
          )}

          {uploadError && (
            <p className="upload-error" role="alert">
              {uploadError}
            </p>
          )}

          {uploadedFiles.length > 0 && (
            <div className="uploaded-files">
              {uploadedFiles.map((file) => (
                <div className="uploaded-file" key={file.filename}>
                  <span className="pdf-icon">PDF</span>
                  <div className="uploaded-file-details">
                    <strong>{file.filename}</strong>
                    <span>{file.chunks} chunks indexed</span>
                  </div>
                  <span className="file-check" aria-label="Uploaded">
                    ✓
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>

        {messages.length === 0 ? (
          <section className="welcome">
            <div className="welcome-icon">✦</div>
            <h2>Your documentation, explained.</h2>
            <p>
              Upload a PDF, then ask questions about its content.
              Answers include source pages when available.
            </p>

            <button
              type="button"
              className="suggestion"
              onClick={() =>
                sendSuggestion('What is the main topic of the document?')
              }
            >
              What is the main topic of the document?
              <span>→</span>
            </button>

            <button
              type="button"
              className="suggestion"
              onClick={() =>
                sendSuggestion('Summarize the proposed approach.')
              }
            >
              Summarize the proposed approach
              <span>→</span>
            </button>
          </section>
        ) : (
          <section className="messages" aria-live="polite">
            {messages.map((message, index) => (
              <article
                className={`message ${message.role}`}
                key={`${message.role}-${index}`}
              >
                <div className="message-label">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </div>

                <div className="message-content">{renderMessageContent(message.content)}</div>

                {message.role === 'assistant' &&
                  message.sources?.length > 0 && (
                    <div className="sources">
                      <h3 className="sources-title">
                        Sources ({message.sources.length})
                      </h3>

                      <div className="sources-list">
                        {message.sources.map((source, sourceIndex) => {
                          const sourceName =
                            typeof source === 'object' && source !== null
                              ? source.source
                              : 'Retrieved document'

                          const sourcePage =
                            typeof source === 'object' && source !== null
                              ? source.page
                              : null

                          const sourceText =
                            typeof source === 'string'
                              ? source
                              : source?.text || ''

                          return (
                            <div
                              className="source-item"
                              key={`${sourceName}-${sourcePage ?? sourceIndex}-${sourceIndex}`}
                            >
                              <div className="source-heading">
                                <strong>{sourceName || 'Retrieved document'}</strong>
                                {sourcePage != null && (
                                  <span className="source-page">
                                    Page {sourcePage}
                                  </span>
                                )}
                              </div>

                              {sourceText && (
                                <p className="source-text">{sourceText}</p>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}
              </article>
            ))}

            {loading && (
              <div className="loading">
                <span className="loading-dot" />
                Searching documents and generating an answer...
              </div>
            )}
          </section>
        )}
      </main>

      <form className="input-area" onSubmit={sendMessage}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your documentation..."
          rows={2}
          disabled={loading || uploading}
          onKeyDown={(event) => {
            if (
              event.key === 'Enter' &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault()
              event.currentTarget.form.requestSubmit()
            }
          }}
        />

        <button
          type="submit"
          className="send-button"
          disabled={loading || uploading || !question.trim()}
        >
          {loading ? 'Thinking...' : 'Send ↑'}
        </button>

        <p className="input-note">
          Answers by Google Gemini (free tier) · Documents are processed
          by Google's API and may be used to improve its products ·
          Please don't upload private files
        </p>
      </form>
    </div>
  )
}

export default App
