
import { useRef, useState } from 'react'
import './App.css'

const API = 'http://127.0.0.1:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState([])
  const fileInput = useRef(null)

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

      const response = await fetch(`${API}/upload-document`, {
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
      const response = await fetch(`${API}/chat`, {
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
          content: `Could not get an answer: ${error.message}. Check that the backend and Ollama are running.`,
          sources: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function useSuggestion(text) {
    setQuestion(text)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand-icon">R</div>

        <div className="brand-text">
          <h1>Documentation Chatbot</h1>
          <p>Ask questions about your technical documents</p>
        </div>

        <span className="status">
          <span className="status-dot" />
          Local AI
        </span>
      </header>

      <main className="chat-container">
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
                useSuggestion('What is the main topic of the document?')
              }
            >
              What is the main topic of the document?
              <span>→</span>
            </button>

            <button
              type="button"
              className="suggestion"
              onClick={() =>
                useSuggestion('Summarize the proposed approach.')
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

                <div className="message-content">{message.content}</div>

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
          Powered by local Ollama · Your questions are processed on your
          computer
        </p>
      </form>
    </div>
  )
}

export default App
