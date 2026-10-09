
import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  async function sendMessage(e) {
    e.preventDefault()

    const userQuestion = question.trim()
    if (!userQuestion || loading) return

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userQuestion },
    ])
    setQuestion('')
    setLoading(true)

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userQuestion }),
      })

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`)
      }

      const data = await response.json()

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
        },
      ])
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            'Could not connect to the backend. Make sure FastAPI and Ollama are running.',
        },
      ])
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand-icon">R</div>
        <div>
          <h1>Documentation Chatbot</h1>
          <p>Ask questions about your technical documents</p>
        </div>
        <span className="status">
          <span className="status-dot"></span>
          Local AI
        </span>
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <section className="welcome">
            <div className="welcome-icon">✦</div>
            <h2>Your documentation, explained.</h2>
            <p>
              Ask a question and I'll search your indexed PDF and generate an
              answer using its content.
            </p>
            <button
              className="suggestion"
              onClick={() =>
                setQuestion('What does the retriever do?')
              }
            >
              What does the retriever do? <span>→</span>
            </button>
            <button
              className="suggestion"
              onClick={() =>
                setQuestion('What is document chunking?')
              }
            >
              What is document chunking? <span>→</span>
            </button>
          </section>
        ) : (
          <section className="messages">
            {messages.map((message, index) => (
              <article
                className={`message ${message.role}`}
                key={index}
              >
                <div className="message-label">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </div>
                <div className="message-content">
                  {message.content}
                </div>

                {message.sources?.length > 0 && (
                  <details className="sources">
                    <summary>View retrieved sources</summary>
                    {message.sources.map((source, sourceIndex) => (
                      <p key={sourceIndex}>{source}</p>
                    ))}
                  </details>
                )}
              </article>
            ))}
            {loading && (
              <div className="loading">Searching documents and generating an answer...</div>
            )}
          </section>
        )}
      </main>

      <form className="input-area" onSubmit={sendMessage}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documentation..."
          rows="2"
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              e.currentTarget.form.requestSubmit()
            }
          }}
        />
        <button
          type="submit"
          className="send-button"
          disabled={loading || !question.trim()}
        >
          {loading ? 'Thinking...' : 'Send ↑'}
        </button>
        <p className="input-note">
          Powered by local Ollama · Your questions are processed on your computer
        </p>
      </form>
    </div>
  )
}

export default App
