import { useEffect, useRef, useState } from "react";
import { Streamdown } from "streamdown";

const initialMessages = [
  {
    role: "assistant",
    content:
      "Hi, ich bin Mistral. Wie kann ich dir weiter helfen?",
  },
];

function LoaderBubble() {
  return (
    <div className="message-row assistant">
      <div className="message-bubble loader-bubble" aria-label="Antwort wird geladen">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamStarted, setStreamStarted] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);
  const messagesRef = useRef(null);
  const shellRef = useRef(null);
  const formRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) {
      return;
    }

    let rafId = null;
    let lastX = -100;
    let lastY = -100;
    let isOutsideMessages = false;

    const applyPointerGlow = () => {
      rafId = null;
      shell.style.setProperty("--mouse-x", `${lastX}px`);
      shell.style.setProperty("--mouse-y", `${lastY}px`);
      shell.classList.toggle("pixel-hover-active", isOutsideMessages);
    };

    const onPointerMove = (event) => {
      const messagesEl = messagesRef.current;
      const targetNode = event.target;
      const inMessages =
        messagesEl && targetNode instanceof Node
          ? messagesEl.contains(targetNode)
          : false;

      lastX = event.clientX;
      lastY = event.clientY;
      isOutsideMessages = !inMessages;

      if (rafId === null) {
        rafId = window.requestAnimationFrame(applyPointerGlow);
      }
    };

    const onPointerLeave = () => {
      isOutsideMessages = false;
      if (rafId === null) {
        rafId = window.requestAnimationFrame(applyPointerGlow);
      }
    };

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerleave", onPointerLeave, { passive: true });

    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerleave", onPointerLeave);
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    const content = input.trim();
    if (!content || isLoading) {
      return;
    }

    const nextMessages = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setInput("");
    setError("");
    setIsLoading(true);
    setStreamStarted(false);

    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";

    // Localhost-Variante (zum lokalen Testen) bewusst auskommentiert lassen:
    // const frontendHost = window.location.hostname;
    // const backendHost =
    //   frontendHost === "localhost" || frontendHost === "::1"
    //     ? "127.0.0.1"
    //     : frontendHost;
    // const defaultWsUrl = `${wsProtocol}://${backendHost}:8000/ws/chat`;

    // Deployment unter /test/ auf derselben Domain:
    const defaultWsUrl = `${wsProtocol}://${window.location.host}/test/ws/chat`;
    const wsUrl = import.meta.env.VITE_WS_URL || defaultWsUrl;

    let gotFirstDelta = false;
    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;

    socket.onopen = () => {
      socket.send(
        JSON.stringify({
          messages: nextMessages.map(({ role, content: messageContent }) => ({
            role,
            content: messageContent,
          })),
        }),
      );
    };

    socket.onmessage = (eventMessage) => {
      try {
        const payload = JSON.parse(eventMessage.data);

        if (payload.type === "delta") {
          if (!gotFirstDelta) {
            gotFirstDelta = true;
            setStreamStarted(true);
          }

          setMessages((current) => {
            const chunk = payload.content || "";
            if (!chunk) {
              return current;
            }

            const copy = [...current];
            const last = copy[copy.length - 1];

            if (last && last.role === "assistant") {
              copy[copy.length - 1] = {
                ...last,
                content: `${last.content}${chunk}`,
              };
            } else {
              copy.push({ role: "assistant", content: chunk });
            }

            return copy;
          });
          return;
        }

        if (payload.type === "done") {
          setIsLoading(false);
          setStreamStarted(false);
          socket.close();
          wsRef.current = null;
          return;
        }

        if (payload.type === "error") {
          setError(payload.content || "Beim Streaming ist ein Fehler aufgetreten.");
          setIsLoading(false);
          setStreamStarted(false);
          socket.close();
          wsRef.current = null;
        }
      } catch (_parseError) {
        setError("Ungueltige Antwort vom WebSocket-Server.");
        setIsLoading(false);
        setStreamStarted(false);
        socket.close();
        wsRef.current = null;
      }
    };

    socket.onerror = () => {
      setError(`WebSocket-Verbindung fehlgeschlagen (${wsUrl}).`);
      setIsLoading(false);
      setStreamStarted(false);
    };

    socket.onclose = (eventClose) => {
      if (wsRef.current === socket) {
        wsRef.current = null;
      }
      if (!eventClose.wasClean && !error) {
        setError(
          `WebSocket getrennt (Code ${eventClose.code}${
            eventClose.reason ? `, ${eventClose.reason}` : ""
          }).`,
        );
      }
      setIsLoading(false);
      setStreamStarted(false);
    };
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  }

  return (
    <div ref={shellRef} className="app-shell">
      <div className="pixel-grid" aria-hidden="true" />
      <div className="pixel-hover" aria-hidden="true" />
      <div className="pixel-art-scene" aria-hidden="true">
        <div className="ai-ref-card ref-gd">
          <img src="./gradient-descent.png" alt="" loading="lazy" />
        </div>
        <div className="ai-ref-card ref-nn">
          <img src="./Streudiagramm.png" alt="" loading="lazy" />
        </div>
        <div className="ai-ref-card ref-llm">
          <img src="./Attention_paper.png" alt="" loading="lazy" />
        </div>
        <div className="ai-ref-card ref-weights">
          <img src="./Sigmoid.png" alt="" loading="lazy" />
        </div>
        <div className="ai-ref-card ref-neuronales-netz">
          <img src="./Neoronales_Netz.png" alt="" loading="lazy" />
        </div>
      </div>
      <main className="chat-layout">
        <header className="hero">
          <div className="badge">Live Chat</div>
          <h1>Mistral AI RAG Interface</h1>
          <p>
            Ein simples Chat Interface mit dem du über das Planspiel reden kannst
          </p>
        </header>

        <section className="chat-panel">
          <div ref={messagesRef} className="messages">
            {messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`message-row ${message.role}`}>
                <div className="message-bubble">
                  <div className="message-role">
                    {message.role === "assistant" ? "Tutor Assistant" : "You"}
                  </div>
                  <div className="message-markdown">
                    <Streamdown>{message.content}</Streamdown>
                  </div>
                </div>
              </div>
            ))}

            {isLoading && !streamStarted ? <LoaderBubble /> : null}
            <div ref={messagesEndRef} />
          </div>

          {error ? <div className="error-banner">{error}</div> : null}

          <form ref={formRef} className="composer" onSubmit={handleSubmit}>
            <div className="composer-inner">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Schreibe deine Nachricht..."
                rows={1}
              />
              <button
                type="submit"
                className="send-button"
                aria-label={isLoading ? "Wird geladen" : "Nachricht senden"}
                disabled={isLoading || !input.trim()}
              >
                {isLoading ? (
                  <span className="send-wait">...</span>
                ) : (
                  <>
                    <span className="pixel-arrow" aria-hidden="true" />
                    <span className="sr-only">Senden</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </section>
        <div className="Warningdiv">
          <span className="Warningtext">Generative KI kann Fehler machen. Prüfe die Ausgaben. Du bist weiterhin selber verantwortlich</span>
        </div>
      </main>
    </div>
  );
}

export default App;
