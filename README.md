# RAG System TOPSIM

<p align="center">
  <img src="frontend/public/Frontend1.png" alt="Frontend" width="46%" />
  <img src="frontend/public/Frontend2.png" alt="Frontend" width="46%" />
</p>

Ein schlankes RAG-basiertes Chat-System fuer das TOPSIM-Planspiel.  
Das Projekt kombiniert ein React-Frontend mit einem FastAPI-Backend und bindet festen Wissenskontext aus einem Handbuch bzw. einer Knowledge Base in jede Anfrage an Mistral AI ein.

## Ueberblick

Das System ist darauf ausgelegt, Studierende bei Planspielentscheidungen zu unterstuetzen, ohne Entscheidungen abzunehmen. Antworten werden mit einem vordefinierten System-Prompt und einem geladenen Handbuch-Kontext angereichert und koennen sowohl klassisch per HTTP als auch per WebSocket-Streaming ausgeliefert werden.

## Features

- React-Frontend mit Chat-Oberflaeche und Markdown-Ausgabe
- FastAPI-Backend fuer Chat- und Health-Endpunkte
- WebSocket-Streaming fuer laufende Antworten
- Fester Handbuch-Kontext aus der Knowledge Base
- Konfigurierbares Mistral-Modell ueber Umgebungsvariablen
- Einfache lokale Entwicklung mit Vite und Uvicorn

## Architektur

```text
React/Vite UI -> FastAPI Backend -> Mistral Chat API
                     |
                     -> Handbuch / Knowledge Base als fester Kontext
```

## Projektstruktur

```text
.
|- frontend/                 # React-Frontend mit Vite
|- knowledge_base/           # Fachlicher Kontext / Handbuch
|- main.py                   # FastAPI-Backend
|- requirements.txt          # Python-Abhaengigkeiten
|- .env.example              # Beispiel fuer Umgebungsvariablen
`- README.md
```

## Voraussetzungen

- Python 3.10+
- Node.js 18+
- Ein gueltiger Mistral API Key

## Installation

### 1. Backend einrichten

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Anschliessend die `.env` mit einem gueltigen API-Key befuellen.

### 2. Frontend einrichten

```powershell
cd frontend
npm install
```

## Konfiguration

Die folgenden Umgebungsvariablen werden unterstuetzt:

| Variable | Beschreibung | Standardwert |
| --- | --- | --- |
| `MISTRAL_API_KEY` | API-Key fuer Mistral AI | leer |
| `MISTRAL_MODEL` | Zu verwendendes Chat-Modell | `mistral-small-latest` |
| `HANDBUCH_PATH` | Pfad zur Handbuch-/Kontextdatei | `knowledge_base/Handbuch.md` |

Hinweis: Wenn das Handbuch nicht geladen werden kann, startet das Backend trotzdem, liefert aber Antworten ohne diesen Zusatzkontext.

## Anwendung starten

### Backend starten

```powershell
.venv\Scripts\activate
uvicorn main:app --reload
```

Das Backend ist anschliessend unter `http://127.0.0.1:8000` erreichbar.

### Frontend starten

```powershell
cd frontend
npm run dev
```

Das Frontend laeuft standardmaessig unter `http://localhost:5173`.

## API-Schnittstellen

### Health Check

- `GET /api/health`

Beispielhafte Rueckgabe:

```json
{
  "status": "ok",
  "model": "mistral-small-latest",
  "handbuch_path": "knowledge_base/Handbuch.md",
  "handbuch_loaded": true,
  "handbuch_chars": 12345
}
```

### Chat per HTTP

- `POST /api/chat`

Request:

```json
{
  "messages": [
    { "role": "user", "content": "Wie sollte ich die Preisstrategie bewerten?" }
  ]
}
```

Response:

```json
{
  "message": "..."
}
```

### Chat per WebSocket

- `WS /ws/chat`

Client sendet:

```json
{
  "messages": [
    { "role": "user", "content": "Analysiere die aktuelle Situation." }
  ]
}
```

Server sendet Streaming-Events:

- `{ "type": "delta", "content": "..." }`
- `{ "type": "done" }`
- `{ "type": "error", "content": "..." }`

## Frontend-Hinweis zur WebSocket-URL

Das Frontend unterstuetzt eine explizite WebSocket-Konfiguration ueber `VITE_WS_URL`.  
Ohne diese Variable verwendet die Anwendung aktuell standardmaessig eine Deployment-URL nach dem Muster:

```text
ws(s)://<host>/test/ws/chat
```

Fuer lokale Entwicklung kann `VITE_WS_URL` daher sinnvoll sein, zum Beispiel:

```env
VITE_WS_URL=ws://127.0.0.1:8000/ws/chat
```

## Entwicklungs-Stack

- Backend: FastAPI, httpx, python-dotenv, Uvicorn
- Frontend: React 18, Vite, Streamdown
- KI-Anbindung: Mistral Chat Completions API

## Einsatzkontext

Dieses Projekt ist als Tutor- und Beratungsoberflaeche fuer ein TOPSIM-/Planspiel-Szenario ausgelegt. Die Antworten sollen fachlich unterstuetzen, aber keine Managemententscheidung automatisiert treffen. Die Verantwortung fuer die Bewertung und Umsetzung bleibt bei den Nutzenden.

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).
