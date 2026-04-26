# TOPSIM Assistant

Kontext injection Chat-System fuer das TOPSIM-Planspiel mit React-Frontend und FastAPI-Backend.
Das System kombiniert:

- festen Wissenskontext aus einem Handbuch (`knowledge_base/Handbuch_erweitert.md`)
- Mistral Chat Completions inklusive Tool Calling
- zwei lokale ML-Inferenz-Tools fuer Periode 1
- optionale Observability via Langfuse

## Ziel und Scope

Das Projekt ist ein didaktischer Tutor-Assistent fuer Studierende im Planspiel-Kontext.
Der Assistent erklaert, strukturiert und unterstuetzt bei Entscheidungen, trifft aber keine Entscheidungen fuer Nutzer.

- Unternehmensdaten werden nicht live aus TOPSIM oder einer Datenbank geladen.

## Architektur

```text
React/Vite UI
  -> FastAPI Backend (main.py)
      -> Mistral Chat Completions API
      -> Tool Layer (mistral_tools.py)
          -> Open-Meteo (weather_info)
          -> Joblib ML Modelle (Absatz/Erfolgswert, P1)
      -> Statischer Handbuchkontext (knowledge_base/*.md)
      -> Observability (Langfuse, optional)
```

## Kernfunktionen

- Streaming-Chat ueber WebSocket (`/ws/chat`)
- HTTP-Chat-Endpunkt (`/api/chat`)
- Tool Calling mit `tool_choice=auto`
- Tool-Orchestrierung mit mehrstufiger Tool-Loop im Backend
- Perioden- und Zeitkontext zur Laufzeit aus `settings.py`
- Nutzerfeedback als Score-Event (`/api/feedback`)
- Health-Endpoint inkl. Langfuse-Status (`/api/health`)

## Repository-Struktur

```text
.
|- frontend/
|- knowledge_base/
|- ML_models/
|  |- Potenzieller_Absatz_p1.joblib
|  |- Erfolgswert_p1.joblib
|- deploy/
|- main.py
|- mistral_tools.py
|- observability.py
|- settings.py
|- locustfile.py
|- requirements.txt
|- .env.example
```

### Periodenlogik (`settings.py`)

`PERIOD_DATE_RANGES` definiert statisch Start-/Endfenster der TOPSIM-Perioden.
Diese Information wird in den Runtime-Systemkontext eingebettet.

## Tooling

In `mistral_tools.py` sind aktuell drei Tools registriert:

1. `weather_info`
2. `predict_potentieller_absatz_p1`
3. `predict_erfolgswert_p1`

Die Tool-Loop liegt in `main.py` (`_run_mistral_with_tools`):

- Assistant liefert `tool_calls`
- Backend fuehrt Tool(s) aus
- Tool-Resultate werden als `role="tool"` zur Historie hinzugefuegt
- Folgerunde wird gestartet, bis finale Antwort ohne `tool_calls` vorliegt

## Lasttests

`locustfile.py` enthaelt einen Ramptest fuer Chat-Endpunkte (HTTP oder WebSocket):

- Ziel-URL via `TARGET_URL` (Default: `https://llmroute.de/test/`)
- Lastprofil via `TEST_DURATION_SECONDS`, `PEAK_USERS`, `SPAWN_RATE`
- Optionales Verhalten via Flags wie `COUNT_429_AS_SUCCESS`, `VERIFY_TLS`

Beispiel:

```powershell
locust -f locustfile.py --headless -u 5 -r 1 -t 2m --host https://llmroute.de
```

## Transport-Sicherheit

Die Kommunikation wird bevorzugt durchgaengig ueber TLS 1.3 abgesichert, mit X25519 fuer den Schluesselaustausch und AES_256_GCM als Cipher Suite. Diese Konfiguration bietet einen starken, modernen Standard.

## Betriebsgrenzen und Risiken

- Keine Live-Anbindung an TOPSIM-Datenquellen
- ML-Modelle sind auf Periode 1 und vorgegebene Feature-Schemata begrenzt
- Prognosen sind schaetzend, nicht deterministisch belastbar
- Tool-Ergebnisse koennen durch fehlerhafte Eingaben oder Schemaabweichungen scheitern

Empfehlung fuer Nutzerkommunikation:

- Ergebnisse als Entscheidungshilfe framend darstellen
- bei kritischen Entscheidungen Verifikation durch Tutor/Fachperson einfordern

  ## Voraussetzungen
  
  - Python 3.10+
  - Node.js 18+
  - gueltiger `MISTRAL_API_KEY`

## Lizenz

MIT, siehe `LICENSE`.
