import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mistral_tools import TOOLS_JSON, run_tool
from settings import PERIOD_DATE_RANGES

load_dotenv()


def _configure_logging() -> logging.Logger:
    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv(
        "APP_LOG_FORMAT",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.basicConfig(level=level, format=fmt)
    logger_obj = logging.getLogger("rag.main")
    logger_obj.debug("Logging initialisiert (level=%s).", level_name)
    return logger_obj


logger = _configure_logging()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
HANDBUCH_PATH = os.getenv("HANDBUCH_PATH", "knowledge_base/Handbuch_erweitert.md")

SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent. Du bist Mistral AI in einem von Jan"
    "entwickelten RAG System. Dir stehen Fragen und Unternehmenskennzahlen der letzten "
    "Semester zur Verfuegung, um auf Fragen zu antworten. Du hilfst den Studierenden bei "
    "Planspielentscheidungen, agierst aber als freundlicher Tutor/Lehrer. Du erklaerst und "
    "unterstuetzt, nimmst aber keine Entscheidungen ab. Du agierst als Berater fuer die Geschaeftsfuehrung/die Studierenden."
    "Challange dabei und fördere das Verständnis stelle nich einfach Informationen bereit, sondern stelle Rückfragen und hilf beim lernen."
    "Du antwortest nur mit Wissen aus diesem Kontext, keine anderen Sachen dazu erfinden."
    "Du antwortest nur dann auf Fragen die nicht im Kontext stehen, wenn es um das erklären von Betriebswirtschaftlichen Themen, wie Themen der Einführung in der BWL geht."
    "Antworte immer in gut formatiertem Markdown. " \
    "Wenn eine Information nicht aus diesen Kontext findest, sagst du, dass du es nicht weißt und man es am besten nachlesen soll"
    "Wenn es um spezifische Probleme geht sage immer, dass du Fehler machen kannst und man sich bei dem Tutor melden kann."
    "Wenn explizit nach Wetter gefragt wird, nutze dafuer das Tool weather_info."
    "Wenn nach Absatz-/Erfolgswert-Prognosen fuer Periode 1 gefragt wird, nutze die Tools predict_potentieller_absatz_p1 und predict_erfolgswert_p1. Schreibe keine Interpretation oder Rechenbeispiele dazu."
    "Antworte kurz." 
)

def _load_handbuch_text(path: str) -> str:
    file_path = Path(path)
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


HANDBUCH_TEXT = _load_handbuch_text(HANDBUCH_PATH)
if HANDBUCH_TEXT:
    HANDBUCH_PROMPT = (
        "Das folgende Handbuch ist fester Kontext und muss bei Antworten beruecksichtigt "
        "werden:\n\n"
        f"{HANDBUCH_TEXT}"
    )
else:
    HANDBUCH_PROMPT = (
        f"Das Handbuch unter '{HANDBUCH_PATH}' konnte nicht geladen werden. "
        "Bitte antworte ohne Handbuch-Kontext und weise kurz auf fehlende Datenbasis hin."
    )


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message]


class ChatResponse(BaseModel):
    message: str


app = FastAPI(title="Mistral Live Chat Wrapper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MISTRAL_MODEL,
        "handbuch_path": HANDBUCH_PATH,
        "handbuch_loaded": bool(HANDBUCH_TEXT),
        "handbuch_chars": len(HANDBUCH_TEXT),
    }


def _as_dict(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _extract_content_from_completion(result: Any) -> str:
    data = _as_dict(result)
    choices = data.get("choices", [])
    if not choices and hasattr(result, "choices"):
        choices = getattr(result, "choices", [])
    if not choices:
        return ""

    first_choice = choices[0]
    first_choice_data = _as_dict(first_choice)
    message = first_choice_data.get("message", {})
    message_data = _as_dict(message)
    content = message_data.get("content", "")

    if isinstance(content, list):
        text_chunks = []
        for part in content:
            part_data = _as_dict(part)
            text_chunks.append(part_data.get("text", ""))
        content = "".join(text_chunks)

    if isinstance(content, str):
        return content.strip()
    return ""


def _extract_message_from_completion(result: Any) -> dict:
    data = _as_dict(result)
    choices = data.get("choices", [])
    if not choices and hasattr(result, "choices"):
        choices = getattr(result, "choices", [])
    if not choices:
        return {}
    return _as_dict(_as_dict(choices[0]).get("message", {}))


def _extract_delta_from_stream_event(event: Any) -> str:
    data = _as_dict(event)
    choices = data.get("choices", [])
    if not choices:
        stream_data = data.get("data", {})
        choices = _as_dict(stream_data).get("choices", [])
    if not choices:
        return ""

    first_choice = _as_dict(choices[0])
    delta = _as_dict(first_choice.get("delta", {}))
    content = delta.get("content", "")

    if isinstance(content, list):
        text_chunks = []
        for part in content:
            part_data = _as_dict(part)
            text_chunks.append(part_data.get("text", ""))
        content = "".join(text_chunks)

    if isinstance(content, str):
        return content
    return ""


def _require_api_key() -> None:
    if not MISTRAL_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY is missing. Add it to your environment or .env file.",
        )


def _get_current_period_info(now: datetime) -> str:
    def _parse_date_value(value: str):
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    def _parse_time_value(value: str):
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).time()
            except ValueError:
                continue
        return None

    for period, date_range in PERIOD_DATE_RANGES.items():
        start_raw = ""
        end_raw = ""
        end_time_raw = "23:59"

        if isinstance(date_range, dict):
            start_raw = str(date_range.get("start_date", ""))
            end_raw = str(date_range.get("end_date", ""))
            end_time_raw = str(
                date_range.get("end_uhrzeit", date_range.get("uhrzeit", "23:59"))
            )
        elif isinstance(date_range, tuple) and len(date_range) == 2:
            start_raw, end_raw = date_range

        start_date = _parse_date_value(str(start_raw))
        end_date = _parse_date_value(str(end_raw))
        end_time = _parse_time_value(str(end_time_raw))
        if not start_date or not end_date or not end_time:
            continue

        period_start = datetime.combine(start_date, datetime.min.time(), tzinfo=now.tzinfo)
        period_end = datetime.combine(end_date, end_time, tzinfo=now.tzinfo)

        if period_start <= now <= period_end:
            return (
                f"Es wird aktuell Periode {period} gespielt, welche "
                f"von {start_date.isoformat()} 00:00 bis {end_date.isoformat()} {end_time.strftime('%H:%M')} geht."
            )

    return (
        "Aktuelle Periode: keine Zuordnung gefunden. "
        "Pruefe die Einstellungen in settings.py (PERIOD_DATE_RANGES)."
    )


async def _build_runtime_context_prompt() -> str:
    now = datetime.now().astimezone()
    system_tz_label = now.tzname() or "Systemzeit"

    weekday_names = {
        0: "Montag",
        1: "Dienstag",
        2: "Mittwoch",
        3: "Donnerstag",
        4: "Freitag",
        5: "Samstag",
        6: "Sonntag",
    }
    weekday = weekday_names[now.weekday()]
    date_time_info = (
        f"Heute ist {weekday}, der {now.strftime('%d.%m.%Y')}."
        f"Es ist {now.strftime('%H:%M')} Uhr. Das ist {system_tz_label}."
    )
    period_info = _get_current_period_info(now=now)

    return (
        "Zusaetzliche Informationen fuer dich, wenn du danach gefragt wirst:\n"
        f"- {date_time_info}\n"
        f"- {period_info}"
    )

async def _build_messages(messages: List[Message]) -> List[dict]:
    model_messages = [message.model_dump() for message in messages]
    runtime_context_prompt = await _build_runtime_context_prompt()
    model_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    model_messages.insert(1, {"role": "system", "content": HANDBUCH_PROMPT})
    model_messages.insert(2, {"role": "system", "content": runtime_context_prompt})
    logger.debug(
        "Messages gebaut: incoming=%d, outgoing=%d (inkl. 3 Systemnachrichten).",
        len(messages),
        len(model_messages),
    )
    return model_messages


async def _post_mistral(messages: List[dict], timeout_seconds: float = 60.0) -> dict:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "stream": False,
        "response_format": {"type": "text"},
        "tools": TOOLS_JSON,
        "tool_choice": "auto",
    }
    logger.debug(
        "POST Mistral: model=%s, messages=%d, tools=%d, timeout=%.1fs",
        MISTRAL_MODEL,
        len(messages),
        len(TOOLS_JSON),
        timeout_seconds,
    )

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            MISTRAL_API_URL,
            headers=headers,
            json=payload,
        )
        logger.debug("Mistral response status=%s", response.status_code)
        response.raise_for_status()
        return response.json()


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_chunks = []
        for part in content:
            part_data = _as_dict(part)
            text_chunks.append(part_data.get("text", ""))
        return "".join(text_chunks).strip()
    return ""


async def _run_mistral_with_tools(messages: List[dict], max_rounds: int = 5) -> str:
    working_messages = list(messages)

    for round_idx in range(1, max_rounds + 1):
        logger.debug("Tool-Loop Runde %d gestartet (messages=%d).", round_idx, len(working_messages))
        result = await _post_mistral(messages=working_messages, timeout_seconds=90.0)
        assistant_message = _extract_message_from_completion(result)
        if not assistant_message:
            logger.warning("Leere Assistant-Message in Runde %d.", round_idx)
            return ""

        tool_calls = assistant_message.get("tool_calls") or []
        assistant_content = _normalize_content(assistant_message.get("content", ""))
        logger.debug(
            "Runde %d: tool_calls=%d, assistant_content_len=%d",
            round_idx,
            len(tool_calls),
            len(assistant_content),
        )

        if not tool_calls:
            logger.debug("Finale Antwort in Runde %d ohne weitere Tool-Calls.", round_idx)
            return assistant_content

        history_assistant_message = {
            "role": "assistant",
            "content": assistant_message.get("content", ""),
            "tool_calls": tool_calls,
        }
        working_messages.append(history_assistant_message)

        for tool_call in tool_calls:
            tool_call_data = _as_dict(tool_call)
            function_data = _as_dict(tool_call_data.get("function", {}))
            tool_name = function_data.get("name", "")
            tool_args = function_data.get("arguments", {})
            logger.debug("Tool-Call erkannt: name=%s, call_id=%s", tool_name, tool_call_data.get("id", ""))
            tool_result = run_tool(tool_name=tool_name, arguments_raw=tool_args)
            logger.debug(
                "Tool-Call abgeschlossen: name=%s, result_keys=%s",
                tool_name,
                list(tool_result.keys()) if isinstance(tool_result, dict) else type(tool_result).__name__,
            )

            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_data.get("id", ""),
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

    logger.warning("Maximale Tool-Rundenzahl erreicht (%d).", max_rounds)
    return "Die Tool-Ausfuehrung hat die maximale Rundenzahl erreicht."


async def _stream_mistral_via_http(messages: List[dict], websocket: WebSocket) -> None:
    final_text = await _run_mistral_with_tools(messages=messages)
    if not final_text:
        logger.warning("Kein finaler Text fuer Streaming erzeugt.")
        return

    chunk_size = 120
    logger.debug("Streaming gestartet: text_len=%d, chunk_size=%d", len(final_text), chunk_size)
    for i in range(0, len(final_text), chunk_size):
        await websocket.send_json({"type": "delta", "content": final_text[i : i + chunk_size]})


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    client = getattr(websocket, "client", None)
    logger.info("WebSocket connected: %s", client)

    if not MISTRAL_API_KEY:
        await websocket.send_json(
            {
                "type": "error",
                "content": "MISTRAL_API_KEY is missing. Add it to your environment or .env file.",
            }
        )
        await websocket.close(code=1011)
        return

    try:
        while True:
            payload = await websocket.receive_json()
            request = ChatRequest.model_validate(payload)
            logger.debug("WS request empfangen: messages=%d", len(request.messages))
            messages = await _build_messages(request.messages)

            try:
                await _stream_mistral_via_http(messages=messages, websocket=websocket)
                await websocket.send_json({"type": "done"})
            except Exception as exc:
                logger.exception("Fehler im WS-Request-Loop: %s", exc)
                await websocket.send_json({"type": "error", "content": str(exc)})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", client)
        return
    except Exception as exc:
        logger.exception("Unerwarteter WebSocket-Fehler: %s", exc)
        await websocket.send_json({"type": "error", "content": str(exc)})
        await websocket.close(code=1011)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    _require_api_key()
    logger.info("HTTP /api/chat request: messages=%d", len(request.messages))

    messages = await _build_messages(request.messages)
    try:
        message = await _run_mistral_with_tools(messages=messages, max_rounds=5)
    except httpx.HTTPStatusError as exc:
        logger.exception("HTTPStatusError von Mistral: %s", exc)
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Mistral API error: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("HTTPError von Mistral: %s", exc)
        raise HTTPException(status_code=502, detail=f"Mistral API error: {str(exc)}") from exc
    if not message:
        logger.warning("Leere Antwort von Mistral fuer /api/chat.")
        raise HTTPException(status_code=502, detail="Mistral API returned an empty reply.")

    logger.debug("HTTP /api/chat response_len=%d", len(message))
    return ChatResponse(message=message)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("UVICORN_HOST", "127.0.0.1"),
        port=int(os.getenv("UVICORN_PORT", "8004")),
        reload=os.getenv("UVICORN_RELOAD", "true").lower() == "true",
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )

