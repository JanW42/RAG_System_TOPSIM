import json
import logging
import os
import random
import asyncio
import time
import uuid
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, List, Literal

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from mistral_tools import TOOLS_JSON, run_tool
from observability import (
    LANGFUSE_ENABLED,
    create_score,
    flush_observability,
    get_current_trace_id,
    initialize_observability,
    langfuse_auth_ok,
    observe,
    trace_attributes,
    update_current_generation,
    update_current_span,
)
from settings import PERIOD_DATE_RANGES


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
MISTRAL_MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "4"))
MISTRAL_RETRY_BASE_DELAY_SECONDS = float(os.getenv("MISTRAL_RETRY_BASE_DELAY_SECONDS", "1.5"))
MISTRAL_INPUT_COST_PER_MILLION_EUR = float(
    os.getenv("MISTRAL_INPUT_COST_PER_MILLION_EUR", "0.5")
)
MISTRAL_OUTPUT_COST_PER_MILLION_EUR = float(
    os.getenv("MISTRAL_OUTPUT_COST_PER_MILLION_EUR", "1.5")
)
EUR_TO_USD_RATE = float(os.getenv("EUR_TO_USD_RATE", "1.08"))

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
    trace_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1)
    score_name: str = Field(default="user_feedback", min_length=1)
    value: float
    comment: str | None = None


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
        "langfuse_enabled": LANGFUSE_ENABLED,
        "langfuse_auth_ok": langfuse_auth_ok(),
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


def _extract_finish_reason_from_stream_event(event: Any) -> str:
    data = _as_dict(event)
    choices = data.get("choices", [])
    if not choices:
        stream_data = data.get("data", {})
        choices = _as_dict(stream_data).get("choices", [])
    if not choices:
        return ""
    return str(_as_dict(choices[0]).get("finish_reason", "") or "")


def _extract_usage_from_completion(result: Any) -> dict:
    data = _as_dict(result)
    usage = _as_dict(data.get("usage", {}))
    if not usage:
        return {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    usage_details: dict[str, Any] = {}
    if isinstance(input_tokens, (int, float)):
        usage_details["input"] = int(input_tokens)
    elif isinstance(prompt_tokens, (int, float)):
        usage_details["input"] = int(prompt_tokens)

    if isinstance(output_tokens, (int, float)):
        usage_details["output"] = int(output_tokens)
    elif isinstance(completion_tokens, (int, float)):
        usage_details["output"] = int(completion_tokens)

    if isinstance(total_tokens, (int, float)):
        usage_details["total"] = int(total_tokens)

    return usage_details


def _extract_usage_from_stream_event(event: Any) -> dict:
    data = _as_dict(event)
    usage = _as_dict(data.get("usage", {}))
    if not usage:
        usage = _as_dict(_as_dict(data.get("data", {})).get("usage", {}))
    if not usage:
        return {}

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")

    parsed: dict[str, Any] = {}
    if isinstance(input_tokens, (int, float)):
        parsed["input"] = int(input_tokens)
    if isinstance(output_tokens, (int, float)):
        parsed["output"] = int(output_tokens)
    if isinstance(total_tokens, (int, float)):
        parsed["total"] = int(total_tokens)
    return parsed


def _estimate_mistral_cost_eur(usage_details: dict) -> dict:
    input_tokens = int(usage_details.get("input", 0) or 0)
    output_tokens = int(usage_details.get("output", 0) or 0)
    input_cost = (input_tokens / 1_000_000) * MISTRAL_INPUT_COST_PER_MILLION_EUR
    output_cost = (output_tokens / 1_000_000) * MISTRAL_OUTPUT_COST_PER_MILLION_EUR
    total_cost = input_cost + output_cost
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_eur": round(input_cost, 10),
        "output_cost_eur": round(output_cost, 10),
        "total_cost_eur": round(total_cost, 10),
        "pricing": {
            "input_per_million_eur": MISTRAL_INPUT_COST_PER_MILLION_EUR,
            "output_per_million_eur": MISTRAL_OUTPUT_COST_PER_MILLION_EUR,
            "model": MISTRAL_MODEL,
        },
    }


def _build_langfuse_cost_details_usd(usage_details: dict) -> dict:
    # Langfuse cost_details are interpreted as USD for built-in cost analytics.
    input_tokens = int(usage_details.get("input", 0) or 0)
    output_tokens = int(usage_details.get("output", 0) or 0)

    input_cost_eur = (input_tokens / 1_000_000) * MISTRAL_INPUT_COST_PER_MILLION_EUR
    output_cost_eur = (output_tokens / 1_000_000) * MISTRAL_OUTPUT_COST_PER_MILLION_EUR
    input_cost_usd = input_cost_eur * EUR_TO_USD_RATE
    output_cost_usd = output_cost_eur * EUR_TO_USD_RATE

    return {
        "input": round(input_cost_usd, 10),
        "output": round(output_cost_usd, 10),
        "total": round(input_cost_usd + output_cost_usd, 10),
    }


def _require_api_key() -> None:
    if not MISTRAL_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY is missing. Add it to your environment or .env file.",
        )


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        retry_after = float(value.strip())
        if retry_after >= 0:
            return retry_after
    except ValueError:
        return None
    return None


def _compute_429_backoff(attempt: int, retry_after_value: str | None) -> float:
    retry_after = _parse_retry_after_seconds(retry_after_value)
    if retry_after is not None:
        return retry_after
    base = MISTRAL_RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, 0.75)
    return min(30.0, base + jitter)


def _build_fallback_user_id(client_host: str | None) -> str:
    if client_host:
        return f"anonymous-{client_host}"
    return "anonymous"


def _resolve_http_user_and_session_id(http_request: Request) -> tuple[str, str]:
    client_host = http_request.client.host if http_request.client else None
    user_id = (
        http_request.headers.get("x-user-id")
        or http_request.query_params.get("user_id")
        or _build_fallback_user_id(client_host)
    )
    session_id = (
        http_request.headers.get("x-session-id")
        or http_request.query_params.get("session_id")
        or f"http-{uuid.uuid4()}"
    )
    return user_id, session_id


def _resolve_ws_user_and_session_id(websocket: WebSocket) -> tuple[str, str]:
    client = getattr(websocket, "client", None)
    client_host = client.host if client else None
    user_id = (
        websocket.headers.get("x-user-id")
        or websocket.query_params.get("user_id")
        or _build_fallback_user_id(client_host)
    )
    session_id = (
        websocket.headers.get("x-session-id")
        or websocket.query_params.get("session_id")
        or f"ws-{uuid.uuid4()}"
    )
    return user_id, session_id


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


@observe(name="build-runtime-context-prompt", as_type="span", capture_output=False)
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

@observe(name="build-messages", as_type="span", capture_output=False)
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


@observe(name="mistral-chat-completion", as_type="generation", capture_input=False)
async def _post_mistral(messages: List[dict], timeout_seconds: float = 60.0) -> dict:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
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

    request_started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for attempt in range(1, MISTRAL_MAX_RETRIES + 1):
            response = await client.post(
                MISTRAL_API_URL,
                headers=headers,
                json=payload,
            )
            logger.debug("Mistral response status=%s (attempt=%d)", response.status_code, attempt)
            if response.status_code == 429 and attempt < MISTRAL_MAX_RETRIES:
                wait_seconds = _compute_429_backoff(
                    attempt=attempt,
                    retry_after_value=response.headers.get("retry-after"),
                )
                logger.warning(
                    "Mistral rate limit (429) in _post_mistral. Retry in %.2fs (attempt=%d/%d).",
                    wait_seconds,
                    attempt,
                    MISTRAL_MAX_RETRIES,
                )
                await asyncio.sleep(wait_seconds)
                continue

            response.raise_for_status()
            if not response.content:
                raise httpx.HTTPError("Mistral API returned an empty response body.")
            try:
                data = response.json()
            except JSONDecodeError as exc:
                content_type = response.headers.get("content-type", "")
                body_preview = response.text[:300].replace("\n", " ").strip()
                logger.error(
                    "Mistral response is not valid JSON (status=%s, content_type=%s, body_preview=%s)",
                    response.status_code,
                    content_type,
                    body_preview,
                )
                raise httpx.HTTPError(
                    "Mistral API returned non-JSON response. "
                    f"status={response.status_code}, content_type={content_type}"
                ) from exc
            usage_details = _extract_usage_from_completion(data)
            cost_estimate = _estimate_mistral_cost_eur(usage_details)
            cost_details = _build_langfuse_cost_details_usd(usage_details)
            latency_ms = int((time.perf_counter() - request_started) * 1000)
            update_current_generation(
                model=MISTRAL_MODEL,
                input=messages,
                metadata={
                    "status_code": response.status_code,
                    "attempt": attempt,
                    "latency_ms": latency_ms,
                    "mistral_request_id": response.headers.get("x-request-id", ""),
                    "mistral_processing_ms": response.headers.get("x-processing-ms", ""),
                    "cost_estimate_eur": cost_estimate,
                    "mode": "non_stream",
                },
                usage_details=usage_details,
                cost_details=cost_details,
                output=_extract_content_from_completion(data),
            )
            return data

    raise httpx.HTTPError("Mistral API request failed after retries.")


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


@observe(name="mistral-tool-loop", as_type="agent", capture_input=False)
async def _run_mistral_with_tools(messages: List[dict], max_rounds: int = 5) -> str:
    working_messages = list(messages)
    tool_call_count = 0
    tool_names: list[str] = []

    for round_idx in range(1, max_rounds + 1):
        update_current_span(metadata={"round_index": round_idx})
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
            tool_call_count += 1
            if tool_name:
                tool_names.append(tool_name)
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
            update_current_span(
                metadata={
                    "tool_call_count": tool_call_count,
                    "tool_names": tool_names,
                }
            )

    logger.warning("Maximale Tool-Rundenzahl erreicht (%d).", max_rounds)
    return "Die Tool-Ausfuehrung hat die maximale Rundenzahl erreicht."


@observe(name="ws-stream-response", as_type="generation", capture_input=False, capture_output=False)
async def _stream_mistral_via_http(messages: List[dict], websocket: WebSocket) -> None:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "stream": True,
        "response_format": {"type": "text"},
        "tools": TOOLS_JSON,
        "tool_choice": "auto",
    }

    got_stream_text = False
    saw_tool_call_finish = False
    streamed_output_chunks: list[str] = []
    stream_usage_details: dict[str, Any] = {}
    stream_attempt = 0
    stream_status_code = 0
    stream_request_id = ""
    request_started = time.perf_counter()

    async with httpx.AsyncClient(timeout=90.0) as client:
        stream_error: Exception | None = None
        for attempt in range(1, MISTRAL_MAX_RETRIES + 1):
            stream_attempt = attempt
            try:
                async with client.stream(
                    "POST",
                    MISTRAL_API_URL,
                    headers=headers,
                    json=payload,
                ) as response:
                    stream_status_code = response.status_code
                    stream_request_id = response.headers.get("x-request-id", "")
                    logger.debug(
                        "Mistral stream response status=%s (attempt=%d)",
                        response.status_code,
                        attempt,
                    )
                    if response.status_code == 429 and attempt < MISTRAL_MAX_RETRIES:
                        wait_seconds = _compute_429_backoff(
                            attempt=attempt,
                            retry_after_value=response.headers.get("retry-after"),
                        )
                        logger.warning(
                            "Mistral rate limit (429) in stream. Retry in %.2fs (attempt=%d/%d).",
                            wait_seconds,
                            attempt,
                            MISTRAL_MAX_RETRIES,
                        )
                        await asyncio.sleep(wait_seconds)
                        continue

                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw_data = line[5:].strip()
                        if not raw_data or raw_data == "[DONE]":
                            continue

                        try:
                            event = json.loads(raw_data)
                        except json.JSONDecodeError:
                            logger.debug("Ungueltiges SSE-Event ignoriert: %s", raw_data[:120])
                            continue

                        event_usage = _extract_usage_from_stream_event(event)
                        if event_usage:
                            stream_usage_details = event_usage

                        finish_reason = _extract_finish_reason_from_stream_event(event)
                        if finish_reason == "tool_calls":
                            saw_tool_call_finish = True
                            break

                        delta = _extract_delta_from_stream_event(event)
                        if delta:
                            got_stream_text = True
                            streamed_output_chunks.append(delta)
                            await websocket.send_json({"type": "delta", "content": delta})
                stream_error = None
                break
            except (httpx.HTTPStatusError, httpx.HTTPError) as exc:
                stream_error = exc
                if attempt >= MISTRAL_MAX_RETRIES:
                    break
                wait_seconds = _compute_429_backoff(attempt=attempt, retry_after_value=None)
                logger.warning(
                    "Stream request failed (%s). Retry in %.2fs (attempt=%d/%d).",
                    type(exc).__name__,
                    wait_seconds,
                    attempt,
                    MISTRAL_MAX_RETRIES,
                )
                await asyncio.sleep(wait_seconds)
        if stream_error is not None:
            raise stream_error

    if saw_tool_call_finish:
        logger.debug("Tool-Call im Stream erkannt, fallback auf Tool-Loop.")
        final_text = await _run_mistral_with_tools(messages=messages)
        if final_text:
            latency_ms = int((time.perf_counter() - request_started) * 1000)
            cost_estimate = _estimate_mistral_cost_eur(stream_usage_details)
            cost_details = _build_langfuse_cost_details_usd(stream_usage_details)
            update_current_generation(
                model=MISTRAL_MODEL,
                input=messages,
                usage_details=stream_usage_details,
                cost_details=cost_details,
                metadata={
                    "stream": True,
                    "fallback_tool_loop": True,
                    "latency_ms": latency_ms,
                    "attempt": stream_attempt,
                    "status_code": stream_status_code,
                    "mistral_request_id": stream_request_id,
                    "cost_estimate_eur": cost_estimate,
                    "mode": "stream_then_tool_loop",
                },
                output=final_text,
            )
            await websocket.send_json({"type": "delta", "content": final_text})
        return

    if not got_stream_text:
        logger.warning("Stream lieferte keinen Text-Delta-Output.")
    else:
        latency_ms = int((time.perf_counter() - request_started) * 1000)
        combined_output = "".join(streamed_output_chunks)
        cost_estimate = _estimate_mistral_cost_eur(stream_usage_details)
        cost_details = _build_langfuse_cost_details_usd(stream_usage_details)
        update_current_generation(
            model=MISTRAL_MODEL,
            input=messages,
            usage_details=stream_usage_details,
            cost_details=cost_details,
            metadata={
                "stream": True,
                "fallback_tool_loop": False,
                "latency_ms": latency_ms,
                "attempt": stream_attempt,
                "status_code": stream_status_code,
                "mistral_request_id": stream_request_id,
                "output_chars": len(combined_output),
                "cost_estimate_eur": cost_estimate,
                "mode": "stream",
            },
            output=combined_output,
        )


@app.websocket("/ws/chat")
@observe(name="ws-chat", as_type="span", capture_output=False)
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
            user_id, session_id = _resolve_ws_user_and_session_id(websocket)
            with trace_attributes(
                session_id=session_id,
                user_id=user_id,
                trace_name="ws-chat",
                metadata={"endpoint": "ws_chat"},
            ):
                messages = await _build_messages(request.messages)

                try:
                    await _stream_mistral_via_http(messages=messages, websocket=websocket)
                    await websocket.send_json(
                        {
                            "type": "done",
                            "trace_id": get_current_trace_id(),
                            "user_id": user_id,
                            "session_id": session_id,
                        }
                    )
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
@observe(name="http-chat", as_type="span", capture_output=False)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    _require_api_key()
    logger.info("HTTP /api/chat request: messages=%d", len(request.messages))

    user_id, session_id = _resolve_http_user_and_session_id(http_request)
    with trace_attributes(
        session_id=session_id,
        user_id=user_id,
        trace_name="http-chat",
        metadata={"endpoint": "api_chat"},
    ):
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
    return ChatResponse(
        message=message,
        trace_id=get_current_trace_id(),
        user_id=user_id,
        session_id=session_id,
    )


@app.post("/api/feedback")
@observe(name="http-feedback", as_type="span", capture_output=False)
async def feedback(request: FeedbackRequest) -> dict:
    create_score(
        name=request.score_name,
        value=request.value,
        trace_id=request.trace_id,
        data_type="NUMERIC",
        comment=request.comment,
    )
    flush_observability()
    return {"status": "ok"}


@app.on_event("shutdown")
async def on_shutdown() -> None:
    flush_observability()


@app.on_event("startup")
async def on_startup() -> None:
    initialize_observability()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("UVICORN_HOST", "127.0.0.1"),
        port=int(os.getenv("UVICORN_PORT", "8004")),
        reload=os.getenv("UVICORN_RELOAD", "true").lower() == "true",
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )

