import json
import os
from pathlib import Path
from typing import Any, List, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
HANDBUCH_PATH = os.getenv("HANDBUCH_PATH", "knowledge_base/Handbuch.md")

SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent. Du bist Mistral AI in einem vom Jan Wobker "
    "entwickelten RAG System. Dir stehen Fragen und Unternehmenskennzahlen der letzten "
    "Semester zur Verfuegung, um auf Fragen zu antworten. Du hilfst den Studierenden bei "
    "Planspielentscheidungen, agierst aber als freundlicher Tutor/Lehrer. Du erklaerst und "
    "unterstuetzt, nimmst aber keine Entscheidungen ab. Agiere eher als Berater fuer die "
    "Geschaeftsfuehrung/die Studierenden. Antworte immer in gut formatiertem Markdown."
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


def _build_messages(messages: List[Message]) -> List[dict]:
    model_messages = [message.model_dump() for message in messages]
    model_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    model_messages.insert(1, {"role": "system", "content": HANDBUCH_PROMPT})
    return model_messages


async def _stream_mistral_via_http(messages: List[dict], websocket: WebSocket) -> None:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "stream": True,
        "response_format": {"type": "text"},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            MISTRAL_API_URL,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise RuntimeError(
                    f"Mistral API error {response.status_code}: {body.decode('utf-8', errors='ignore')}"
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = _extract_delta_from_stream_event(event)
                if delta:
                    await websocket.send_json({"type": "delta", "content": delta})


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()

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
            messages = _build_messages(request.messages)

            try:
                await _stream_mistral_via_http(messages=messages, websocket=websocket)
                await websocket.send_json({"type": "done"})
            except Exception as exc:
                await websocket.send_json({"type": "error", "content": str(exc)})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "content": str(exc)})
        await websocket.close(code=1011)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    _require_api_key()

    messages = _build_messages(request.messages)
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "stream": False,
        "response_format": {"type": "text"},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                MISTRAL_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Mistral API error: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Mistral API error: {str(exc)}") from exc

    message = _extract_content_from_completion(result)
    if not message:
        raise HTTPException(status_code=502, detail="Mistral API returned an empty reply.")

    return ChatResponse(message=message)
