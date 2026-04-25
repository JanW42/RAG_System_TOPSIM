from __future__ import annotations

import json
import os
import random
import ssl
import time
import uuid
from urllib.parse import urlparse

from locust import HttpUser, LoadTestShape, between, events, task

# Ziel-URL aus der Anforderung
TARGET_URL = os.getenv("TARGET_URL", "https://llmroute.de/test/")

# Chat-Endpunkt kann WS oder HTTP sein (in main.py existieren /ws/chat und /api/chat)
# Beispiele:
#   $env:CHAT_PATH="/test/ws/chat"
#   $env:CHAT_PATH="/test/api/chat"
CHAT_PATH = os.getenv("CHAT_PATH", "/test/ws/chat")
VERIFY_TLS = os.getenv("VERIFY_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
INCLUDE_PAGE_GET = os.getenv("INCLUDE_PAGE_GET", "false").strip().lower() in {"1", "true", "yes", "on"}
COUNT_429_AS_SUCCESS = os.getenv("COUNT_429_AS_SUCCESS", "false").strip().lower() in {"1", "true", "yes", "on"}

# Testdauer / Peak laut Vorgabe
TEST_DURATION_SECONDS = int(os.getenv("TEST_DURATION_SECONDS", "120"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "5"))
SPAWN_RATE = float(os.getenv("SPAWN_RATE", "1"))  # sanfter Anstieg
WAIT_MIN_SECONDS = float(os.getenv("WAIT_MIN_SECONDS", "5"))
WAIT_MAX_SECONDS = float(os.getenv("WAIT_MAX_SECONDS", "10"))

# Beispielprompts für den KI-Chat
PROMPTS = [
    "Hallo! Bitte antworte kurz: Was ist ein Ramptest?",
    "Gib mir in 3 Stichpunkten Tipps zur Lasttest-Auswertung.",
    "Schreibe eine kurze Antwort auf Deutsch.",
    "Was misst die 95. Perzentil-Latenz?",
]


class ChatBotUser(HttpUser):
    wait_time = between(WAIT_MIN_SECONDS, WAIT_MAX_SECONDS)

    parsed = urlparse(TARGET_URL)
    host = f"{parsed.scheme}://{parsed.netloc}"
    page_path = parsed.path or "/"
    use_websocket = (
        CHAT_PATH.startswith("ws://")
        or CHAT_PATH.startswith("wss://")
        or "/ws" in CHAT_PATH
    )

    def on_start(self) -> None:
        # Falls lokal kein passender CA-Store vorhanden ist, kann VERIFY_TLS=false gesetzt werden.
        self.client.verify = VERIFY_TLS
        if not VERIFY_TLS:
            try:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

    @task
    def chat_interaction(self) -> None:
        # Optional: Landing-Page laden (fuer End-to-End, nicht fuer reinen Mistral-Bottleneck-Test)
        if INCLUDE_PAGE_GET:
            self.client.get(self.page_path, name="GET /test/")

        # Chat-Backend senden (WS oder HTTP)
        payload = {"messages": [{"role": "user", "content": random.choice(PROMPTS)}]}

        if self.use_websocket:
            self._send_ws_chat(payload)
        else:
            self._send_http_chat(payload)

    def _send_http_chat(self, payload: dict) -> None:
        http_path = CHAT_PATH
        if http_path.startswith("ws://") or http_path.startswith("wss://"):
            http_path = "/test/api/chat"
        if not http_path.startswith("/"):
            http_path = f"/{http_path}"

        with self.client.post(
            http_path,
            json=payload,
            name="POST chat",
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                if COUNT_429_AS_SUCCESS:
                    resp.success()
                else:
                    resp.failure("429 Too Many Requests")
                return

            if resp.status_code != 200:
                resp.failure(f"Unerwarteter Status: {resp.status_code}")
                return

            body = resp.text.strip()
            if not body:
                resp.failure("Leere Antwort vom Chat-Endpunkt")
            else:
                resp.success()

    def _send_ws_chat(self, payload: dict) -> None:
        start = time.perf_counter()
        response_length = 0
        exc = None
        metric_name = "WS chat"
        ws = None

        try:
            # Voraussetzung: pip install websocket-client
            import websocket  # type: ignore

            if CHAT_PATH.startswith("ws://") or CHAT_PATH.startswith("wss://"):
                ws_url = CHAT_PATH
            else:
                ws_scheme = "wss" if self.parsed.scheme == "https" else "ws"
                ws_path = CHAT_PATH if CHAT_PATH.startswith("/") else f"/{CHAT_PATH}"
                ws_url = f"{ws_scheme}://{self.parsed.netloc}{ws_path}"

            # Gleiche Query-Parameter wie im Frontend
            user_id = f"locust-{uuid.uuid4()}"
            session_id = f"locust-session-{uuid.uuid4()}"
            ws_url = f"{ws_url}?user_id={user_id}&session_id={session_id}"

            # Manche Proxies/CDNs erwarten Origin bei WS-Handshake.
            origin = f"{self.parsed.scheme}://{self.parsed.netloc}"

            ws_kwargs = {
                "timeout": 30,
                "origin": origin,
                "header": ["User-Agent: Locust-WebSocket-Client/1.0"],
            }
            if ws_url.startswith("wss://") and not VERIFY_TLS:
                ws_kwargs["sslopt"] = {"cert_reqs": ssl.CERT_NONE}

            ws = websocket.create_connection(ws_url, **ws_kwargs)
            ws.send(json.dumps(payload))

            while True:
                raw = ws.recv()
                if not isinstance(raw, str):
                    continue

                response_length += len(raw)
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "error":
                    error_text = str(msg.get("content", "WebSocket-Fehler"))
                    if "429" in error_text:
                        metric_name = "WS chat 429"
                        if COUNT_429_AS_SUCCESS:
                            exc = None
                            break
                    raise RuntimeError(error_text)
                if msg_type == "done":
                    break

        except Exception as err:
            exc = err
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

            events.request.fire(
                request_type="WS",
                name=metric_name,
                response_time=(time.perf_counter() - start) * 1000,
                response_length=response_length,
                response=None,
                context={},
                exception=exc,
            )


class RampToFiveUsers(LoadTestShape):
    """Ramptest: 0 -> 5 User in 2 Minuten, danach Stop."""

    def tick(self):
        run_time = self.get_run_time()

        if run_time > TEST_DURATION_SECONDS:
            return None

        target_users = int((run_time / TEST_DURATION_SECONDS) * PEAK_USERS)
        target_users = min(target_users, PEAK_USERS)

        # Am Ende sicherstellen, dass die Spitze wirklich erreicht wird
        if run_time >= TEST_DURATION_SECONDS - 1:
            target_users = PEAK_USERS

        return (target_users, SPAWN_RATE)
