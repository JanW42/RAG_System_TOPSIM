import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("rag.observability")

try:
    from langfuse import get_client, observe as _langfuse_observe, propagate_attributes
except Exception as exc:  # pragma: no cover - fallback for optional runtime dependency
    _LANGFUSE_IMPORT_ERROR = exc

    def get_client() -> None:
        return None

    def _langfuse_observe(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    @contextmanager
    def propagate_attributes(**_kwargs):
        yield
else:
    _LANGFUSE_IMPORT_ERROR = None


_LANGFUSE_AUTH_CHECK_OK: bool | None = None

LANGFUSE_ENABLED = (
    _LANGFUSE_IMPORT_ERROR is None
    and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    and bool(os.getenv("LANGFUSE_SECRET_KEY"))
)

if _LANGFUSE_IMPORT_ERROR:
    logger.warning("Langfuse konnte nicht importiert werden: %s", _LANGFUSE_IMPORT_ERROR)
elif not LANGFUSE_ENABLED:
    logger.info(
        "Langfuse ist installiert, aber deaktiviert (LANGFUSE_PUBLIC_KEY/SECRET_KEY fehlen)."
    )


def observe(*args, **kwargs):
    if LANGFUSE_ENABLED:
        return _langfuse_observe(*args, **kwargs)

    def decorator(fn):
        return fn

    return decorator


def initialize_observability() -> bool:
    global _LANGFUSE_AUTH_CHECK_OK
    if not LANGFUSE_ENABLED:
        _LANGFUSE_AUTH_CHECK_OK = None
        return False

    client = get_client()
    if client is None:
        _LANGFUSE_AUTH_CHECK_OK = False
        logger.error("Langfuse client konnte nicht initialisiert werden.")
        return False

    try:
        _LANGFUSE_AUTH_CHECK_OK = bool(client.auth_check())
    except Exception as exc:
        _LANGFUSE_AUTH_CHECK_OK = False
        logger.exception("Langfuse auth_check fehlgeschlagen: %s", exc)
        return False

    if _LANGFUSE_AUTH_CHECK_OK:
        logger.info("Langfuse ist authentifiziert und bereit.")
    else:
        logger.error(
            "Langfuse auth_check ist fehlgeschlagen. Pruefe Keys/BASE_URL."
        )
    return _LANGFUSE_AUTH_CHECK_OK


def langfuse_auth_ok() -> bool | None:
    return _LANGFUSE_AUTH_CHECK_OK


def create_score(
    *,
    name: str,
    value: Any,
    trace_id: str | None,
    observation_id: str | None = None,
    data_type: str | None = None,
    comment: str | None = None,
) -> None:
    if not LANGFUSE_ENABLED or not trace_id:
        return
    client = get_client()
    if client is None:
        return
    try:
        payload = {
            "name": name,
            "value": value,
            "trace_id": trace_id,
        }
        if observation_id:
            payload["observation_id"] = observation_id
        if data_type:
            payload["data_type"] = data_type
        if comment:
            payload["comment"] = comment
        client.create_score(**payload)
    except Exception as exc:
        logger.exception("Langfuse create_score fehlgeschlagen: %s", exc)


def get_current_trace_id() -> str | None:
    if not LANGFUSE_ENABLED:
        return None
    client = get_client()
    if client is None:
        return None
    try:
        trace_id = client.get_current_trace_id()
        return str(trace_id) if trace_id else None
    except Exception:
        return None


@contextmanager
def trace_attributes(**attributes: Any) -> Iterator[None]:
    if not LANGFUSE_ENABLED:
        yield
        return

    filtered = {key: value for key, value in attributes.items() if value is not None}
    with propagate_attributes(**filtered):
        yield


def update_current_span(**kwargs: Any) -> None:
    if not LANGFUSE_ENABLED:
        return
    client = get_client()
    if client is None:
        return
    client.update_current_span(**kwargs)


def update_current_generation(**kwargs: Any) -> None:
    if not LANGFUSE_ENABLED:
        return
    client = get_client()
    if client is None:
        return
    client.update_current_generation(**kwargs)


def flush_observability() -> None:
    if not LANGFUSE_ENABLED:
        return
    client = get_client()
    if client is None:
        return
    client.flush()
