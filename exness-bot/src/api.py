"""Minimal ASGI health/metrics endpoint served by uvicorn (no web framework needed).

GET /health  -> {"status": "ok" | "tripped"}
GET /metrics -> full bot state (risk, signals, latency, positions)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Awaitable, Callable

from loguru import logger

StateProvider = Callable[[], dict[str, Any]]
ASGIApp = Callable[[dict[str, Any], Callable[[], Awaitable[dict[str, Any]]], Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[None]]


def create_app(state_provider: StateProvider) -> ASGIApp:
    async def _send_json(send: Callable[[dict[str, Any]], Awaitable[None]], status: int, payload: Any) -> None:
        body = json.dumps(payload, default=str).encode()
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})

    async def app(scope: dict[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]], send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        if scope.get("method") != "GET":
            await _send_json(send, 405, {"error": "method not allowed"})
            return
        try:
            path = scope.get("path", "/")
            if path == "/health":
                state = state_provider()
                tripped = state.get("risk", {}).get("state") == "tripped"
                await _send_json(send, 503 if tripped else 200, {"status": "tripped" if tripped else "ok"})
            elif path == "/metrics":
                await _send_json(send, 200, state_provider())
            else:
                await _send_json(send, 404, {"error": "not found"})
        except Exception as exc:
            logger.exception("API handler error")
            await _send_json(send, 500, {"error": str(exc)})

    return app


async def serve_api(app: ASGIApp, host: str, port: int, stop: asyncio.Event) -> None:
    """Run uvicorn inside the bot's event loop until ``stop`` is set."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False, lifespan="on")
    server = uvicorn.Server(config)
    # The bot owns process signals; stop uvicorn from installing its own handlers.
    if hasattr(server, "capture_signals"):
        server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]  # older uvicorn

    task = asyncio.create_task(server.serve())
    logger.info("Metrics API listening on http://{}:{}", host, port)
    stop_wait = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({task, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        server.should_exit = True
        stop_wait.cancel()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (asyncio.TimeoutError, SystemExit):
            logger.warning("API server did not shut down cleanly")
        except BaseException as exc:  # uvicorn calls sys.exit on bind failure
            logger.error("API server error: {}", exc)
