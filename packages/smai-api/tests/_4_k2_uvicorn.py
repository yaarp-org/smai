"""Test helper: run a FastAPI app under uvicorn on an ephemeral port.

Per Task 4.K2 the conformance suite's SSE tests require disconnect
propagation when the test exits the stream context. ``httpx``'s
``ASGITransport`` does not propagate disconnect to streaming-response
body generators (the body runs to completion before ``aclose()`` lets
the test exit), so an SSE stream that stays open indefinitely deadlocks
the conformance tests under ASGITransport.

The fix: run the app under a real ``uvicorn.Server`` on an ephemeral
loopback port and point the conformance suite's ``httpx.AsyncClient``
at that URL. Real HTTP cancels the body generator on TCP close, and
the stream tests pass without any test-mode hack in the SSE handler
itself.

The lifecycle is async-context-managed: enter the context to start the
server, exit to shut it down. The yielded base URL is what the
conformance suite's ``client`` fixture uses.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI


def _pick_free_port() -> int:
    """Bind a TCP socket to port 0 to learn an ephemeral free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def serve_app(app: FastAPI) -> AsyncIterator[str]:
    """Start ``app`` under uvicorn on an ephemeral port; yield base URL.

    The server runs in a background asyncio task; on context exit we
    set ``server.should_exit`` and await the task to drain cleanly.
    """
    port = _pick_free_port()
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        # Keep the loop responsive to shutdown signals on test teardown.
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    # Wait for the server to bind. ``server.started`` flips once the
    # transports are listening; race it against a small timeout so a
    # boot failure does not hang the test.
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.02)
    if not server.started:
        server.should_exit = True
        await serve_task
        raise RuntimeError("uvicorn server failed to start within 1s")

    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=5.0)
        except TimeoutError:
            server.force_exit = True
            try:
                await asyncio.wait_for(serve_task, timeout=5.0)
            except TimeoutError:
                serve_task.cancel()
                try:
                    await serve_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass


__all__ = ["serve_app"]
