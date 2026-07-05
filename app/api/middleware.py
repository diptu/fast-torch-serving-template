import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Tags every request with an ID (reusing an incoming X-Request-ID if the
    caller already has one, e.g. from an upstream gateway), makes it
    available to log statements via request_id_var, and echoes it back on
    the response so a client can correlate their request with server logs."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Tag the request with an ID, then echo it back on the response.

        Parameters
        ----------
        request : Request
        call_next : Callable
            Passed through unchanged to the next middleware/handler.

        Returns
        -------
        Response
            With an ``X-Request-ID`` header set.
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        # Deliberately not reset in a finally: if call_next() raises, the
        # exception unwinds through here on its way to the global exception
        # handler further out, and that handler needs to still see this
        # value to log/report it. Each request runs in its own asyncio Task
        # with its own copy of the context, so there's nothing to leak into
        # a later, unrelated request.
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
