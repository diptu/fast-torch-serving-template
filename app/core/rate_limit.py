"""Shared slowapi rate limiter.

A single ``Limiter`` instance is required across the app: ``app/main.py``
registers it on ``app.state`` and wires up the ``RateLimitExceeded`` handler;
``app/api/routes/predict.py`` applies ``@limiter.limit(...)`` to the predict
routes. Two separate ``Limiter()`` instances would each keep their own
independent counters, silently doubling the effective limit.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _rate_limit_key(request: Request) -> str:
    """Key by ``X-API-Key`` when present, else by remote address.

    Parameters
    ----------
    request : Request

    Returns
    -------
    str

    Notes
    -----
    Keying by API key rather than always by IP means clients sharing a
    NAT/gateway each get their own budget once they authenticate, instead of
    fighting over one shared IP-based limit.
    """
    api_key = request.headers.get("x-api-key")
    return api_key if api_key else get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
