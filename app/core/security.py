"""Shared authentication primitives.

There's no full auth system (OAuth/JWT) here yet — see TODO.md. What exists
is opt-in, disabled-by-default shared-secret protection for a couple of
sensitive spots (the admin API, optionally /predict), built on a single
timing-safe comparison so call sites don't each reimplement it.
"""

import secrets


def secrets_match(provided: str | None, expected: str) -> bool:
    """Timing-safe comparison of a header value against a configured secret.

    Parameters
    ----------
    provided : str, optional
        Value from the request (e.g. a header). Missing/empty is treated as
        a non-match rather than raising.
    expected : str
        The configured secret to compare against.

    Returns
    -------
    bool
    """
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)
