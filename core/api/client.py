"""HTTP API clients built from the named configs in ``conf``, with auth + transport retries."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from core.config import get_api


def get_client(name: str, *, timeout: float = 30.0, retries: int = 3) -> httpx.Client:
    """Configured httpx client for the named API: base_url + bearer auth + connection retries."""
    api = get_api(name)
    headers = dict(api.headers)
    if api.token is not None:
        headers["Authorization"] = f"Bearer {api.token.get_secret_value()}"
    transport = httpx.HTTPTransport(retries=retries)
    return httpx.Client(
        base_url=api.base_url, headers=headers, timeout=timeout, transport=transport
    )


def paginate(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    page_param: str = "page",
    start: int = 1,
) -> Iterator[Any]:
    """Yield successive JSON pages, incrementing ``page_param`` until a page comes back empty.

    Pagination conventions vary by API — adjust ``page_param`` and the stopping rule per endpoint.
    """
    page = start
    while True:
        response = client.get(url, params={**(params or {}), page_param: page})
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return
        yield payload
        page += 1
