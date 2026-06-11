"""GraphQL over the configured httpx clients — one POST, fetch exactly the fields you need.

Reuses a named API's config (base_url + auth) from ``conf/apis.yaml`` — point ``base_url`` at the
GraphQL endpoint. No extra dependency: a GraphQL call is a JSON POST of ``{query, variables}``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx

from core.api.client import get_client


class GraphQLError(RuntimeError):
    """Raised when a GraphQL response carries an ``errors`` array."""


def _execute(
    client: httpx.Client, document: str, variables: Mapping[str, Any] | None, endpoint: str
) -> Any:
    response = client.post(endpoint, json={"query": document, "variables": dict(variables or {})})
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise GraphQLError(str(payload["errors"]))
    return payload.get("data")


def graphql(
    name: str, document: str, *, variables: Mapping[str, Any] | None = None, endpoint: str = ""
) -> Any:
    """Run a GraphQL query or mutation on the named API; return ``data`` (raises on ``errors``)."""
    with get_client(name) as client:
        return _execute(client, document, variables, endpoint)


def paginate_graphql(
    name: str,
    document: str,
    connection_path: Sequence[str],
    *,
    variables: Mapping[str, Any] | None = None,
    endpoint: str = "",
    page_size: int = 100,
) -> Iterator[Any]:
    """Yield successive pages of a Relay-style connection, following ``pageInfo.endCursor``.

    The query must accept ``$first`` / ``$after``; the connection must expose
    ``pageInfo { hasNextPage endCursor }``. ``connection_path`` keys into the connection within
    ``data`` (e.g. ``["orders"]``). One client is reused across pages.
    """
    cursor: str | None = None
    base_vars = dict(variables or {})
    with get_client(name) as client:
        while True:
            data = _execute(
                client, document, {**base_vars, "first": page_size, "after": cursor}, endpoint
            )
            connection: Any = data
            for key in connection_path:
                connection = connection[key]
            yield connection
            page_info = connection.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                return
            cursor = page_info.get("endCursor")
