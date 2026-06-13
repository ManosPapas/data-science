"""Capacity & queueing — Erlang C staffing, wait probabilities, utilization economics.

How many servers/agents/slots does a target service level need? The M/M/c queue answers it in
closed form — and exposes the core economics: waiting explodes *nonlinearly* as utilization
approaches 1, so "run everything at 95%" is a queueing disaster, not an efficiency win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QueueMetrics:
    """Steady-state M/M/c performance at a given staffing level."""

    servers: int
    utilization: float
    wait_probability: float
    average_wait: float
    average_queue: float

    # The arrival/service rates the SLA formula needs; required (no defaults) so a hand-built
    # QueueMetrics can't silently report decay=0, and repr=False keeps them out of the printout.
    arrival_rate: float = field(repr=False)
    service_rate: float = field(repr=False)

    def service_level(self, answer_within: float) -> float:
        """P(wait ≤ t): the '80% answered in 20s' style SLA number."""
        if answer_within < 0:
            raise ValueError("answer_within must be non-negative")
        decay = self.servers * self.service_rate - self.arrival_rate
        return 1.0 - self.wait_probability * float(np.exp(-decay * answer_within))


def erlang_c(*, arrival_rate: float, service_rate: float, servers: int) -> QueueMetrics:
    """M/M/c queue metrics: P(wait), average wait, queue length, utilization (Erlang C).

    ``arrival_rate`` = arrivals per unit time, ``service_rate`` = completions per server per unit
    time (1/avg handle time) — keep the units identical. Requires utilization < 1 or the queue
    diverges. The Erlang B recursion keeps it numerically stable at high loads.
    """
    if min(arrival_rate, service_rate) <= 0:
        raise ValueError("arrival_rate and service_rate must be positive")
    if servers < 1:
        raise ValueError("servers must be at least 1")
    offered_load = arrival_rate / service_rate
    utilization = offered_load / servers
    if utilization >= 1.0:
        raise ValueError(f"utilization {utilization:.2f} ≥ 1 — the queue is unstable; add servers")
    blocking = 1.0  # Erlang B via the stable recursion
    for k in range(1, servers + 1):
        blocking = offered_load * blocking / (k + offered_load * blocking)
    wait_probability = blocking / (1.0 - utilization * (1.0 - blocking))
    average_wait = wait_probability / (servers * service_rate - arrival_rate)
    average_queue = wait_probability * utilization / (1.0 - utilization)
    return QueueMetrics(
        servers=servers,
        utilization=float(utilization),
        wait_probability=float(wait_probability),
        average_wait=float(average_wait),
        average_queue=float(average_queue),
        arrival_rate=float(arrival_rate),
        service_rate=float(service_rate),
    )


def required_servers(
    *,
    arrival_rate: float,
    service_rate: float,
    target_wait_probability: float | None = None,
    target_service_level: float | None = None,
    answer_within: float | None = None,
    max_servers: int = 10_000,
) -> QueueMetrics:
    """Smallest staffing level meeting the SLA — by P(wait) cap or 'X% answered within t'.

    Pass either ``target_wait_probability`` (e.g. 0.2) or ``target_service_level`` with
    ``answer_within`` (e.g. 0.8 within 20s). Returns the full metrics at the chosen level so the
    cost-of-service conversation has numbers; capacity *utilization* optimization is reading the
    gap between this and what you run today.
    """
    if (target_wait_probability is None) == (target_service_level is None):
        raise ValueError("pass exactly one of target_wait_probability or target_service_level")
    if target_service_level is not None and answer_within is None:
        raise ValueError("target_service_level needs answer_within")
    minimum = int(np.ceil(arrival_rate / service_rate)) + 1
    for servers in range(max(minimum, 1), max_servers + 1):
        metrics = erlang_c(arrival_rate=arrival_rate, service_rate=service_rate, servers=servers)
        if target_wait_probability is not None:
            if metrics.wait_probability <= target_wait_probability:
                return metrics
        elif (
            target_service_level is not None
            and answer_within is not None
            and metrics.service_level(answer_within) >= target_service_level
        ):
            return metrics
    raise ValueError(f"no staffing level up to {max_servers} meets the target")
