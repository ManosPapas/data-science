"""Lightweight geo features — great-circle distance and bounding box (no geopandas needed).

For spatial joins / choropleths, install the ``geo`` extra (geopandas) and work with GeoDataFrames.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_KM = 6371.0


def haversine(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> NDArray[np.float64]:
    """Great-circle distance (km) between two (lat, lon) points in degrees; vectorized."""
    phi1, lam1, phi2, lam2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)), dtype=float)


def bounding_box(lats: ArrayLike, lons: ArrayLike) -> tuple[float, float, float, float]:
    """(min_lat, min_lon, max_lat, max_lon) covering the points."""
    la = np.asarray(lats, dtype=float)
    lo = np.asarray(lons, dtype=float)
    return float(la.min()), float(lo.min()), float(la.max()), float(lo.max())
