"""Solar position computation via pvlib and EWMA envelope extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Site coordinates for the paper's six PV plants.
# DKASC sites (desert): Alice Springs, Australia.
SITE_COORDS = {
    "dkasc_site7": {"name": "DKASC Site 7 (CdTe)", "lat": -23.7624, "lon": 133.8748, "tz": "Australia/Darwin"},
    "dkasc_site10": {"name": "DKASC Site 10 (mono-Si)", "lat": -23.7624, "lon": 133.8748, "tz": "Australia/Darwin"},
    "dkasc_site2": {"name": "DKASC Site 2 (tracker)", "lat": -23.7624, "lon": 133.8748, "tz": "Australia/Darwin"},
    "pvgis_singapore": {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "tz": "Asia/Singapore"},
    "pvgis_oslo": {"name": "Oslo", "lat": 59.9139, "lon": 10.7522, "tz": "Europe/Oslo"},
    "pvgis_madrid": {"name": "Madrid", "lat": 40.4168, "lon": -3.7038, "tz": "Europe/Madrid"},
}


def compute_solar_position(
    timestamps: pd.DatetimeIndex,
    site: str = "dkasc_site7",
    pressure: float = 101325.0,
    temperature: float = 12.0,
) -> pd.DataFrame:
    """Compute solar elevation (and azimuth) for given timestamps and site.

    Wraps pvlib.location.Location.get_solarposition().

    Args:
        timestamps: pandas DatetimeIndex (timezone-aware recommended)
        site: key in SITE_COORDS
        pressure: atmospheric pressure (Pa)
        temperature: ambient temperature (C)

    Returns:
        DataFrame with columns ['elevation', 'azimuth', 'apparent_elevation'].
    """
    try:
        import pvlib
    except ImportError as e:
        raise ImportError(
            "pvlib is required for solar position computation. "
            "Install with: pip install pvlib"
        ) from e

    coords = SITE_COORDS[site]
    location = pvlib.location.Location(
        latitude=coords["lat"], longitude=coords["lon"], tz=coords["tz"],
    )
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC").tz_convert(coords["tz"])
    else:
        timestamps = timestamps.tz_convert(coords["tz"])
    sp = location.get_solarposition(times=timestamps, pressure=pressure, temperature=temperature)
    return sp[["elevation", "azimuth", "apparent_elevation"]]


def compute_ewma_envelope(
    x: np.ndarray,
    alpha: float = 0.9,
    warm_start: bool = True,
) -> np.ndarray:
    """EWMA envelope extractor (no learnable params, used by SGBA-EWMA).

    Args:
        x: shape (..., L) input series
        alpha: decay rate; half-life h = log(0.5)/log(alpha).
            With alpha=0.9, half-life ~ 6.6 daytime hours (~0.55 solar days).
        warm_start: if True, initialize envelope with x[0]; else 0.

    Returns:
        Envelope of same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False

    out = np.empty_like(x)
    if warm_start:
        out[:, 0] = x[:, 0]
    else:
        out[:, 0] = 0.0
    for t in range(1, x.shape[-1]):
        out[:, t] = alpha * out[:, t - 1] + (1 - alpha) * x[:, t]

    if squeeze:
        out = out[0]
    return out


def clear_sky_index(
    ghi: np.ndarray,
    clear_sky_ghi: np.ndarray,
    eps: float = 1.0,
) -> np.ndarray:
    """Clear-sky index CSI = GHI / clear-sky GHI.

    Args:
        ghi: observed GHI (W/m^2)
        clear_sky_ghi: clear-sky GHI from pvlib's Ineichen model
        eps: small denominator offset to avoid division by zero

    Returns:
        CSI of same shape; values typically in [0, 1.2], where
        <0.2 = Overcast, 0.2-0.5 = Cloudy, 0.5-0.8 = Partly Clear, >0.8 = Clear.
    """
    return ghi / (clear_sky_ghi + eps)
