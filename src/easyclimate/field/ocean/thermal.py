"""
The calculation of ocean thermocline variables.
"""

from __future__ import annotations
from typing import Literal

import numpy as np
import xarray as xr


def _temperature_in_depth_order(
    temperature: xr.DataArray, depth_dim: str
) -> xr.DataArray:
    depth = temperature[depth_dim]
    steps = np.diff(depth.values.astype(float))
    if depth.size == 0 or not np.isfinite(depth.values).all():
        raise ValueError("Depth must be nonempty and all its levels must be finite.")
    if np.all(steps > 0):
        return temperature
    if np.all(steps < 0):
        return temperature.isel({depth_dim: slice(None, None, -1)})
    raise ValueError("Depth must be strictly monotonic.")


def calc_seawater_thermocline_depth(
    seawater_temperature_data: xr.DataArray,
    depth_dim: str = "depth",
    *,
    method: Literal["layer", "centered"] = "layer",
) -> xr.DataArray:
    """
    Find the depth of the strongest vertical cooling in each temperature profile.

    Parameters
    ----------
    seawater_temperature_data : xarray.DataArray
        Seawater temperature with a one-dimensional, positive-down depth coordinate.
    depth_dim : str, default: "depth"
        Name of the depth dimension. Levels may be in either order.
    method : {"layer", "centered"}, default: "layer"
        "layer" computes the cooling rate between adjacent valid levels and returns
        their midpoint. "centered" differentiates on the actual depth coordinates
        and returns a sampled depth. The centered method uses first-order
        one-sided differences at the ends of the depth coordinate.

    Returns
    -------
    xarray.DataArray
        Depth in the input coordinate's units. Ties select the shallowest depth.
        Profiles without a positive cooling rate return NaN.
    """
    if method not in ("layer", "centered"):
        raise ValueError("method must be 'layer' or 'centered'.")

    temperature = _temperature_in_depth_order(seawater_temperature_data, depth_dim)
    temperature = temperature.where(np.isfinite(temperature))
    depth = temperature[depth_dim].astype(float)
    next_depth = depth.shift({depth_dim: -1})
    spacing = next_depth - depth
    next_temperature = temperature.shift({depth_dim: -1})
    layer_cooling = (temperature - next_temperature) / spacing
    if method == "layer":
        cooling = layer_cooling
        candidate_depth = (depth + next_depth) / 2
    else:
        previous_spacing = spacing.shift({depth_dim: 1})
        previous_cooling = layer_cooling.shift({depth_dim: 1})
        # Unequal spacing determines the weights of the two neighboring slopes.
        cooling = (previous_cooling * spacing + layer_cooling * previous_spacing) / (
            previous_spacing + spacing
        )
        cooling = xr.where(depth == depth[0], layer_cooling, cooling)
        cooling = xr.where(depth == depth[-1], previous_cooling, cooling)
        candidate_depth = depth

    cooling = cooling.where(np.isfinite(cooling) & (cooling > 0))
    strongest = cooling == cooling.max(dim=depth_dim)
    result = candidate_depth.where(strongest).min(dim=depth_dim)
    result.attrs = {"long_name": "Thermocline depth"}
    if "units" in depth.attrs:
        result.attrs["units"] = depth.attrs["units"]
    return result.rename(depth_dim)


def calc_Dx_depth(
    seawater_temperature_data: xr.DataArray,
    value: float,
    depth_dim: str = "depth",
    *,
    method: Literal["shallowest", "deepest"] = "shallowest",
) -> xr.DataArray:
    """
    Find the depth of a target temperature in each seawater profile.

    Parameters
    ----------
    seawater_temperature_data : xarray.DataArray
        Seawater temperature in degrees Celsius with a one-dimensional,
        positive-down depth coordinate.
    value : float
        Target temperature in degrees Celsius.
    depth_dim : str, default: "depth"
        Name of the depth dimension. Levels may be in either order.
    method : {"shallowest", "deepest"}, default: "shallowest"
        Select the shallowest or deepest occurrence of the target temperature.
        Exact hits use the sampled depth. Crossings in either direction use
        linear interpolation between adjacent valid levels.

    Returns
    -------
    xarray.DataArray
        Depth in the input coordinate's units. Profiles without a crossing or
        exact hit return NaN.
    """
    if method not in ("shallowest", "deepest"):
        raise ValueError("method must be 'shallowest' or 'deepest'.")

    temperature = _temperature_in_depth_order(seawater_temperature_data, depth_dim)
    temperature = temperature.where(np.isfinite(temperature))
    depth = temperature[depth_dim].astype(float)
    next_depth = depth.shift({depth_dim: -1})
    next_temperature = temperature.shift({depth_dim: -1})

    crosses = ((temperature > value) & (next_temperature < value)) | (
        (temperature < value) & (next_temperature > value)
    )
    fraction = (value - temperature) / (next_temperature - temperature).where(crosses)
    crossing_depth = depth + fraction * (next_depth - depth)
    candidates = xr.where(temperature == value, depth, crossing_depth)
    if method == "shallowest":
        result = candidates.min(dim=depth_dim)
    else:
        result = candidates.max(dim=depth_dim)
    result.attrs = {"long_name": f"Depth of the {value:g} °C isotherm"}
    if "units" in depth.attrs:
        result.attrs["units"] = depth.attrs["units"]
    return result.rename(None)


def calc_D14_depth(
    seawater_temperature_data: xr.DataArray,
    value: float = 14,
    depth_dim: str = "depth",
    *,
    method: Literal["shallowest", "deepest"] = "shallowest",
) -> xr.DataArray:
    """
    Find the depth of the 14 °C isotherm.

    See :func:`calc_Dx_depth` for the temperature, depth coordinate and crossing options.
    """
    return calc_Dx_depth(
        seawater_temperature_data=seawater_temperature_data,
        value=value,
        depth_dim=depth_dim,
        method=method,
    )


def calc_D17_depth(
    seawater_temperature_data: xr.DataArray,
    value: float = 17,
    depth_dim: str = "depth",
    *,
    method: Literal["shallowest", "deepest"] = "shallowest",
) -> xr.DataArray:
    """
    Find the depth of the 17 °C isotherm.

    See :func:`calc_Dx_depth` for the temperature, depth coordinate and crossing options.
    """
    return calc_Dx_depth(
        seawater_temperature_data=seawater_temperature_data,
        value=value,
        depth_dim=depth_dim,
        method=method,
    )


def calc_D20_depth(
    seawater_temperature_data: xr.DataArray,
    value: float = 20,
    depth_dim: str = "depth",
    *,
    method: Literal["shallowest", "deepest"] = "shallowest",
) -> xr.DataArray:
    """
    Find the depth of the 20 °C isotherm.

    See :func:`calc_Dx_depth` for the temperature, depth coordinate and crossing options.
    """
    return calc_Dx_depth(
        seawater_temperature_data=seawater_temperature_data,
        value=value,
        depth_dim=depth_dim,
        method=method,
    )


def calc_D26_depth(
    seawater_temperature_data: xr.DataArray,
    value: float = 26,
    depth_dim: str = "depth",
    *,
    method: Literal["shallowest", "deepest"] = "shallowest",
) -> xr.DataArray:
    """
    Find the depth of the 26 °C isotherm.

    See :func:`calc_Dx_depth` for the temperature, depth coordinate and crossing options.
    """
    return calc_Dx_depth(
        seawater_temperature_data=seawater_temperature_data,
        value=value,
        depth_dim=depth_dim,
        method=method,
    )


def calc_D28_depth(
    seawater_temperature_data: xr.DataArray,
    value: float = 28,
    depth_dim: str = "depth",
    *,
    method: Literal["shallowest", "deepest"] = "shallowest",
) -> xr.DataArray:
    """
    Find the depth of the 28 °C isotherm.

    See :func:`calc_Dx_depth` for the temperature, depth coordinate and crossing options.
    """
    return calc_Dx_depth(
        seawater_temperature_data=seawater_temperature_data,
        value=value,
        depth_dim=depth_dim,
        method=method,
    )
