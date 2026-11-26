"""Statistics over observations within each calendar day."""

from __future__ import annotations

import xarray as xr

__all__ = [
    "calc_daily_mean",
    "calc_daily_sum",
    "calc_daily_std",
    "calc_daily_var",
    "calc_daily_max",
    "calc_daily_min",
]


def calc_daily_mean(
    data_input: xr.DataArray, dim: str = "time", **kwargs
) -> xr.DataArray:
    """
    Calculate the mean of the observations in each calendar day.

    Parameters
    ----------
    data_input : :py:class:`xarray.DataArray`
        Input with a datetime-like coordinate along ``dim``. Hourly data is typical.
    dim : str, default "time"
        Time dimension to resample.
    ``**kwargs``
        Keyword arguments passed to the mean reduction, such as ``skipna``.

    Returns
    -------
    :py:class:`xarray.DataArray`
        Daily values labeled at the start of each day.
        Days without observations have missing values.

    Examples
    --------
    >>> import easyclimate as ecl
    >>> import pandas as pd
    >>> import xarray as xr
    >>> time = pd.date_range("2020-01-01 22:00", periods=4, freq="h")
    >>> hourly = xr.DataArray([2, 4, 6, 8], dims="time", coords={"time": time})
    >>> ecl.calc_daily_mean(hourly).values.tolist()
    [3.0, 7.0]
    """
    return data_input.resample({dim: "1D"}).mean(**kwargs)


def calc_daily_sum(
    data_input: xr.DataArray, dim: str = "time", **kwargs
) -> xr.DataArray:
    """
    Calculate the sum of the observations in each calendar day.

    Parameters
    ----------
    data_input : :py:class:`xarray.DataArray`
        Input with a datetime-like coordinate along ``dim``. Hourly data is typical.
    dim : str, default "time"
        Time dimension to resample.
    ``**kwargs``
        Keyword arguments passed to the sum reduction, such as ``skipna`` and ``min_count``.

    Returns
    -------
    :py:class:`xarray.DataArray`
        Daily values labeled at the start of each day.
        Days without observations have missing values.
    """
    return data_input.resample({dim: "1D"}).sum(**kwargs)


def calc_daily_std(
    data_input: xr.DataArray, dim: str = "time", **kwargs
) -> xr.DataArray:
    """
    Calculate the standard deviation of the observations in each calendar day.

    Parameters
    ----------
    data_input : :py:class:`xarray.DataArray`
        Input with a datetime-like coordinate along ``dim``. Hourly data is typical.
    dim : str, default "time"
        Time dimension to resample.
    ``**kwargs``
        Keyword arguments passed to the reduction, such as ``skipna`` and ``ddof``.

    Returns
    -------
    :py:class:`xarray.DataArray`
        Daily values labeled at the start of each day.
        Days without observations have missing values.
    """
    return data_input.resample({dim: "1D"}).std(**kwargs)


def calc_daily_var(
    data_input: xr.DataArray, dim: str = "time", **kwargs
) -> xr.DataArray:
    """
    Calculate the variance of the observations in each calendar day.

    Parameters
    ----------
    data_input : :py:class:`xarray.DataArray`
        Input with a datetime-like coordinate along ``dim``. Hourly data is typical.
    dim : str, default "time"
        Time dimension to resample.
    ``**kwargs``
        Keyword arguments passed to the variance reduction, such as ``skipna`` and ``ddof``.

    Returns
    -------
    :py:class:`xarray.DataArray`
        Daily values labeled at the start of each day.
        Days without observations have missing values.
    """
    return data_input.resample({dim: "1D"}).var(**kwargs)


def calc_daily_max(
    data_input: xr.DataArray, dim: str = "time", **kwargs
) -> xr.DataArray:
    """
    Calculate the maximum of the observations in each calendar day.

    Parameters
    ----------
    data_input : :py:class:`xarray.DataArray`
        Input with a datetime-like coordinate along ``dim``. Hourly data is typical.
    dim : str, default "time"
        Time dimension to resample.
    ``**kwargs``
        Keyword arguments passed to the maximum reduction, such as ``skipna``.

    Returns
    -------
    :py:class:`xarray.DataArray`
        Daily values labeled at the start of each day.
        Days without observations have missing values.
    """
    return data_input.resample({dim: "1D"}).max(**kwargs)


def calc_daily_min(
    data_input: xr.DataArray, dim: str = "time", **kwargs
) -> xr.DataArray:
    """
    Calculate the minimum of the observations in each calendar day.

    Parameters
    ----------
    data_input : :py:class:`xarray.DataArray`
        Input with a datetime-like coordinate along ``dim``. Hourly data is typical.
    dim : str, default "time"
        Time dimension to resample.
    ``**kwargs``
        Keyword arguments passed to the minimum reduction, such as ``skipna``.

    Returns
    -------
    :py:class:`xarray.DataArray`
        Daily values labeled at the start of each day.
        Days without observations have missing values.
    """
    return data_input.resample({dim: "1D"}).min(**kwargs)
