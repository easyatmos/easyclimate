"""TN01 tests independent of downloaded tutorial data.

The NumPy oracle follows Nishii's monthly NCL equations and opt=0 secants:
https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/TakayaNakamura2001-ncep-monthly.ncl
It is a numerical translation, not an execution of NCL itself.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import easyclimate as ecl


def fields():
    coords = {
        "level": [1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0],
        "lat": np.arange(20.0, 81.0, 10.0),
        "lon": np.arange(0.0, 360.0, 15.0),
    }
    z = -8000 * np.log(np.array(coords["level"]) / 1000)
    phase = (
        2 * np.deg2rad(coords["lon"])[None, None, :]
        + 1.5 * np.deg2rad(coords["lat"])[None, :, None]
        + z[:, None, None] / 6000
    )
    psi = xr.DataArray(2e6 * np.cos(phase), dims=tuple(coords), coords=coords)
    height = psi * (2 * 7.292e-5 * np.sin(np.deg2rad(psi.lat))) / 9.8
    u = xr.full_like(psi, 20.0)
    v = xr.full_like(psi, 7.0)
    temperature = xr.full_like(psi, 250.0)
    return height, u, v, temperature, psi


def reference_numpy(height, u, v, temperature):
    """Independent ndarray implementation, with the NCL constants."""

    def diff(a, coordinate, axis, cyclic=False):
        forward = np.roll(a, -1, axis=axis)
        backward = np.roll(a, 1, axis=axis)
        if cyclic:
            return (forward - backward) / (2 * (coordinate[1] - coordinate[0]))
        distance = np.roll(coordinate, -1) - np.roll(coordinate, 1)
        shape = [1] * a.ndim
        shape[axis] = len(coordinate)
        answer = (forward - backward) / distance.reshape(shape)
        first = [slice(None)] * a.ndim
        second = [slice(None)] * a.ndim
        last = [slice(None)] * a.ndim
        previous = [slice(None)] * a.ndim
        first[axis], second[axis] = 0, 1
        last[axis], previous[axis] = -1, -2
        answer[tuple(first)] = (a[tuple(second)] - a[tuple(first)]) / (
            coordinate[1] - coordinate[0]
        )
        answer[tuple(last)] = (a[tuple(last)] - a[tuple(previous)]) / (
            coordinate[-1] - coordinate[-2]
        )
        return answer

    p = height.level.values[:, None, None] / 1000
    lat = np.deg2rad(height.lat.values)
    lon = np.deg2rad(height.lon.values)
    f = 4 * np.pi / 86400 * np.sin(lat)[None, :, None]
    c = np.cos(lat)[None, :, None]
    psi = height.values * 9.80665 / f
    log_height = -8000 * np.log(p[:, 0, 0])
    dl = diff(psi, lon, 2, True)
    dp = diff(psi, lat, 1)
    dz = diff(psi, log_height, 0)
    n2 = 290 / 8000 * p**0.286 * diff(temperature.values / p**0.286, log_height, 0)
    speed = np.hypot(u.values, v.values)
    a = dl**2 - psi * diff(dl, lon, 2, True)
    b = dl * dp - psi * diff(dl, lat, 1)
    d = dp**2 - psi * diff(dp, lat, 1)
    az = dl * dz - psi * diff(dl, log_height, 0)
    bz = dp * dz - psi * diff(dp, log_height, 0)
    fx = p / (2 * speed * 6378388**2) * (u.values / c * a + v.values * b)
    fy = p / (2 * speed * 6378388**2) * (u.values * b + c * v.values * d)
    fz = p * f**2 / (2 * speed * n2 * 6378388) * (u.values * az + c * v.values * bz)
    return {"psi_p": psi, "fx": fx, "fy": fy, "fz": fz, "N2": n2}


def calculate(height, u, v, temperature, **kwargs):
    return ecl.calc_TN_wave_activity_3D_flux(
        height,
        u,
        v,
        temperature,
        temper_climatology_data_units="degK",
        vertical_dim="level",
        vertical_dim_units="hPa",
        cyclic_lon=True,
        **kwargs,
    )


def test_matches_ncl_numpy_oracle():
    h, u, v, t, _ = fields()
    actual = calculate(h, u, v, t, g=9.80665, R=6378388, omega=2 * np.pi / 86400)
    expected = reference_numpy(h, u, v, t)
    for name, array in expected.items():
        np.testing.assert_allclose(
            actual[name], array, rtol=2e-6, atol=1e-8, equal_nan=True
        )
    assert actual.fz.units == "m^2/s^2"
    assert actual.fz.positive == "up"


def test_vertical_and_horizontal_interfaces_agree():
    h, u, v, t, _ = fields()
    three = calculate(h, u, v, t)
    vertical = ecl.calc_TN_wave_activity_vertical_flux(
        h, u, v, t, "degK", "level", "hPa", cyclic_lon=True
    )
    xr.testing.assert_identical(three.fz, vertical.fz)
    horizontal = ecl.calc_TN_wave_activity_horizontal_flux(h, u, v, "level", "hPa")
    # Legacy boundaries differ; compare the common interior stencil.
    for name in ("fx", "fy"):
        np.testing.assert_allclose(
            three[name].isel(lat=slice(2, -2), lon=slice(2, -2)),
            horizontal[name].isel(lat=slice(2, -2), lon=slice(2, -2)),
            rtol=3e-6,
            atol=1e-7,
        )


def test_3D_keeps_streamfunction_precision_until_derivatives():
    h, u, v, t, psi = fields()
    psi = 1e8 + psi / 1000
    h = psi * (2 * 7.292e-5 * np.sin(np.deg2rad(psi.lat))) / 9.8
    actual = calculate(h, u, v, t)
    expected = ecl.calc_TN_wave_activity_horizontal_flux(
        h, u, v, "level", "hPa", cyclic_lon=True, finite_difference="ncl"
    )
    for name in ("fx", "fy"):
        xr.testing.assert_identical(actual[name], expected[name])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_output_precision_and_roundoff(dtype):
    h, u, v, t, _ = fields()
    result = calculate(h, u, v, t, output_dtype=dtype)
    assert all(var.dtype == np.dtype(dtype) for var in result.data_vars.values())
    double = calculate(h, u, v, t, output_dtype="float64")
    for name in result.data_vars:
        xr.testing.assert_identical(result[name], double[name].astype(dtype))
    vertical = ecl.calc_TN_wave_activity_vertical_flux(
        h, u, v, t, "degK", "level", "hPa", cyclic_lon=True, output_dtype=dtype
    )
    xr.testing.assert_identical(result.fz, vertical.fz)


def test_analytic_linear_wave_vertical_sign_and_magnitude():
    """Exact secants of linear psi and theta check both U/V terms and units."""
    h, u, v, t, _ = fields()
    p = h.level / 1000
    z = -8000 * np.log(p)
    amplitude = 1e6
    psi = amplitude * (np.deg2rad(h.lon) + np.deg2rad(h.lat) + z / 6000)
    psi = psi.transpose(*h.dims)
    t = (300 + 0.003 * z) * p**0.286
    actual = ecl.calc_TN_wave_activity_vertical_flux(
        None,
        u,
        v,
        t,
        "degK",
        "level",
        "hPa",
        psi_prime_data=psi,
    )
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(h.lat))
    n2 = 290 / 8000 * p**0.286 * 0.003
    expected = (
        p
        * f**2
        / (2 * np.hypot(u, v) * n2 * 6371200)
        * (u + v * np.cos(np.deg2rad(h.lat)))
        * amplitude**2
        / 6000
    )
    interior = {"level": slice(1, -1), "lat": slice(1, -1), "lon": slice(1, -1)}
    np.testing.assert_allclose(
        actual.fz.isel(interior), expected.transpose(*h.dims).isel(interior), rtol=1e-6
    )
    assert (actual.fz.isel(interior) > 0).all()
    reversed_wave = ecl.calc_TN_wave_activity_vertical_flux(
        None,
        u,
        v,
        t,
        "degK",
        "level",
        "hPa",
        psi_prime_data=(psi - 2 * amplitude * z / 6000),
    )
    np.testing.assert_allclose(
        reversed_wave.fz.isel(interior), -actual.fz.isel(interior), rtol=1e-6
    )


@pytest.mark.parametrize("units,factor", [("Pa", 100.0), ("mbar", 1.0)])
def test_pressure_units(units, factor):
    h, u, v, t, _ = fields()
    expected = calculate(h, u, v, t)
    arrays = [a.assign_coords(level=a.level * factor) for a in (h, u, v, t)]
    actual = ecl.calc_TN_wave_activity_3D_flux(
        *arrays, "degK", "level", units, cyclic_lon=True
    )
    actual = actual.assign_coords(level=h.level)
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("component", ["vertical", "3D"])
@pytest.mark.parametrize(
    "units,scale,offset",
    [("degK", 1.0, 0.0), ("degC", 1.0, -273.15), ("degF", 1.8, -459.67)],
)
def test_temperature_units(component, units, scale, offset):
    h, u, v, t, _ = fields()
    function = getattr(ecl, f"calc_TN_wave_activity_{component}_flux")
    expected = function(h, u, v, t, "degK", "level", "hPa", cyclic_lon=True)
    actual = function(
        h,
        u,
        v,
        t * scale + offset,
        temper_climatology_data_units=units,
        vertical_dim="level",
        vertical_dim_units="hPa",
        cyclic_lon=True,
    )
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dimension", ["level", "lat", "lon"])
def test_coordinate_reversal(dimension):
    h, u, v, t, _ = fields()
    expected = calculate(h, u, v, t)
    arrays = [a.isel({dimension: slice(None, None, -1)}) for a in (h, u, v, t)]
    actual = calculate(*arrays).isel({dimension: slice(None, None, -1)})
    xr.testing.assert_allclose(actual, expected)


def test_streamfunction_and_custom_dimensions():
    h, u, v, t, psi = fields()
    expected = calculate(h, u, v, t)
    names = {"level": "pressure", "lat": "latitude", "lon": "longitude"}
    arrays = [
        a.rename(names).transpose("longitude", "pressure", "latitude")
        for a in (u, v, t, psi)
    ]
    actual = ecl.calc_TN_wave_activity_3D_flux(
        None,
        *arrays[:3],
        "degK",
        "pressure",
        "hPa",
        psi_prime_data=arrays[3],
        lat_dim="latitude",
        lon_dim="longitude",
        cyclic_lon=True,
    )
    assert actual.fz.dims == arrays[3].dims
    xr.testing.assert_allclose(
        actual.rename({v: k for k, v in names.items()}).transpose(*h.dims), expected
    )


@pytest.mark.parametrize("count", [1, 3])
def test_monthly_climatology_matches_target_time(count):
    h, u, v, t, _ = fields()
    dates = pd.to_datetime(["2020-01-15", "2021-02-15", "2022-01-15"][:count])
    h = h.expand_dims(time=dates)
    background_dates = pd.to_datetime(["1990-02-01", "1990-01-01"])
    backgrounds = [
        xr.concat([a * 1.1, a], dim=xr.IndexVariable("time", background_dates))
        for a in (u, v, t)
    ]
    actual = calculate(h, *backgrounds)
    expected = xr.concat(
        [
            calculate(
                h.isel(time=i, drop=True),
                *[a * (1.1 if date.month == 2 else 1) for a in (u, v, t)],
            )
            for i, date in enumerate(dates)
        ],
        dim=xr.IndexVariable("time", dates),
    )
    xr.testing.assert_allclose(actual, expected)
    month_backgrounds = [
        a.assign_coords(month=("time", [2, 1]))
        .swap_dims({"time": "month"})
        .drop_vars("time")
        for a in backgrounds
    ]
    xr.testing.assert_allclose(calculate(h, *month_backgrounds), expected)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("level_selection", ["singleton", "multiple"])
def test_horizontal_scalar_pressure_matches_monthly_background(count, level_selection):
    h, u, v, _, _ = fields()
    # Vary wind direction with pressure so selecting an arbitrary background
    # level instead of the anomaly's scalar level gives a different flux.
    u = u * u.level / 500
    dates = pd.to_datetime(["2014-11-01", "2014-11-02"][:count])
    anomaly = h.sel(level=500).expand_dims(time=dates)
    basic = [
        a.sel(level=[500]) if level_selection == "singleton" else a for a in (u, v)
    ]
    basic = [a.expand_dims(time=pd.to_datetime(["2000-11-01"])) for a in basic]
    actual = ecl.calc_TN_wave_activity_horizontal_flux(anomaly, *basic, "level", "hPa")
    expected = ecl.calc_TN_wave_activity_horizontal_flux(
        anomaly,
        u.sel(level=500, drop=True),
        v.sel(level=500, drop=True),
        "level",
        "hPa",
    )
    xr.testing.assert_identical(actual, expected)
    assert actual.fx.dims == anomaly.dims
    assert float(actual.level) == 500


@pytest.mark.parametrize("component", ["vertical", "3D"])
@pytest.mark.parametrize("wind", [0.5, 1.0, 3.0, 5.0])
def test_fortran_wind_cutoff_default(component, wind):
    h, u, v, t, _ = fields()
    function = getattr(ecl, f"calc_TN_wave_activity_{component}_flux")
    result = function(h, u * 0 + wind, v, t, "degK", "level", "hPa", cyclic_lon=True)
    for name in ["fz"] if component == "vertical" else ["fx", "fy", "fz"]:
        assert bool(result[name].isel(level=3, lat=3).notnull().all()) == (wind >= 1)
    ncl = function(
        h, u * 0 + wind, v, t, "degK", "level", "hPa", cyclic_lon=True, min_westerly=5
    )
    for name in ["fz"] if component == "vertical" else ["fx", "fy", "fz"]:
        assert bool(ncl[name].isel(level=3, lat=3).notnull().all()) == (wind >= 5)


def test_masks_and_one_sided_vertical_boundaries():
    h, u, v, t, _ = fields()
    actual = calculate(h, u, v, t)
    assert actual.fz.isel(level=[0, -1], lat=3).notnull().all()
    assert np.isfinite(actual.fz.isel(level=3, lat=3)).all()
    assert calculate(h, u * 0, v * 0, t).fz.isnull().all()
    assert calculate(h, u * 0 - 10, v, t).fz.isnull().all()
    assert (
        calculate(h, u * 0 - 10, v, t, min_westerly=None)
        .fz.isel(level=3, lat=3)
        .notnull()
        .all()
    )
    unstable = t * (t.level / 1000) ** 0.5
    bad = calculate(h, u, v, unstable)
    assert bad.fz.isnull().all()
    assert bad.fx.isel(lat=3).notnull().all()


def test_equator_and_poles_masked():
    h, u, v, t, _ = fields()
    arrays = [
        a.assign_coords(lat=[-90.0, -45.0, -5.0, 0.0, 5.0, 45.0, 90.0])
        for a in (h, u, v, t)
    ]
    actual = calculate(*arrays)
    for name in ("fx", "fy", "fz", "psi_p"):
        assert actual[name].sel(lat=[-90.0, -5.0, 0.0, 5.0, 90.0]).isnull().all()
        assert not np.isinf(actual[name]).any()


def test_dask_laziness():
    h, u, v, t, _ = fields()
    expected = calculate(h, u, v, t)
    actual = calculate(
        *[a.chunk({"level": 2, "lat": 3, "lon": 6}) for a in (h, u, v, t)]
    )
    assert hasattr(actual.fz.data, "dask")
    xr.testing.assert_allclose(actual.compute(), expected)


@pytest.mark.parametrize(
    "kind", ["zero_pressure", "duplicate_pressure", "wrong_grid", "regional_cyclic"]
)
def test_invalid_grids_rejected(kind):
    h, u, v, t, _ = fields()
    if kind == "zero_pressure":
        h = h.assign_coords(level=[1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 0.0])
    elif kind == "duplicate_pressure":
        h = h.assign_coords(level=[1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 200.0])
        u, v, t = [a.assign_coords(level=h.level) for a in (u, v, t)]
    elif kind == "wrong_grid":
        t = t.assign_coords(level=t.level + 1)
    else:
        h, u, v, t = [a.isel(lon=slice(0, 10)) for a in (h, u, v, t)]
    with pytest.raises(ValueError):
        calculate(h, u, v, t)
