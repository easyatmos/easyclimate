"""NCL finite differences on irregular and periodic coordinates."""

import numpy as np
import pytest
import xarray as xr

from easyclimate.core.diff import calc_gradient


@pytest.mark.parametrize("descending", [False, True])
def test_ncl_nonuniform_secants(descending):
    x = np.array([0.0, 1.0, 3.0, 6.0])
    if descending:
        x = x[::-1]
    data = xr.DataArray(x**2, dims="x", coords={"x": x}, attrs={"units": "test"})
    expected = xr.DataArray(
        [1.0, 3.0, 7.0, 9.0][::-1] if descending else [1.0, 3.0, 7.0, 9.0],
        dims="x",
        coords={"x": x},
    )
    actual = calc_gradient(data, "x", varargs=x, method="ncl")
    xr.testing.assert_identical(actual, expected)
    np.testing.assert_allclose(calc_gradient(data, "x", varargs=x), 2 * x)


@pytest.mark.parametrize("cyclic", [False, True])
def test_ncl_scalar_dataset_and_dask(cyclic):
    data = xr.DataArray(
        [30.0, 33.0, 39.0, 36.0, 41.0, 37.0], dims="x", coords={"x": np.arange(6)}
    )
    expected = [
        (-1.0 if cyclic else 1.5),
        2.25,
        0.75,
        0.5,
        0.25,
        (-2.75 if cyclic else -2.0),
    ]
    actual = calc_gradient(data, "x", varargs=2.0, method="ncl", cyclic=cyclic)
    np.testing.assert_allclose(actual, expected)
    ds = xr.Dataset({"a": data, "b": data * 2}).chunk({"x": 2})
    lazy = calc_gradient(ds, "x", varargs=2.0, method="ncl", cyclic=cyclic)
    assert lazy.a.chunks is not None
    xr.testing.assert_equal(lazy.a.compute(), actual)
    xr.testing.assert_equal(lazy.b.compute(), actual * 2)


def test_ncl_periodic_nonuniform_and_dimension_order():
    data = xr.DataArray(
        [[0.0, 2.0, 8.0, 18.0], [0.0, 4.0, 16.0, 36.0]],
        dims=("y", "x"),
        coords={"x": [0.0, 1.0, 3.0, 6.0]},
    )
    actual = calc_gradient(data, "x", varargs=data.x, method="ncl", cyclic=True)
    assert actual.dims == data.dims
    np.testing.assert_allclose(
        actual, [[-8.0, 8 / 3, 16 / 5, -4 / 3], [-16.0, 16 / 3, 32 / 5, -8 / 3]]
    )


def test_ncl_missing_neighbors():
    data = xr.DataArray([0.0, 1.0, np.nan, 9.0, 16.0], dims="x")
    result = calc_gradient(data, "x", method="ncl")
    np.testing.assert_allclose(result, [1.0, np.nan, 4.0, np.nan, 7.0], equal_nan=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "bad"},
        {"cyclic": True},
        {"method": "ncl", "varargs": 0},
        {"method": "ncl", "varargs": [0, 1]},
        {"method": "ncl", "varargs": [0, 1, 1]},
    ],
)
def test_gradient_invalid_options(kwargs):
    with pytest.raises(ValueError):
        calc_gradient(xr.DataArray([0.0, 1.0, 4.0], dims="x"), "x", **kwargs)


@pytest.fixture
def geographic_field():
    lon = np.array([0.0, 60.0, 150.0, 240.0, 300.0])
    lat = np.array([20.0, 30.0, 50.0, 65.0])
    return xr.DataArray(
        np.cos(np.deg2rad(lon))[None, :] * lat[:, None] ** 2,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


def secant(values, coordinates, axis, cyclic=False):
    a = np.moveaxis(np.asarray(values), axis, -1)
    x = np.asarray(coordinates)
    result = np.empty_like(a, dtype=float)
    result[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / (x[2:] - x[:-2])
    if cyclic:
        result[..., 0] = (a[..., 1] - a[..., -1]) / (2 * (x[1] - x[0]))
        result[..., -1] = (a[..., 0] - a[..., -2]) / (2 * (x[-1] - x[-2]))
    else:
        result[..., 0] = (a[..., 1] - a[..., 0]) / (x[1] - x[0])
        result[..., -1] = (a[..., -1] - a[..., -2]) / (x[-1] - x[-2])
    return np.moveaxis(result, -1, axis)


@pytest.mark.parametrize(
    "name,axis,unit",
    [
        ("calc_dx_gradient", 1, "meter"),
        ("calc_dy_gradient", 0, "meter"),
        ("calc_dlon_radian_gradient", 1, "radian"),
        ("calc_dlat_radian_gradient", 0, "radian"),
        ("calc_dlon_degree_gradient", 1, "degree"),
        ("calc_dlat_degree_gradient", 0, "degree"),
    ],
)
@pytest.mark.parametrize("cyclic", [False, True])
def test_geographic_ncl_gradients(geographic_field, name, axis, unit, cyclic):
    from easyclimate.core import diff

    data = geographic_field
    coordinate = data.lon if axis == 1 else data.lat
    x = np.asarray(coordinate) if unit == "degree" else np.deg2rad(coordinate)
    expected = secant(data, x, axis, cyclic)
    if unit == "meter":
        expected /= 6371200.0
        if axis == 1:
            expected /= np.cos(np.deg2rad(data.lat.values))[:, None]
    actual = getattr(diff, name)(data, method="ncl", cyclic=cyclic)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-12)


@pytest.mark.parametrize("cyclic", [False, True])
def test_ncl_laplacians_mixed_and_deformation(geographic_field, cyclic):
    from easyclimate.core import diff

    data = geographic_field
    lon, lat = np.deg2rad(data.lon), np.deg2rad(data.lat)
    radius = 6371200.0
    coslat = np.cos(lat.values)[:, None]
    dx = secant(data, lon, 1, cyclic)
    dy = secant(data, lat, 0)
    xx = secant(dx, lon, 1, cyclic) / (radius * coslat) ** 2
    yy = secant(dy, lat, 0) / radius**2
    xy = secant(dy, lon, 1, cyclic) / (radius**2 * coslat)
    np.testing.assert_allclose(
        diff.calc_dx_laplacian(data, method="ncl", cyclic=cyclic),
        xx,
        rtol=1e-13,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        diff.calc_dy_laplacian(data, method="ncl"), yy, rtol=1e-13, atol=1e-20
    )
    np.testing.assert_allclose(
        diff.calc_dxdy_mixed_derivatives(data, method="ncl", cyclic=cyclic),
        xy,
        rtol=1e-13,
        atol=1e-20,
    )
    for spherical in (False, True):
        expected = xx + yy
        if spherical:
            expected -= dy * np.tan(lat.values)[:, None] / radius**2
        np.testing.assert_allclose(
            diff.calc_dxdy_laplacian(
                data, method="ncl", cyclic=cyclic, spherical_coord=spherical
            ),
            expected,
            rtol=1e-13,
            atol=1e-20,
        )
    deformation = diff.calc_shear_stretch_deform(
        data, data * 2, method="ncl", cyclic=cyclic
    )
    shear = 2 * dx / (radius * coslat) + dy / radius
    stretch = dx / (radius * coslat) - 2 * dy / radius
    np.testing.assert_allclose(deformation.shear, shear)
    np.testing.assert_allclose(deformation.stretch, stretch)
    np.testing.assert_allclose(deformation.deform, np.hypot(shear, stretch))


def test_pressure_and_time_ncl():
    from easyclimate.core.diff import calc_p_gradient, calc_time_gradient

    pressure = np.array([1000.0, 800.0, 500.0, 200.0])
    data = xr.DataArray(pressure**2, dims="level", coords={"level": pressure})
    expected = secant(data[::-1], pressure[::-1] * 100, 0)
    np.testing.assert_allclose(
        calc_p_gradient(data, "level", "hPa", method="ncl"), expected
    )
    time = xr.DataArray([0.0, 1.0, 4.0, 9.0], dims="time")
    np.testing.assert_allclose(
        calc_time_gradient(time, "days", method="ncl"),
        np.array([1.0, 2.0, 4.0, 5.0]) / 86400,
    )


@pytest.mark.parametrize(
    "name",
    [
        "calc_dx_gradient",
        "calc_dy_gradient",
        "calc_dlon_radian_gradient",
        "calc_dlon_degree_gradient",
        "calc_dlat_radian_gradient",
        "calc_dlat_degree_gradient",
        "calc_dx_laplacian",
        "calc_dy_laplacian",
        "calc_dxdy_mixed_derivatives",
        "calc_dxdy_laplacian",
    ],
)
def test_wrapper_defaults_and_invalid_cyclic(geographic_field, name):
    from easyclimate.core import diff

    function = getattr(diff, name)
    xr.testing.assert_identical(
        function(geographic_field),
        function(geographic_field, method="numpy", cyclic=False),
    )
    with pytest.raises(ValueError, match="requires method"):
        function(geographic_field, cyclic=True)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "uint16"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("cyclic", [False, True])
def test_overlap_multichunk_boundaries(dtype, axis, cyclic):
    import dask

    shape = (7, 7, 7)
    raw = (np.arange(np.prod(shape)).reshape(shape) % 19).astype(dtype)
    if np.dtype(dtype).kind == "f":
        raw[2, 3, 4] = np.nan
    data = xr.DataArray(
        raw, dims=("a", "b", "c"), name="temperature", attrs={"units": "K"}
    )
    dim = data.dims[axis]
    # Descending, irregular, independent of the label coordinates.
    coordinate = -np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
    expected = secant(raw.astype(float), coordinate, axis, cyclic)
    eager = calc_gradient(data, dim, coordinate, method="ncl", cyclic=cyclic)
    chunked = data.chunk({d: (1, 2, 3, 1) for d in data.dims})
    with dask.callbacks.Callback(
        pretask=lambda *args: pytest.fail("Eager Dask task during graph construction")
    ):
        lazy = calc_gradient(chunked, dim, coordinate, method="ncl", cyclic=cyclic)
    assert lazy.chunks == chunked.chunks
    assert lazy.name == data.name
    assert lazy.attrs == {}
    assert lazy.dims == data.dims
    assert not lazy.coords
    np.testing.assert_allclose(eager, expected, rtol=0, atol=0, equal_nan=True)
    xr.testing.assert_identical(lazy.compute(), eager)


@pytest.mark.parametrize("method", ["numpy", "ncl"])
def test_gradient_dataset_coordinates_and_errors(method):
    data = xr.Dataset(
        {
            "a": ("x", [0.0, 1.0, 4.0, 9.0]),
            "b": (("y", "x"), [[0.0, 2.0, 8.0, 18.0]]),
            "static": ("y", [7.0]),
        },
        coords={"x": [0, 1, 2, 3], "y": [5], "station": ("y", ["A"]), "metadata": 42},
        attrs={"title": "input"},
    )
    result = calc_gradient(data, "x", method=method)
    assert set(result.data_vars) == {"a", "b"}
    xr.testing.assert_identical(
        result.coords.to_dataset(), xr.Dataset(coords=data.coords)
    )
    assert result.attrs == {}
    for name in result.data_vars:
        xr.testing.assert_identical(
            result[name], calc_gradient(data[name], "x", method=method)
        )
    with pytest.raises(ValueError, match="No variables"):
        calc_gradient(data[["static"]], "x", method=method)
    with pytest.raises(ValueError, match="method must"):
        calc_gradient(data, "x", method="invalid")
    with pytest.raises(ValueError, match="Dimension"):
        calc_gradient(data.a, "absent", method=method)
    with pytest.raises(ValueError):
        calc_gradient(data.isel(x=slice(0, 1)), "x", method=method)


def test_ncl_integer_subtraction_does_not_overflow():
    for raw in [
        np.array([32767, -32768, 32767], dtype="int16"),
        np.array([65535, 0, 65535], dtype="uint16"),
    ]:
        data = xr.DataArray(raw, dims="x")
        expected = np.array(
            [float(raw[1]) - float(raw[0]), 0.0, float(raw[2]) - float(raw[1])]
        )
        for field in (data, data.chunk({"x": 1})):
            np.testing.assert_array_equal(
                calc_gradient(field, "x", method="ncl").compute(), expected
            )
