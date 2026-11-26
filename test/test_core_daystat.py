import easyclimate as ecl
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        ("mean", [1.0, np.nan, 4.0]),
        ("sum", [2.0, np.nan, 8.0]),
        ("std", [1.0, np.nan, 2.0]),
        ("var", [1.0, np.nan, 4.0]),
        ("max", [2.0, np.nan, 6.0]),
        ("min", [0.0, np.nan, 2.0]),
    ],
)
def test_daily_statistics(statistic, expected):
    time = pd.to_datetime(
        ["2020-02-28 22:00", "2020-02-28 23:00", "2020-03-01 00:00", "2020-03-01 01:00"]
    )
    data = xr.DataArray(
        [[0.0, np.nan], [2.0, 2.0], [2.0, 2.0], [6.0, 6.0]],
        dims=["valid_time", "station"],
        coords={"valid_time": time, "station": ["A", "B"]},
        attrs={"units": "K"},
    )

    calculate = getattr(ecl, f"calc_daily_{statistic}")
    result = calculate(data, dim="valid_time", skipna=False, keep_attrs=True)

    expected_data = xr.DataArray(
        np.column_stack([expected, [np.nan, np.nan, expected[-1]]]),
        dims=data.dims,
        coords={
            "valid_time": pd.date_range("2020-02-28", periods=3, freq="D"),
            "station": data.station,
        },
        attrs=data.attrs,
    )
    xr.testing.assert_identical(result, expected_data)


def test_daily_mean_with_cftime_calendar():
    time = xr.date_range(
        "2000-02-30T23:00", periods=3, freq="h", calendar="360_day", use_cftime=True
    )
    data = xr.DataArray([2.0, 4.0, 6.0], dims=["time"], coords={"time": time})

    result = ecl.calc_daily_mean(data)

    expected_time = xr.date_range(
        "2000-02-30", periods=2, freq="D", calendar="360_day", use_cftime=True
    )
    expected = xr.DataArray([2.0, 5.0], dims=["time"], coords={"time": expected_time})
    xr.testing.assert_identical(result, expected)
