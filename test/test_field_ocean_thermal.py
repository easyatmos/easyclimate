"""
pytest for field.ocean.thermal.py
"""

import pytest

import easyclimate as ecl
import numpy as np
import xarray as xr
from pathlib import Path
from .const_define import TEST_DATA_PATH

temper_data = xr.open_dataset(
    str(Path(TEST_DATA_PATH, "test_input_field_ocean_mixlayer_temper.nc"))
).temp.isel(time=1)


def test_calc_seawater_thermocline_depth():
    result_data = ecl.field.ocean.calc_seawater_thermocline_depth(
        seawater_temperature_data=temper_data
    ).data.flatten()
    refer_data = np.array(
        [
            82.18989944,
            40.46740150,
            61.05749893,
            50.71479988,
            40.46740150,
            61.05749893,
            20.16000032,
            10.06709981,
            71.53229904,
            np.nan,
            71.53229904,
            20.16000032,
            np.nan,
            np.nan,
            np.nan,
            61.05749893,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            10.06709981,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert np.allclose(result_data, refer_data, equal_nan=True)


def test_calc_D14_depth():
    result_data = ecl.field.ocean.calc_D14_depth(
        seawater_temperature_data=temper_data
    ).data.flatten()
    refer_data = np.array(
        [
            183.98488327,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            182.42414231,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            182.64443773,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert np.allclose(result_data, refer_data, equal_nan=True)


def test_calc_D17_depth():
    result_data = ecl.field.ocean.calc_D17_depth(
        seawater_temperature_data=temper_data
    ).data.flatten()
    refer_data = np.array(
        [
            124.93302659,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            127.49124545,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            126.37067400,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            125.90300132,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert np.allclose(result_data, refer_data, equal_nan=True)


def test_calc_D20_depth():
    result_data = ecl.field.ocean.calc_D20_depth(
        seawater_temperature_data=temper_data
    ).data.flatten()
    refer_data = np.array(
        [
            91.46646472,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            93.85737862,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            89.90620490,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            86.80035309,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert np.allclose(result_data, refer_data, equal_nan=True)


def test_calc_D26_depth():
    result_data = ecl.field.ocean.calc_D26_depth(
        seawater_temperature_data=temper_data
    ).data.flatten()
    refer_data = np.array(
        [
            47.85098618,
            np.nan,
            74.35317623,
            61.24003366,
            np.nan,
            51.49039806,
            np.nan,
            np.nan,
            72.63476218,
            np.nan,
            30.89619009,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            43.07298863,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert np.allclose(result_data, refer_data, equal_nan=True)


def test_calc_D28_depth():
    result_data = ecl.field.ocean.calc_D28_depth(
        seawater_temperature_data=temper_data
    ).data.flatten()
    refer_data = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            31.53331674,
            42.37996053,
            np.nan,
            np.nan,
            np.nan,
            40.76206133,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert np.allclose(result_data, refer_data, equal_nan=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_isotherm_crossings(reverse):
    temperature = xr.DataArray(
        [
            [22, 21, 18, 22],
            [19, 18, 17, 16],
            [24, 23, 22, 21],
            [20, 20, 18, 17],
            [22, np.nan, 18, 17],
            [np.nan] * 4,
        ],
        dims=("profile", "z"),
        coords={"z": [0, 100, 200, 300]},
    )
    if reverse:
        temperature = temperature.isel(z=slice(None, None, -1))
    temperature = temperature.chunk({"profile": 2, "z": 2})
    d20 = ecl.field.ocean.calc_D20_depth
    shallow = d20(temperature, depth_dim="z")
    deep = d20(temperature, depth_dim="z", method="deepest")
    np.testing.assert_allclose(
        shallow.compute(), [400 / 3, np.nan, np.nan, 0, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        deep.compute(), [250, np.nan, np.nan, 100, np.nan, np.nan]
    )


@pytest.mark.parametrize(("method", "expected"), [("layer", 25), ("centered", 10)])
def test_thermocline_is_profile_local(method, expected):
    temperature = xr.DataArray(
        [[30, 28, 14, 13], [30, 28, 26, 24], [20] * 4, [np.nan] * 4],
        dims=("profile", "depth"),
        coords={"depth": [0, 10, 40, 70]},
    )
    thermocline = ecl.field.ocean.calc_seawater_thermocline_depth
    alone = thermocline(temperature.isel(profile=0), method=method)
    chunked = temperature.isel(depth=slice(None, None, -1)).chunk({"depth": 2})
    together = thermocline(chunked, method=method).compute()
    assert float(alone) == expected
    assert float(together.isel(profile=0)) == expected
    assert together.isel(profile=slice(2, None)).isnull().all()
