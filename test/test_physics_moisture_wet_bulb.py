"""
pytest for physics/moisture/wet_bulb.py
"""

import pytest
import numpy as np
import xarray as xr
from easyclimate.physics.moisture.wet_bulb import (
    calc_wet_bulb_temperature_iteration,
    calc_wet_bulb_potential_temperature_iteration,
    calc_wet_bulb_potential_temperature_davies_jones2008,
    calc_wet_bulb_temperature_stull2011,
    calc_wet_bulb_temperature_sadeghi2013,
)


@pytest.mark.parametrize("method", ["easyclimate-rust", "easyclimate-backend"])
def test_wet_bulb_iteration_relative_humidity_units(method):
    temperature = xr.DataArray(20.0)
    pressure = xr.DataArray(1013.25)

    percent = calc_wet_bulb_temperature_iteration(
        temperature, xr.DataArray(50.0), pressure, "celsius", "%", "hPa", method=method
    )
    fraction = calc_wet_bulb_temperature_iteration(
        temperature,
        xr.DataArray(0.5),
        pressure,
        "celsius",
        "dimensionless",
        "hPa",
        method=method,
    )

    assert percent.item() == pytest.approx(13.84156164, abs=1e-4)
    xr.testing.assert_identical(fraction, percent)


@pytest.mark.parametrize("method", ["easyclimate-rust", "easyclimate-backend"])
def test_wet_bulb_iteration_convergence_at_limit(method):
    def calculate(humidity):
        return calc_wet_bulb_temperature_iteration(
            xr.DataArray(20.0),
            xr.DataArray(humidity),
            xr.DataArray(1013.25),
            "celsius",
            "%",
            "hPa",
            max_iter=1,
            method=method,
        ).item()

    assert calculate(100.0) == 20.0
    assert calculate(99.999) == pytest.approx(19.99988961, abs=1e-5)
    with pytest.raises(RuntimeError, match="converge"):
        calculate(50.0)


@pytest.mark.parametrize("method", ["easyclimate-rust", "easyclimate-backend"])
def test_wet_bulb_iteration_ice(method):
    temperature = xr.DataArray([5.0, -10.0, np.nan], dims="point")
    humidity = xr.DataArray([20.0, 100.0, 50.0], dims="point")
    pressure = xr.DataArray(1000.0)
    result = calc_wet_bulb_temperature_iteration(
        temperature,
        humidity,
        pressure,
        "celsius",
        "%",
        "hPa",
        method=method,
        phase="ice",
    )

    assert result[0].item() == pytest.approx(-1.35860392, abs=0.01)
    # Water-saturated air below freezing is supersaturated relative to ice.
    assert -10.0 < result[1].item() < 0.0
    assert np.isnan(result[2].item())
    assert result.attrs["A"] == 0.000584
    with pytest.raises(ValueError, match="ice-bulb solution"):
        calc_wet_bulb_temperature_iteration(
            xr.DataArray(20.0),
            xr.DataArray(50.0),
            pressure,
            "celsius",
            "%",
            "hPa",
            method=method,
            phase="ice",
        )


@pytest.mark.parametrize("bad_pressure", [0.0, -10.0, np.inf])
def test_wet_bulb_iteration_rejects_invalid_pressure(bad_pressure):
    with pytest.raises(ValueError, match="pressure"):
        calc_wet_bulb_temperature_iteration(
            xr.DataArray(20.0),
            xr.DataArray(50.0),
            xr.DataArray(bad_pressure),
            "celsius",
            "%",
            "hPa",
        )


@pytest.mark.parametrize("parameter", ["A", "tolerance"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_wet_bulb_iteration_rejects_nonfinite_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        calc_wet_bulb_temperature_iteration(
            xr.DataArray(20.0),
            xr.DataArray(50.0),
            xr.DataArray(1000.0),
            "celsius",
            "%",
            "hPa",
            **{parameter: value},
        )


class TestCalcWetBulbPotentialTemperatureIteration:
    """Test cases for calc_wet_bulb_potential_temperature_iteration function."""

    def test_basic_calculation(self):
        """Test basic calculation with simple inputs."""
        temp = xr.DataArray(np.array([20, 25, 30]), dims=["point"])
        rh = xr.DataArray(np.array([50, 60, 70]), dims=["point"])
        pressure = xr.DataArray(np.array([1000, 950, 900]), dims=["point"])

        result = calc_wet_bulb_potential_temperature_iteration(
            temperature_data=temp,
            relative_humidity_data=rh,
            pressure_data=pressure,
            temperature_data_units="celsius",
            relative_humidity_data_units="%",
            pressure_data_units="hPa",
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("point",)
        assert result.attrs["units"] == "degC"

    def test_2d_input(self):
        """Test with 2D input data."""
        temp_2d = xr.DataArray(np.random.rand(2, 2) * 30, dims=["lat", "lon"])
        rh_2d = xr.DataArray(np.random.rand(2, 2) * 100, dims=["lat", "lon"])
        pres_2d = xr.DataArray(np.random.rand(2, 2) * 200 + 800, dims=["lat", "lon"])

        result = calc_wet_bulb_potential_temperature_iteration(
            temp_2d, rh_2d, pres_2d, "celsius", "%", "hPa"
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("lat", "lon")

    def test_unit_conversions(self):
        """Test that unit conversions work properly."""
        temp = xr.DataArray(
            np.array([293.15, 298.15]), dims=["point"]
        )  # 20C, 25C in Kelvin
        rh = xr.DataArray(
            np.array([0.5, 0.6]), dims=["point"]
        )  # 50%, 60% as dimensionless
        pressure = xr.DataArray(
            np.array([100000, 95000]), dims=["point"]
        )  # 1000hPa, 950hPa in Pa

        result = calc_wet_bulb_potential_temperature_iteration(
            temperature_data=temp,
            relative_humidity_data=rh,
            pressure_data=pressure,
            temperature_data_units="kelvin",
            relative_humidity_data_units="dimensionless",
            pressure_data_units="Pa",
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["units"] == "degC"


class TestCalcWetBulbPotentialTemperatureDaviesJones2008:
    """Test cases for calc_wet_bulb_potential_temperature_davies_jones2008 function."""

    def test_basic_calculation(self):
        """Test basic calculation with simple inputs."""
        pressure = xr.DataArray(np.array([1000, 950]), dims=["point"])
        temp = xr.DataArray(np.array([293.15, 298.15]), dims=["point"])  # 20C, 25C
        dewpoint = xr.DataArray(np.array([283.15, 288.15]), dims=["point"])  # 10C, 15C

        result = calc_wet_bulb_potential_temperature_davies_jones2008(
            pressure_data=pressure,
            temperature_data=temp,
            dewpoint_data=dewpoint,
            pressure_data_units="hPa",
            temperature_data_units="kelvin",
            dewpoint_data_units="kelvin",
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("point",)
        assert result.attrs["units"] == "K"

    def test_unit_conversions(self):
        """Test that unit conversions work properly."""
        pressure = xr.DataArray(
            np.array([100000, 95000]), dims=["point"]
        )  # 1000hPa, 950hPa in Pa
        temp = xr.DataArray(np.array([20, 25]), dims=["point"])  # in Celsius
        dewpoint = xr.DataArray(np.array([10, 15]), dims=["point"])  # in Celsius

        result = calc_wet_bulb_potential_temperature_davies_jones2008(
            pressure_data=pressure,
            temperature_data=temp,
            dewpoint_data=dewpoint,
            pressure_data_units="Pa",
            temperature_data_units="celsius",
            dewpoint_data_units="celsius",
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["units"] == "K"


class TestCalcWetBulbTemperatureStull2011:
    """Test cases for calc_wet_bulb_temperature_stull2011 function."""

    def test_basic_calculation(self):
        """Test basic calculation with simple inputs."""
        temp = xr.DataArray(np.array([20, 25]), dims=["point"])
        rh = xr.DataArray(np.array([50, 60]), dims=["point"])

        result = calc_wet_bulb_temperature_stull2011(
            temperature_data=temp,
            relative_humidity_data=rh,
            temperature_data_units="celsius",
            relative_humidity_data_units="%",
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("point",)
        assert result.attrs["units"] == "K"

    def test_unit_conversions(self):
        """Test that unit conversions work properly."""
        temp = xr.DataArray(
            np.array([293.15, 298.15]), dims=["point"]
        )  # 20C, 25C in Kelvin
        rh = xr.DataArray(
            np.array([0.5, 0.6]), dims=["point"]
        )  # 50%, 60% as dimensionless

        result = calc_wet_bulb_temperature_stull2011(
            temperature_data=temp,
            relative_humidity_data=rh,
            temperature_data_units="kelvin",
            relative_humidity_data_units="dimensionless",
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["units"] == "K"


class TestCalcWetBulbTemperatureSadeghi2013:
    """Test cases for calc_wet_bulb_temperature_sadeghi2013 function."""

    @pytest.mark.parametrize(
        "units, scale, offset",
        [("celsius", 1, 0), ("kelvin", 1, 273.15), ("fahrenheit", 1.8, 32)],
    )
    def test_validated_dry_bulb_temperature_range(self, units, scale, offset):
        temperature = xr.DataArray(
            [-30.0, -17.01, -17.0, 20.0, 40.0, 40.01, 45.0, np.nan], dims="point"
        )
        result = calc_wet_bulb_temperature_sadeghi2013(
            temperature * scale + offset,
            xr.DataArray(0),
            xr.DataArray(50.0),
            units,
            "m",
            "%",
        )

        np.testing.assert_array_equal(
            result.isnull(), [True, True, False, False, False, True, True, True]
        )
        assert result.isel(point=2).item() == pytest.approx(-18.407658)

    def test_paper_calibration(self):
        temperature = xr.DataArray([20.0, 20.0, 2.0, -10.0], dims="point")
        height = xr.DataArray([0, 4500, 3500, 0], dims="point")
        humidity = xr.DataArray([50.0, 50.0, 23.0, 50.0], dims="point")

        result = calc_wet_bulb_temperature_sadeghi2013(
            temperature, height, humidity, "celsius", "m", "%"
        )

        # Reference values from Eq. (5a), Table 1, Eqs. (8)-(12), and Fig. 3.
        np.testing.assert_allclose(
            result.values,
            [12.710055, 11.610553, -6.569564, -12.498645],
            rtol=0,
            atol=1e-6,
        )

    def test_basic_calculation(self):
        """Test basic calculation with simple inputs."""
        temp = xr.DataArray(np.array([20, 25]), dims=["point"])
        height = xr.DataArray(np.array([100, 200]), dims=["point"])
        rh = xr.DataArray(np.array([50, 60]), dims=["point"])

        result = calc_wet_bulb_temperature_sadeghi2013(
            temperature_data=temp,
            height_data=height,
            relative_humidity_data=rh,
            temperature_data_units="celsius",
            height_data_units="m",
            relative_humidity_data_units="%",
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("point",)
        assert result.attrs["units"] == "degC"

    def test_unit_conversions(self):
        """Test that unit conversions work properly."""
        temp = xr.DataArray(
            np.array([293.15, 298.15]), dims=["point"]
        )  # 20C, 25C in Kelvin
        height = xr.DataArray(np.array([0.1, 0.2]), dims=["point"])  # 100m, 200m in km
        rh = xr.DataArray(
            np.array([0.5, 0.6]), dims=["point"]
        )  # 50%, 60% as dimensionless

        result = calc_wet_bulb_temperature_sadeghi2013(
            temperature_data=temp,
            height_data=height,
            relative_humidity_data=rh,
            temperature_data_units="kelvin",
            height_data_units="km",
            relative_humidity_data_units="dimensionless",
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["units"] == "degC"
