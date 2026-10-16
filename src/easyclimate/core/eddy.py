"""
The calculation of transient eddy
"""

from __future__ import annotations
import xarray as xr
import numpy as np
from ..physics.geo import get_coriolis_parameter
from ..physics.temperature import calc_potential_temperature_vertical
from ..physics.convection.stability import calc_brunt_vaisala_frequency_atm
from .diff import (
    calc_gradient,
    calc_dx_gradient,
    calc_dy_gradient,
    calc_dlon_radian_gradient,
    calc_dlat_radian_gradient,
)
from .utility import transfer_deg2rad
from .units import (
    transfer_data_multiple_units,
    transfer_units_coeff,
    transfer_data_temperature_units,
)
from typing import Literal

__all__ = [
    "calc_eady_growth_rate",
    "calc_apparent_heat_source",
    "calc_total_diabatic_heating",
    "calc_apparent_moisture_sink",
    "calc_Plumb_wave_activity_horizontal_flux",
    "calc_TN_wave_activity_horizontal_flux",
    "calc_TN_wave_activity_vertical_flux",
    "calc_TN_wave_activity_3D_flux",
    "calc_EP_horizontal_flux",
    "calc_monthly_rossby_wave_source",
]


def calc_eady_growth_rate(
    u_daily_data: xr.DataArray,
    z_daily_data: xr.DataArray,
    temper_daily_data: xr.DataArray,
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    lat_dim: str = "lat",
) -> xr.Dataset:
    """
    Calculate the maximum Eady growth rate.

    .. math::
        \\sigma = 0.3098 \\frac{f}{N} \\frac{\\mathrm{d} U}{\\mathrm{d} z}

    .. caution::
        Eady growth rate (EGR) is a non-linear quantity. Hence, `calc_eady_growth_rate` should **NOT** be **directly applied to monthly means** variables.
        If a monthly climatology of EGR is desired, the EGR values at the high frequency temporal time steps should be calculated;
        then, use calculate monthly mean.

    Parameters
    ----------
    u_daily_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The zonal wind daily data.
    z_daily_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Daily atmospheric geopotential height.

    .. attention:: The unit of `z_daily_data` should be **meters**, NOT :math:`\\mathrm{m^2 \\cdot s^2}` which is the unit used in the representation of potential energy.

    temper_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Daily air temperature.
    vertical_dim: :py:class:`str <str>`.
        Vertical coordinate dimension name.
    vertical_dim_units: :py:class:`str <str>`.
        The unit corresponding to the vertical p-coordinate value. Optional values are `hPa`, `Pa`, `mbar`.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.

    Returns
    -------
    The maximum Eady growth rate (:py:class:`xarray.Dataset<xarray.Dataset>`).

    - `eady_growth_rate`: The maximum Eady growth rate.
    - `dudz`: :math:`\\frac{\\mathrm{d} U}{\\mathrm{d} z}`
    - `brunt_vaisala_frequency`: Brunt-väisälä frequency.

    .. seealso::
        - Eady, E. T. (1949). Long Waves and Cyclone Waves. Tellus, 1(3), 33–52. https://doi.org/10.3402/tellusa.v1i3.8507, https://www.tandfonline.com/doi/abs/10.3402/tellusa.v1i3.8507
        - Lindzen, R. S. , & Farrell, B. (1980). A Simple Approximate Result for the Maximum Growth Rate of Baroclinic Instabilities. Journal of Atmospheric Sciences, 37(7), 1648-1654. https://journals.ametsoc.org/view/journals/atsc/37/7/1520-0469_1980_037_1648_asarft_2_0_co_2.xml
        - Simmonds, I., and E.-P. Lim (2009), Biases in the calculation of Southern Hemisphere mean baroclinic eddy growth rate, Geophys. Res. Lett., 36, L01707, https://doi.org/10.1029/2008GL036320.
        - Sloyan, B. M., and T. J. O'Kane (2015), Drivers of decadal variability in the Tasman Sea, J. Geophys. Res. Oceans, 120, 3193–3210, https://doi.org/10.1002/2014JC010550.
        - `eady_growth_rate -NCL <https://www.ncl.ucar.edu/Document/Functions/Contributed/eady_growth_rate.shtml>`__
        - `瞬变涡旋诊断量 <https://renqlsysu.github.io/2020/02/16/wave_activity_flux/>`__

    .. ecl-minigallery::
        :add-heading: Example(s) related to the function

        egr
    """
    dim_tuple = u_daily_data.dims
    f = get_coriolis_parameter(u_daily_data[lat_dim])
    dp = 1
    dudp = calc_gradient(u_daily_data, dim=vertical_dim) / dp
    dzdp = calc_gradient(z_daily_data, dim=vertical_dim) / dp
    dudz = dudp / dzdp
    pt = calc_potential_temperature_vertical(
        temper_daily_data,
        vertical_dim=vertical_dim,
        vertical_dim_units=vertical_dim_units,
    )
    brunt_vaisala_atm = calc_brunt_vaisala_frequency_atm(
        pt, z_daily_data, vertical_dim=vertical_dim
    )
    eady_growth_rate = 0.3098 * np.abs(f) * np.abs(dudz) / brunt_vaisala_atm
    eady_growth_rate = eady_growth_rate.transpose(*dim_tuple)
    result_dataset = xr.Dataset(
        data_vars={
            "eady_growth_rate": eady_growth_rate,
            "dudz": dudz,
            "brunt_vaisala_frequency": brunt_vaisala_atm,
        }
    )
    return result_dataset


def calc_apparent_heat_source(
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    omega_data: xr.DataArray,
    temper_data: xr.DataArray,
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    time_units: str,
    lon_dim="lon",
    lat_dim="lat",
    time_dim="time",
    c_p=1005.7,
) -> xr.DataArray:
    """
    Calculate the apparent heat source.

    .. math::
        Q_1 = C_p \\frac{T}{\\theta} \\left( \\frac{\\partial \\theta}{\\partial t} + u \\frac{\\partial \\theta}{\\partial x} + v \\frac{\\partial \\theta}{\\partial y} + \\omega \\frac{\\partial \\theta}{\\partial p} \\right)

    Parameters
    ----------
    u_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The zonal wind data.
    v_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The meridional wind data.
    omega_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The vertical velocity data (:math:`\\frac{\\mathrm{d} p}{\\mathrm{d} t}`).
    temper_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Air temperature.
    vertical_dim: :py:class:`str <str>`.
        Vertical coordinate dimension name.
    vertical_dim_units: :py:class:`str <str>`.
        The unit corresponding to the vertical p-coordinate value. Optional values are `hPa`, `Pa`, `mbar`.
    time_units: :py:class:`str <str>`.
        The unit corresponding to the time dimension value. Optional values are `seconds`, `months`, `years` and so on.
    lon_dim: :py:class:`str <str>`, default: `lon`.
        Longitude coordinate dimension name. By default extracting is applied over the `lon` dimension.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.
    time_dim: :py:class:`str <str>`.
        The time coordinate dimension name.
    c_p: :py:class:`float <float>`, default: `1005.7`.
        The specific heat at constant pressure of dry air.

        .. note::
            `specific heat capacity - Glossary of Meteorology <https://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__

    Returns
    -------
    The apparent heat source (:py:class:`xarray.DataArray<xarray.DataArray>`).

    .. seealso::
        - `Yanai, M., & Tomita, T. (1998). Seasonal and Interannual Variability of Atmospheric Heat Sources and Moisture Sinks as Determined from NCEP–NCAR Reanalysis, Journal of Climate, 11(3), 463-482. <https://journals.ametsoc.org/view/journals/clim/11/3/1520-0442_1998_011_0463_saivoa_2.0.co_2.xml>`__
        - `Ling, J., & Zhang, C. (2013). Diabatic Heating Profiles in Recent Global Reanalyses, Journal of Climate, 26(10), 3307-3325. <https://doi.org/10.1175/JCLI-D-12-00384.1>`__
    """
    # Convert time units to seconds
    dt = transfer_units_coeff(time_units, "seconds")
    # Convert the pressure unit to Pascal
    dp_base = transfer_units_coeff(vertical_dim_units, "Pa")

    pt = calc_potential_temperature_vertical(
        temper_data, vertical_dim=vertical_dim, vertical_dim_units=vertical_dim_units
    )
    dtheta_dt = calc_gradient(pt, dim=time_dim) / dt
    dtheta_dx = calc_dx_gradient(pt, lon_dim=lon_dim, lat_dim=lat_dim)
    dtheta_dy = calc_dy_gradient(pt, lat_dim=lat_dim)

    dp = calc_gradient(pt[vertical_dim], dim=vertical_dim) * dp_base
    domega_dp = calc_gradient(pt, dim=vertical_dim) / dp

    Q1 = (
        c_p
        * (temper_data / pt)
        * (dtheta_dt + u_data * dtheta_dx + v_data * dtheta_dy + omega_data * domega_dp)
    )
    Q1.name = "apparent_heat_source"
    return Q1


def calc_total_diabatic_heating(
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    omega_data: xr.DataArray,
    temper_data: xr.DataArray,
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    time_units: str,
    lat_dim="lat",
    lon_dim="lon",
    time_dim="time",
    c_p=1005.7,
) -> xr.DataArray:
    """
    Calculate the total diabatic heating.

    Calculated in exactly the same way as for the apparent heat source.

    Parameters
    ----------
    u_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The zonal wind data.
    v_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The meridional wind data.
    omega_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The vertical velocity data (:math:`\\frac{\\mathrm{d} p}{\\mathrm{d} t}`).
    temper_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Air temperature.
    vertical_dim: :py:class:`str <str>`.
        Vertical coordinate dimension name.
    vertical_dim_units: :py:class:`str <str>`.
        The unit corresponding to the vertical p-coordinate value. Optional values are `hPa`, `Pa`, `mbar`.
    time_units: :py:class:`str <str>`.
        The unit corresponding to the time dimension value. Optional values are `seconds`, `months`, `years` and so on.
    lon_dim: :py:class:`str <str>`, default: `lon`.
        Longitude coordinate dimension name. By default extracting is applied over the `lon` dimension.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.
    time_dim: :py:class:`str <str>`, default: `time`.
        The time coordinate dimension name.
    c_p: :py:class:`float <float>`, default: `1005.7` (:math:`\\mathrm{J \\cdot kg^{-1} \\cdot K^{-1}}`).
        The specific heat at constant pressure of dry air.

        .. note::
            `specific heat capacity - Glossary of Meteorology <https://glossary.ametsoc.org/wiki/Specific_heat_capacity>`__

    Returns
    -------
    The total diabatic heating (:py:class:`xarray.DataArray<xarray.DataArray>`).

    .. seealso::
        :py:func:`calc_apparent_heat_source <calc_apparent_heat_source>`
    """
    Q1 = calc_apparent_heat_source(
        u_data=u_data,
        v_data=v_data,
        omega_data=omega_data,
        temper_data=temper_data,
        vertical_dim=vertical_dim,
        vertical_dim_units=vertical_dim_units,
        time_units=time_units,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        time_dim=time_dim,
        c_p=c_p,
    )
    Q1.name = "total_diabatic_heating"
    return Q1


def calc_apparent_moisture_sink(
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    omega_data: xr.DataArray,
    specific_humidity_data: xr.DataArray,
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    time_units: str,
    specific_humidity_data_units: str,
    lon_dim="lon",
    lat_dim="lat",
    time_dim="time",
    latent_heat_of_condensation=2.501e6,
) -> xr.DataArray:
    """
    Calculate the apparent moisture sink.

    .. math::
        Q_2 = -L \\left( \\frac{\\partial q}{\\partial t} + u \\frac{\\partial q}{\\partial x} + v \\frac{\\partial q}{\\partial y} + \\omega \\frac{\\partial q}{\\partial p}  \\right)

    Parameters
    ----------
    u_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The zonal wind data.
    v_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The meridional wind data.
    omega_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The vertical velocity data (:math:`\\frac{\\mathrm{d} p}{\\mathrm{d} t}`).
    specific_humidity_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The absolute humidity data.
    vertical_dim: :py:class:`str <str>`.
        Vertical coordinate dimension name.
    vertical_dim_units: :py:class:`str <str>`.
        The unit corresponding to the vertical p-coordinate value. Optional values are `hPa`, `Pa`, `mbar`.
    time_units: :py:class:`str <str>`.
        The unit corresponding to the time dimension value. Optional values are `seconds`, `months`, `years` and so on.
    specific_humidity_data_units: :py:class:`str <str>`.
        The unit corresponding to `specific_humidity` value. Optional values are `kg/kg`, `g/kg` and so on.
    lon_dim: :py:class:`str <str>`, default: `lon`.
        Longitude coordinate dimension name. By default extracting is applied over the `lon` dimension.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.
    time_dim: :py:class:`str <str>`, default: `time`.
        The time coordinate dimension name.
    latent_heat_of_condensation: :py:class:`float <float>`, default: `2.5008e6` (:math:`\\mathrm{J \\cdot kg^{-1}}`).
        Latent heat of condensation of water at 0°C.

        .. note::
            - `latent heat - Glossary of Meteorology <https://glossary.ametsoc.org/wiki/Latent_heat>`__
            - `Latent heat - Wikipedia <https://en.wikipedia.org/wiki/Latent_heat>`__

    Returns
    -------
    The apparent moisture sink (:py:class:`xarray.DataArray<xarray.DataArray>`).

    .. seealso::
        - `Yanai, M., & Tomita, T. (1998). Seasonal and Interannual Variability of Atmospheric Heat Sources and Moisture Sinks as Determined from NCEP–NCAR Reanalysis, Journal of Climate, 11(3), 463-482. <https://journals.ametsoc.org/view/journals/clim/11/3/1520-0442_1998_011_0463_saivoa_2.0.co_2.xml>`__
        - `HAO Lisheng, MA Ning, HE Liye. Circulation anomalies characteritics of the abnormal drought and high temperature event in the middle and lower reaches of the Yangtze River in summer of 2022[J]. Arid Meteorology, 2022, 40(5): 721-732 <https://doi.org/10.11755/j.issn.1006-7639(2022)-05-0721>`__
    """
    specific_humidity_data = transfer_data_multiple_units(
        specific_humidity_data, specific_humidity_data_units, "kg/kg"
    )

    # Convert time units to seconds
    dt = transfer_units_coeff(time_units, "seconds")
    # Convert the pressure unit to Pascal
    dp_base = transfer_units_coeff(vertical_dim_units, "Pa")

    dqs_dt = calc_gradient(specific_humidity_data, dim=time_dim) / dt
    dqs_dx = calc_dx_gradient(specific_humidity_data, lon_dim=lon_dim, lat_dim=lat_dim)
    dqs_dy = calc_dy_gradient(specific_humidity_data, lat_dim=lat_dim)

    dp = calc_gradient(specific_humidity_data[vertical_dim], dim=vertical_dim) * dp_base
    domega_dp = calc_gradient(specific_humidity_data, dim=vertical_dim) / dp

    Q2 = (
        (-1)
        * latent_heat_of_condensation
        * (dqs_dt + u_data * dqs_dx + v_data * dqs_dy + omega_data * domega_dp)
    )
    Q2.name = "apparent_moisture_sink"
    return Q2


def calc_Plumb_wave_activity_horizontal_flux(
    z_prime_data: xr.DataArray,
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    lon_dim="lon",
    lat_dim="lat",
    omega=7.292e-5,
    g=9.8,
    R=6370000,
) -> xr.Dataset:
    """
    Calculate Plumb wave activity horizontal flux.

    Parameters
    ----------
    z_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The anormaly of atmospheric geopotential height.
    vertical_dim: :py:class:`str <str>`.
        Vertical coordinate dimension name.
    vertical_dim_units: :py:class:`str <str>`.
        The unit corresponding to the vertical p-coordinate value. Optional values are `hPa`, `Pa`, `mbar`.
    lon_dim: :py:class:`str <str>`, default: `lon`.
        Longitude coordinate dimension name. By default extracting is applied over the `lon` dimension.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.
    omega: :py:class:`float <float>`, default: `7.292e-5`.
        The angular speed of the earth.
    g: :py:class:`float <float>`, default: `9.8`.
        The acceleration of gravity.
    R: :py:class:`float <float>`, default: `6370000`.
        Radius of the Earth.

    Returns
    -------
    The Plumb wave activity horizontal flux (:py:class:`xarray.DataArray<xarray.DataArray>`).

    .. seealso::
        - `Plumb, R. A., 1985: On the Three-Dimensional Propagation of Stationary Waves. J. Atmos. Sci., 42, 217–229 <https://journals.ametsoc.org/view/journals/atsc/42/3/1520-0469_1985_042_0217_ottdpo_2_0_co_2.xml>`__
    """
    coordinate_sample_data = z_prime_data

    dim_tuple = z_prime_data.dims

    lat_array = coordinate_sample_data["lat"].astype("float64")
    coslat = np.cos(transfer_deg2rad(lat_array))

    f = get_coriolis_parameter(lat_array, omega=omega)
    psi_p = z_prime_data * g / f

    p_lev = transfer_data_multiple_units(
        coordinate_sample_data[vertical_dim], vertical_dim_units, "Pa"
    )
    p = p_lev / 1e5

    dpsi_dlambda = calc_dlon_radian_gradient(psi_p, lon_dim=lon_dim)
    dpsi_dphi = calc_dlat_radian_gradient(psi_p, lat_dim=lat_dim)

    d2psi_dlambda2 = calc_dlon_radian_gradient(dpsi_dlambda, lon_dim=lon_dim)
    d2psi_dlambdadphi = calc_dlon_radian_gradient(dpsi_dphi, lon_dim=lon_dim)

    term_xu = dpsi_dlambda**2 - (psi_p * d2psi_dlambda2)
    term_xv = dpsi_dlambda * dpsi_dphi - (psi_p * d2psi_dlambdadphi)

    fx = p * ((1 / (2 * R**2 * coslat)) * term_xu)
    fy = p * ((1 / (2 * R**2)) * term_xv)

    result = xr.Dataset()

    psi_p = psi_p.transpose(*dim_tuple)
    psi_p.attrs = {
        "long_name": "Perturbation stream function",
        "units": "m^2/s",
        "description": "Geostrophic stream function perturbation calculated from geopotential height anomaly",
    }
    result["psi_p"] = psi_p

    fx = fx.transpose(*dim_tuple)
    fx.attrs = {
        "long_name": "Zonal component of Plumb wave activity horizontal flux",
        "units": "m^2/s^2",
        "description": "Eastward component of Plumb wave activity flux",
        "standard_name": "tn_wave_activity_flux_x",
    }
    result["fx"] = fx

    fy = fy.transpose(*dim_tuple)
    fy.attrs = {
        "long_name": "Meridional component of Plumb wave activity horizontal flux",
        "units": "m^2/s^2",
        "description": "Northward component of Plumb wave activity flux",
        "standard_name": "Plumb_wave_activity_flux_y",
    }
    result["fy"] = fy

    result.attrs = {
        "title": "Plumb Wave Activity Flux",
        "description": "Horizontal components of quasi-geostrophic wave activity flux following Plumb (1985)",
        "reference": "Plumb, R. A., 1985: On the Three-Dimensional Propagation of Stationary Waves. J. Atmos. Sci., 42, 217–229",
        "created_with": "easyclimate: calc_Plumb_wave_activity_horizontal_flux function",
    }
    return result.astype("float32")


def _tn_climatology_for_sample(data, sample, time_dim):
    """Match an already aligned background or one climatological field per month."""
    # A single-level anomaly may keep pressure as a scalar coordinate while
    # the background still carries a level dimension. Select that same level
    # before month indexing, including when the background has multiple levels.
    for dim in tuple(data.dims):
        if (
            dim != time_dim
            and dim not in sample.dims
            and dim in sample.coords
            and sample[dim].ndim == 0
        ):
            data = data.sel({dim: sample[dim].item()}, drop=True)
    if time_dim in data.dims:
        if time_dim in sample.dims and data[time_dim].equals(sample[time_dim]):
            return data
        if data.sizes[time_dim] == 1 and time_dim not in sample.dims:
            return data.isel({time_dim: 0}, drop=True)
        months = data[time_dim].dt.month.values
        if len(np.unique(months)) != len(months):
            raise ValueError("Climatology must contain exactly one field per month.")
        data = data.assign_coords(month=(time_dim, months)).swap_dims(
            {time_dim: "month"}
        )
        data = data.drop_vars(time_dim)
    if "month" in data.dims:
        if time_dim not in sample.dims:
            raise ValueError("Select a climatology month for a sample without time.")
        # Index only by time/month: scalar level or member coordinates on the
        # anomaly are not indexing coordinates for the climatological month.
        months = sample[time_dim].dt.month.reset_coords(drop=True)
        data = data.sel(month=months)
        data = data.drop_vars("month")
    return data


def _tn_validate_cyclic_longitude(coordinate):
    """Require a uniform global longitude grid without a duplicate seam."""
    values = np.asarray(coordinate, dtype=float)
    steps = np.diff(values)
    if (
        values.size < 3
        or not np.allclose(steps, steps[0])
        or not np.isclose(abs(steps[0]) * values.size, 2 * np.pi)
    ):
        raise ValueError(
            "cyclic_lon requires a uniform global grid without a duplicate endpoint."
        )


def calc_TN_wave_activity_horizontal_flux(
    z_prime_data: xr.DataArray | None,
    u_climatology_data: xr.DataArray,
    v_climatology_data: xr.DataArray,
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    psi_prime_data: xr.DataArray | None = None,
    lon_dim: str = "lon",
    lat_dim: str = "lat",
    time_dim: str = "time",
    omega: float = 7.292e-5,
    g: float = 9.8,
    R: float = 6371200.0,
    cyclic_lon: bool = False,
    finite_difference: Literal["legacy", "ncl"] = "ncl",
    output_dtype: Literal["float32", "float64"] = "float32",
) -> xr.Dataset:
    """
    Calculate TN wave activity horizontal flux.

    .. math::
        \\mathbf{W_h} = \\frac{p\\cos\\varphi}{2\\lvert \\mathbf{U_c} \\rvert}\\begin{pmatrix}
                              \\frac{U_c}{R^2 \\cos^2 \\varphi} \\left[ \\left( \\frac{\\partial \\psi'}{\\partial \\lambda} \\right)^2 - \\psi'\\frac{\\partial^2 \\psi'}{\\partial \\lambda^2} \\right] + \\frac{V_c}{R^2 \\cos \\varphi} \\left[ \\frac{\\partial \\psi'}{\\partial \\lambda} \\frac{\\partial \\psi'}{\\partial \\varphi} - \\psi' \\frac{\\partial^2 \\psi'}{\\partial \\lambda \\partial \\varphi} \\right] \\\\
                              \\frac{U_c}{R^2 \\cos \\varphi} \\left[ \\frac{\\partial \\psi'}{\\partial \\lambda} \\frac{\\partial \\psi'}{\\partial \\varphi} - \\psi' \\frac{\\partial^2 \\psi'}{\\partial \\lambda \\partial \\varphi} \\right] + \\frac{V_c}{R^2} \\left[ \\left( \\frac{\\partial \\psi'}{\\partial \\varphi} \\right)^2 - \\psi'\\frac{\\partial^2 \\psi'}{\\partial \\varphi^2} \\right] \\\\
                               \\end{pmatrix}

    Parameters
    ----------
    z_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The anormaly of atmospheric geopotential height.

    .. attention:: The unit of `z_prime_data` should be **meters**, NOT :math:`\\mathrm{m^2 \\cdot s^2}` which is the unit used in the representation of potential energy.

    u_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The climatology of zonal wind data.
    v_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The climatology of meridional wind data.
    vertical_dim: :py:class:`str <str>`.
        Vertical coordinate dimension name.
    vertical_dim_units: :py:class:`str <str>`.
        The unit corresponding to the vertical p-coordinate value. Optional values are `hPa`, `Pa`, `mbar`.
    psi_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Perturbation stream function. Geostrophic stream function perturbation calculated from geopotential height anomaly.
    lon_dim: :py:class:`str <str>`, default: `lon`.
        Longitude coordinate dimension name. By default extracting is applied over the `lon` dimension.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.
    omega: :py:class:`float <float>`, default: `7.292e-5`.
        The angular speed of the earth.
    g: :py:class:`float <float>`, default: `9.8`.
        The acceleration of gravity.
    R: :py:class:`float <float>`, default: `6371200.0`.
        Radius of the Earth.
    cyclic_lon : :py:class:`bool <bool>`, default False
        Use periodic centred longitude differences on a uniform global grid
        without a duplicate endpoint.
    finite_difference : {"legacy", "ncl"}, default "ncl"
        Preserve the existing one-sided boundary differences, or use Nishii's
        NCL centred secants with first-order noncyclic endpoints. This option is
        also used by :func:`calc_TN_wave_activity_3D_flux`.
    output_dtype : {"float32", "float64"}, default "float32"
        Output precision. The NCL scheme computes in float64 before casting.
        The legacy gradient routines retain their historical float32 precision.

    Returns
    -------
    The TN wave activity horizontal flux (:py:class:`xarray.Dataset<xarray.Dataset>`).

    Reference
    --------------
    - Takaya, K., & Nakamura, H. (2001). A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow. Journal of the Atmospheric Sciences, 58(6), 608-627. https://journals.ametsoc.org/view/journals/atsc/58/6/1520-0469_2001_058_0608_afoapi_2.0.co_2.xml

    .. seealso::
        - http://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/index.html
        - https://github.com/laishenggx/T-N_Wave-Activity-Flux
    """
    # Check data validity
    if z_prime_data is None and psi_prime_data is None:
        raise ValueError(
            "`z_prime_data` and `psi_prime_data` should not both be None at the same time."
        )

    # An explicitly supplied streamfunction takes precedence.
    vertical_level_data = psi_prime_data if psi_prime_data is not None else z_prime_data

    # Verify data type
    if not isinstance(vertical_level_data, xr.DataArray):
        raise TypeError(
            f"The input data must be of the xarray.DataArray type. Current type:{type(vertical_level_data)}"
        )

    u_c = _tn_climatology_for_sample(u_climatology_data, vertical_level_data, time_dim)
    v_c = _tn_climatology_for_sample(v_climatology_data, vertical_level_data, time_dim)
    xr.align(vertical_level_data, u_c, v_c, join="exact", copy=False)
    if output_dtype not in ("float32", "float64"):
        raise ValueError("output_dtype must be 'float32' or 'float64'.")
    if finite_difference == "ncl":
        u_c, v_c = u_c.astype("float64"), v_c.astype("float64")
        vertical_level_data = vertical_level_data.astype("float64")

    coordinate_sample_data = vertical_level_data
    dim_tuple = vertical_level_data.dims

    lat_array = coordinate_sample_data[lat_dim].astype("float64")
    coslat = np.cos(transfer_deg2rad(lat_array))

    if psi_prime_data is None:
        f = get_coriolis_parameter(lat_array, omega=omega)
        psi_p = vertical_level_data * g / f
    else:
        psi_p = vertical_level_data

    p_lev = transfer_data_multiple_units(
        coordinate_sample_data[vertical_dim], vertical_dim_units, "Pa"
    )
    p = p_lev / 1e5

    if finite_difference not in ("legacy", "ncl"):
        raise ValueError("finite_difference must be 'legacy' or 'ncl'.")

    if cyclic_lon:
        _tn_validate_cyclic_longitude(psi_p[lon_dim].astype("float64") * np.pi / 180.0)

    def dlon(data):
        if cyclic_lon or finite_difference == "ncl":
            return calc_gradient(
                data,
                dim=lon_dim,
                varargs=data[lon_dim].astype("float64") * np.pi / 180.0,
                method="ncl",
                cyclic=cyclic_lon,
            )
        return calc_dlon_radian_gradient(data, lon_dim=lon_dim)

    def dlat(data):
        if finite_difference == "ncl":
            return calc_gradient(
                data,
                dim=lat_dim,
                varargs=data[lat_dim].astype("float64") * np.pi / 180.0,
                method="ncl",
            )
        return calc_dlat_radian_gradient(data, lat_dim=lat_dim)

    dpsi_dlambda = dlon(psi_p)
    dpsi_dphi = dlat(psi_p)
    d2psi_dlambda2 = dlon(dpsi_dlambda)
    d2psi_dphi2 = dlat(dpsi_dphi)
    d2psi_dlambdadphi = (
        dlat(dpsi_dlambda) if finite_difference == "ncl" else dlon(dpsi_dphi)
    )

    term_xu = dpsi_dlambda**2 - (psi_p * d2psi_dlambda2)
    term_xv = dpsi_dlambda * dpsi_dphi - (psi_p * d2psi_dlambdadphi)
    term_yv = dpsi_dphi**2 - (psi_p * d2psi_dphi2)

    magU = np.sqrt(u_c**2 + v_c**2).where((u_c**2 + v_c**2) > 0)
    coeff = p / (2 * magU)

    fx = coeff * ((u_c / (R**2 * coslat) * term_xu) + (v_c / (R**2) * term_xv))
    fy = coeff * ((u_c / (R**2) * term_xv) + (v_c * coslat / (R**2) * term_yv))

    result = xr.Dataset()

    psi_p = psi_p.transpose(*dim_tuple)
    psi_p.attrs = {
        "long_name": "Perturbation stream function",
        "units": "m^2/s",
        "description": "Geostrophic stream function perturbation calculated from geopotential height anomaly",
    }
    result["psi_p"] = psi_p

    fx = fx.transpose(*dim_tuple)
    fx.attrs = {
        "long_name": "Zonal component of TN wave activity horizontal flux",
        "units": "m^2/s^2",
        "description": "Eastward component of Takaya and Nakamura wave activity flux",
        "standard_name": "tn_wave_activity_flux_x",
    }
    result["fx"] = fx

    fy = fy.transpose(*dim_tuple)
    fy.attrs = {
        "long_name": "Meridional component of TN wave activity horizontal flux",
        "units": "m^2/s^2",
        "description": "Northward component of Takaya and Nakamura wave activity flux",
        "standard_name": "tn_wave_activity_flux_y",
    }
    result["fy"] = fy

    result.attrs = {
        "title": "Takaya and Nakamura Wave Activity Flux",
        "description": "Horizontal components of quasi-geostrophic wave activity flux following Takaya and Nakamura (2001)",
        "reference": "Takaya, K. and H. Nakamura, 2001: A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow. J. Atmos. Sci., 58, 608-627.",
        "created_with": "easyclimate: calc_TN_wave_activity_horizontal_flux function",
    }
    return result.astype(output_dtype)


def calc_TN_wave_activity_vertical_flux(
    z_prime_data: xr.DataArray | None,
    u_climatology_data: xr.DataArray,
    v_climatology_data: xr.DataArray,
    temper_climatology_data: xr.DataArray,
    temper_climatology_data_units: Literal["degC", "degF", "degK"],
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    lon_dim: str = "lon",
    lat_dim: str = "lat",
    omega: float = 7.292e-5,
    g: float = 9.8,
    R: float = 6371200.0,
    scale_height: float = 8000.0,
    kappa: float = 0.286,
    p_0: float = 1000.0,
    psi_prime_data: xr.DataArray | None = None,
    time_dim: str = "time",
    gas_constant: float = 290.0,
    cyclic_lon: bool = False,
    min_westerly: float | None = 1.0,
    min_latitude: float = 10.0,
    output_dtype: Literal["float32", "float64"] = "float32",
) -> xr.Dataset:
    r"""Calculate the stationary TN01 vertical wave-activity flux in log-pressure height.

    This follows TN01 equation:

    .. math::
        F_z = \frac{(p/p_0) f^2}{2 |\mathbf{U}| N^2 R}
        \left[U(\psi'_\lambda\psi'_z-\psi'\psi'_{\lambda z})
        + V\cos\varphi(\psi'_\varphi\psi'_z-\psi'\psi'_{\varphi z})\right]

    Here :math:`z=-H\log(p/p_0)`. The squared buoyancy frequency is
    evaluated in log-pressure height:

    .. math::
        N^2 = \frac{R_d}{H}(p/p_0)^\kappa
        \frac{\partial[T_c(p_0/p)^\kappa]}{\partial z}

    Parameters
    ----------
    z_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>` or ``None``.
        Geopotential **height anomaly in metres**, relative to the basic state.
        Provide ``None`` when supplying ``psi_prime_data`` instead.

    .. attention:: The unit of `z_prime_data` should be **meters**, NOT :math:`\\mathrm{m^2 \\cdot s^2}` which is the unit used in the representation of potential energy.

    u_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Basic-state zonal geostrophic wind in m/s. It may be static, aligned
        with the anomaly's time coordinate, or supplied as a monthly climatology
        (using a datetime time coordinate or a month dimension). Monthly
        values are matched to the anomaly independently for each wind field.
    v_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Basic-state meridional geostrophic wind in m/s. Supports static,
        time-aligned, and monthly-climatology inputs as described above.
    temper_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Basic-state absolute temperature, defined on the same
        pressure grid. It will be used to calculate the squared buoyancy frequency
        :math:`N^2` in log-pressure height.
    temper_climatology_data_units: :py:class:`str <str>`.
        The unit corresponding to `temper_climatology_data` value. Optional values are ``degC``, ``degF``, ``degK`` and so on.
    vertical_dim: :py:class:`str <str>`.
        Name of the pressure-coordinate dimension. The coordinate must
        contain at least three positive, monotonically ordered pressure levels.
    vertical_dim_units: :py:class:`str <str>`.
        Units of the input pressure coordinate. Supported values are
        ``"hPa"``, ``"Pa"``, and ``"mbar"``.
    lon_dim: :py:class:`str <str>`, default: ``"lon"``.
        Longitude dimension name. Longitude coordinates are in degrees.
    lat_dim: :py:class:`str <str>`, default: ``"lat"``.
        Latitude dimension name. Latitude coordinates are in degrees.
    omega: :py:class:`float <float>`, default: ``7.292e-5``.
        Earth's angular rotation rate in rad/s.
    g: :py:class:`float <float>`, default: ``9.8``.
        Gravitational acceleration in :math:`\mathrm{m\,s^{-2}}`.
    R: :py:class:`float <float>`, default: ``6371200.0``.
        Earth's radius in metres.
    scale_height: :py:class:`float <float>`, default: ``8000.0``.
        Constant log-pressure scale height :math:`H` in metres, used in
        :math:`z=-H\log(p/p_0)`.
    kappa: :py:class:`float <float>`, default: ``0.286``.
        Potential-temperature exponent :math:`\\kappa`.

    .. note::
        `Poisson constant - Glossary of Meteorology <https://glossary.ametsoc.org/wiki/Poisson_constant>`__

    p_0: :py:class:`float <float>`, default: ``1000.0``.
        Reference pressure in **hPa**, even if ``vertical_dim_units="Pa"``.
        Also used to normalize pressure in the wave-activity flux.
    psi_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>` or ``None``, default: ``None``.
        Perturbation geostrophic streamfunction in :math:`\mathrm{m^2\,s^{-1}}`.
        If supplied, it takes precedence over ``z_prime_data``.
    time_dim: :py:class:`str <str>`, default: ``"time"``.
        Name of the time dimension used for the anomaly and time-dependent
        background fields.
    gas_constant: :py:class:`float <float>`, default: ``290.0`` (:math:`\mathrm{J\,kg^{-1}\,K^{-1}}`).
        Dry-air gas constant (:math:`R_d`). The log-pressure :math:`N^2`
        formulation differs from the geometric-height expression
        :math:`(g/\theta)\,\partial\theta/\partial z`.
    cyclic_lon: :py:class:`bool <bool>`, default: ``False``.
        If ``True``, use periodic centred longitude differences on a uniform
        global grid without a duplicate endpoint, following the NCL scheme.
        Leave ``False`` for noncyclic or regional grids.
    min_westerly: :py:class:`float <float>` or ``None``, default: ``1.0``.
        Minimum basic-state **zonal westerly wind** :math:`U` in m/s retained
        by the wave-activity flux mask. Flux is masked wherever
        :math:`U < \mathrm{min\_westerly}`, including weak westerlies and
        easterlies. This applies to :math:`U`, not the total horizontal wind
        speed. ``None`` disables this additional wind-threshold mask.

        .. note::

            Takaya and Nakamura (2001) derive the wave-activity flux for
            small-amplitude quasigeostrophic disturbances on a slowly varying
            basic flow. The stationary-wave propagation interpretation may
            become unreliable in weak-westerly regions, especially near
            critical layers or where nonlinear effects are important.

            This practical cutoff is **not** a universal critical wind speed
            prescribed by TN01. Exceeding it does not guarantee that the
            assumptions underlying the flux diagnostic are satisfied.

            The default 1 m/s follows ``ubmin=1.`` in
            `Nishii's multi-level Fortran program
            <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/tnf_xyz_multilevel.f90>`_.
            Set ``min_westerly=5.0`` to reproduce the wind mask in the
            `monthly NCL program
            <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/TakayaNakamura2001-ncep-monthly.ncl>`_.
            These are implementation-specific choices, not theoretical
            TN01 thresholds. Changing the parameter changes the wind mask,
            not the log-pressure wave-activity flux formulas.
    min_latitude: :py:class:`float <float>`, default: ``10.0``.
        Mask latitudes where the absolute latitude is below this threshold
        in degrees (10 degrees by default). The equator and poles
        are always masked.
    output_dtype: :py:class:`str <str>`, default: ``"float32"``.
        Output precision: ``"float32"`` or ``"float64"``. Derivatives and
        flux expressions are calculated in float64 before the final cast.

    Returns
    -------
    :py:class:`xarray.Dataset<xarray.Dataset>`.
        A dataset containing:

        - ``fz``: vertical wave-activity flux in
          :math:`\mathrm{m^2\,s^{-2}}` (positive upward).
        - ``psi_p``: perturbation geostrophic streamfunction in
          :math:`\mathrm{m^2\,s^{-1}}`.
        - ``N2``: squared buoyancy frequency in :math:`\mathrm{s^{-2}}`.

        Differences use neighbouring secants, including for nonuniform
        pressure levels. Noncyclic boundaries use first-order one-sided
        differences. Nonpositive stability, zero background-wind speed,
        and invalid quasigeostrophic latitudes are masked. Original
        dimensions, coordinates, and their order are retained; Dask inputs
        remain lazy.

    References
    ----------
    - Takaya, K., & Nakamura, H. (2001). A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow. Journal of the Atmospheric Sciences, 58(6), 608-627. https://journals.ametsoc.org/view/journals/atsc/58/6/1520-0469_2001_058_0608_afoapi_2.0.co_2.xml

    .. seealso::
        - http://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/index.html
    """
    sample = psi_prime_data if psi_prime_data is not None else z_prime_data
    if not isinstance(sample, xr.DataArray):
        raise TypeError(
            "Provide z_prime_data or psi_prime_data as an xarray.DataArray."
        )
    if output_dtype not in ("float32", "float64"):
        raise ValueError("output_dtype must be 'float32' or 'float64'.")
    sample = sample.astype("float64")
    if not all(
        np.isfinite(value) and value > 0
        for value in (scale_height, p_0, gas_constant, kappa, R, g, omega)
    ):
        raise ValueError("Physical constants must be finite and positive.")
    if not 0 <= min_latitude < 90:
        raise ValueError("min_latitude must be in [0, 90).")
    for dim in (vertical_dim, lat_dim, lon_dim):
        if dim not in sample.dims or sample[dim].dims != (dim,):
            raise ValueError(f"A one-dimensional {dim!r} coordinate is required.")
    p_hpa = transfer_data_multiple_units(
        sample[vertical_dim].astype("float64"), vertical_dim_units, "hPa"
    )
    if bool((p_hpa <= 0).any()):
        raise ValueError("Pressure levels must be positive.")
    p = p_hpa / p_0
    log_height = -scale_height * np.log(p)

    temper_climatology_data = transfer_data_temperature_units(
        temper_climatology_data, temper_climatology_data_units, "K"
    )  # units: K
    u_c, v_c, t_c = (
        _tn_climatology_for_sample(data, sample, time_dim).astype("float64")
        for data in (u_climatology_data, v_climatology_data, temper_climatology_data)
    )
    for data in (u_c, v_c, t_c):
        if any(dim not in sample.dims for dim in data.dims):
            raise ValueError(
                "Background dimensions must be a subset of anomaly dimensions."
            )
    if vertical_dim not in t_c.dims:
        raise ValueError("Basic-state temperature must include the pressure dimension.")
    xr.align(sample, u_c, v_c, t_c, join="exact", copy=False)

    lat = sample[lat_dim].astype("float64")
    coslat = np.cos(lat * np.pi / 180.0)
    valid_lat = (abs(lat) >= min_latitude) & (abs(lat) > 0) & (abs(lat) < 90)
    f = get_coriolis_parameter(lat, omega=omega).where(valid_lat)
    psi_p = sample.astype("float64") if psi_prime_data is not None else sample * g / f
    psi_p = psi_p.where(valid_lat)
    if cyclic_lon:
        _tn_validate_cyclic_longitude(sample[lon_dim].astype("float64") * np.pi / 180.0)
    dlambda = calc_gradient(
        psi_p,
        dim=lon_dim,
        varargs=sample[lon_dim].astype("float64") * np.pi / 180.0,
        method="ncl",
        cyclic=cyclic_lon,
    )
    dphi = calc_gradient(psi_p, dim=lat_dim, varargs=lat * np.pi / 180.0, method="ncl")
    dz = calc_gradient(psi_p, dim=vertical_dim, varargs=log_height, method="ncl")
    dlambda_z = calc_gradient(
        dlambda, dim=vertical_dim, varargs=log_height, method="ncl"
    )
    dphi_z = calc_gradient(dphi, dim=vertical_dim, varargs=log_height, method="ncl")
    theta = t_c * (p_0 / p_hpa) ** kappa
    N2 = (
        (gas_constant * p**kappa)
        / scale_height
        * calc_gradient(theta, dim=vertical_dim, varargs=log_height, method="ncl")
    )
    N2 = N2.where(N2 > 0)
    speed = np.sqrt(u_c**2 + v_c**2)
    speed = speed.where(speed > 0)
    fz = (
        p
        * f**2
        / (2 * speed * N2 * R)
        * (
            u_c * (dlambda * dz - psi_p * dlambda_z)
            + v_c * coslat * (dphi * dz - psi_p * dphi_z)
        )
    )
    if min_westerly is not None:
        fz = fz.where(u_c >= min_westerly)

    result = xr.Dataset(
        {
            "psi_p": psi_p.transpose(*sample.dims),
            "fz": fz.transpose(*sample.dims),
            "N2": N2.broadcast_like(sample).transpose(*sample.dims),
        }
    )
    result.psi_p.attrs = {"long_name": "Perturbation stream function", "units": "m^2/s"}
    result.fz.attrs = {
        "long_name": "Upward component of TN wave activity flux in log-pressure height",
        "units": "m^2/s^2",
        "positive": "up",
    }
    result.N2.attrs = {
        "long_name": "Squared buoyancy frequency in log-pressure height",
        "units": "s^-2",
    }
    result.attrs = {
        "title": "Takaya and Nakamura vertical wave activity flux",
        "reference": "Takaya and Nakamura (2001), equation (38); Nishii monthly NCL implementation",
        "vertical_coordinate": "z = -scale_height * log(p / p_0)",
        "scale_height": scale_height,
        "reference_pressure_hPa": p_0,
    }
    return result.astype(output_dtype)


def calc_TN_wave_activity_3D_flux(
    z_prime_data: xr.DataArray | None,
    u_climatology_data: xr.DataArray,
    v_climatology_data: xr.DataArray,
    temper_climatology_data: xr.DataArray,
    temper_climatology_data_units: Literal["degC", "degF", "degK"],
    vertical_dim: str,
    vertical_dim_units: Literal["hPa", "Pa", "mbar"],
    lon_dim: str = "lon",
    lat_dim: str = "lat",
    omega: float = 7.292e-5,
    g: float = 9.8,
    R: float = 6371200.0,
    scale_height: float = 8000.0,
    kappa: float = 0.286,
    p_0: float = 1000.0,
    psi_prime_data: xr.DataArray | None = None,
    time_dim: str = "time",
    gas_constant: float = 290.0,
    cyclic_lon: bool = False,
    min_westerly: float | None = 1.0,
    min_latitude: float = 10.0,
    output_dtype: Literal["float32", "float64"] = "float32",
) -> xr.Dataset:
    r"""Calculate all three stationary TN01 wave-activity flux components.

    Return the zonal, meridional, and vertical flux components using the
    conventions of :func:`calc_TN_wave_activity_vertical_flux`. The horizontal
    components use NCL centred secants; noncyclic boundaries use first-order
    one-sided differences, unlike the legacy horizontal interface. The
    stability condition :math:`N^2 > 0` affects only the vertical flux
    (``fz``); the horizontal components (``fx`` and ``fy``) do not depend on
    :math:`N^2`.

    Parameters
    ----------
    z_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>` or ``None``.
        Geopotential **height anomaly in metres**, relative to the basic state.
        Provide ``None`` when supplying ``psi_prime_data`` instead.

    .. attention:: The unit of `z_prime_data` should be **meters**, NOT :math:`\\mathrm{m^2 \\cdot s^2}` which is the unit used in the representation of potential energy.

    u_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Basic-state zonal geostrophic wind in m/s. It may be static, aligned
        with the anomaly's time coordinate, or supplied as a monthly climatology
        (using a datetime time coordinate or a month dimension). Monthly
        values are matched to the anomaly independently for each wind field.
    v_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Basic-state meridional geostrophic wind in m/s. Supports static,
        time-aligned, and monthly-climatology inputs as described above.
    temper_climatology_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        Basic-state absolute temperature, defined on the same
        pressure grid. It will be used to calculate the squared buoyancy frequency
        :math:`N^2` in log-pressure height.
    temper_climatology_data_units: :py:class:`str <str>`.
        The unit corresponding to `temper_climatology_data` value. Optional values are ``degC``, ``degF``, ``degK`` and so on.
    vertical_dim: :py:class:`str <str>`.
        Name of the pressure-coordinate dimension. The coordinate must
        contain at least three positive, monotonically ordered pressure levels.
    vertical_dim_units: :py:class:`str <str>`.
        Units of the input pressure coordinate. Supported values are
        ``"hPa"``, ``"Pa"``, and ``"mbar"``.
    lon_dim: :py:class:`str <str>`, default: ``"lon"``.
        Longitude dimension name. Longitude coordinates are in degrees.
    lat_dim: :py:class:`str <str>`, default: ``"lat"``.
        Latitude dimension name. Latitude coordinates are in degrees.
    omega: :py:class:`float <float>`, default: ``7.292e-5``.
        Earth's angular rotation rate in rad/s.
    g: :py:class:`float <float>`, default: ``9.8``.
        Gravitational acceleration in :math:`\mathrm{m\,s^{-2}}`.
    R: :py:class:`float <float>`, default: ``6371200.0``.
        Earth's radius in metres.
    scale_height: :py:class:`float <float>`, default: ``8000.0``.
        Constant log-pressure scale height :math:`H` in metres, used in
        :math:`z=-H\log(p/p_0)`.
    kappa: :py:class:`float <float>`, default: ``0.286``.
        Potential-temperature exponent :math:`\\kappa`.

    .. note::
        `Poisson constant - Glossary of Meteorology <https://glossary.ametsoc.org/wiki/Poisson_constant>`__

    p_0: :py:class:`float <float>`, default: ``1000.0``.
        Reference pressure in **hPa**, even if ``vertical_dim_units="Pa"``.
        Also used to normalize pressure in the wave-activity flux.
    psi_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>` or ``None``, default: ``None``.
        Perturbation geostrophic streamfunction in :math:`\mathrm{m^2\,s^{-1}}`.
        If supplied, it takes precedence over ``z_prime_data``.
    time_dim: :py:class:`str <str>`, default: ``"time"``.
        Name of the time dimension used for the anomaly and time-dependent
        background fields.
    gas_constant: :py:class:`float <float>`, default: ``290.0`` (:math:`\mathrm{J\,kg^{-1}\,K^{-1}}`).
        Dry-air gas constant (:math:`R_d`). The log-pressure :math:`N^2`
        formulation differs from the geometric-height expression
        :math:`(g/\theta)\,\partial\theta/\partial z`.
    cyclic_lon: :py:class:`bool <bool>`, default: ``False``.
        If ``True``, use periodic centred longitude differences on a uniform
        global grid without a duplicate endpoint, following the NCL scheme.
        Leave ``False`` for noncyclic or regional grids.
    min_westerly: :py:class:`float <float>` or ``None``, default: ``1.0``.
        Minimum basic-state **zonal westerly wind** :math:`U` in m/s retained
        by the wave-activity flux mask. Flux is masked wherever
        :math:`U < \mathrm{min\_westerly}`, including weak westerlies and
        easterlies. This applies to :math:`U`, not the total horizontal wind
        speed. ``None`` disables this additional wind-threshold mask.

        .. note::

            Takaya and Nakamura (2001) derive the wave-activity flux for
            small-amplitude quasigeostrophic disturbances on a slowly varying
            basic flow. The stationary-wave propagation interpretation may
            become unreliable in weak-westerly regions, especially near
            critical layers or where nonlinear effects are important.

            This practical cutoff is **not** a universal critical wind speed
            prescribed by TN01. Exceeding it does not guarantee that the
            assumptions underlying the flux diagnostic are satisfied.

            The default 1 m/s follows ``ubmin=1.`` in
            `Nishii's multi-level Fortran program
            <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/tnf_xyz_multilevel.f90>`_.
            Set ``min_westerly=5.0`` to reproduce the wind mask in the
            `monthly NCL program
            <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/TakayaNakamura2001-ncep-monthly.ncl>`_.
            These are implementation-specific choices, not theoretical
            TN01 thresholds. Changing the parameter changes the wind mask,
            not the log-pressure wave-activity flux formulas.
    min_latitude: :py:class:`float <float>`, default: ``10.0``.
        Mask latitudes where the absolute latitude is below this threshold
        in degrees (10 degrees by default). The equator and poles
        are always masked.
    output_dtype: :py:class:`str <str>`, default: ``"float32"``.
        Output precision: ``"float32"`` or ``"float64"``. Derivatives and
        flux expressions are calculated in float64 before the final cast.

    Returns
    -------
    :py:class:`xarray.Dataset<xarray.Dataset>`.
        A dataset containing:

        - ``fx``: zonal (eastward) wave-activity flux in
          :math:`\mathrm{m^2\,s^{-2}}`.
        - ``fy``: meridional (northward) wave-activity flux in
          :math:`\mathrm{m^2\,s^{-2}}`.
        - ``fz``: vertical wave-activity flux in
          :math:`\mathrm{m^2\,s^{-2}}` (positive upward).
        - ``psi_p``: perturbation geostrophic streamfunction in
          :math:`\mathrm{m^2\,s^{-1}}`.
        - ``N2``: squared buoyancy frequency in :math:`\mathrm{s^{-2}}`.

        Both horizontal and vertical flux components are subject to the
        background-zonal-wind and latitude masks. Nonpositive stability
        additionally masks ``fz`` only. Longitude and pressure differences
        use centred secants in the interior and first-order noncyclic
        endpoints. Original dimensions, coordinates, and their order are
        retained; Dask inputs remain lazy.

    References
    ----------
    - Takaya, K., & Nakamura, H. (2001). A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow. Journal of the Atmospheric Sciences, 58(6), 608-627. https://journals.ametsoc.org/view/journals/atsc/58/6/1520-0469_2001_058_0608_afoapi_2.0.co_2.xml

    .. seealso::
        - http://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/index.html
        - https://github.com/laishenggx/T-N_Wave-Activity-Flux
    """
    result = calc_TN_wave_activity_vertical_flux(
        z_prime_data,
        u_climatology_data,
        v_climatology_data,
        temper_climatology_data,
        temper_climatology_data_units,
        vertical_dim,
        vertical_dim_units,
        lon_dim=lon_dim,
        lat_dim=lat_dim,
        omega=omega,
        g=g,
        R=R,
        scale_height=scale_height,
        kappa=kappa,
        p_0=p_0,
        psi_prime_data=psi_prime_data,
        time_dim=time_dim,
        gas_constant=gas_constant,
        cyclic_lon=cyclic_lon,
        min_westerly=min_westerly,
        min_latitude=min_latitude,
        # Keep the streamfunction precise for the horizontal derivatives below.
        output_dtype="float64",
    )
    horizontal = calc_TN_wave_activity_horizontal_flux(
        None,
        u_climatology_data,
        v_climatology_data,
        vertical_dim,
        vertical_dim_units,
        psi_prime_data=result.psi_p,
        lon_dim=lon_dim,
        lat_dim=lat_dim,
        time_dim=time_dim,
        omega=omega,
        g=g,
        R=R,
        cyclic_lon=cyclic_lon,
        finite_difference="ncl",
        output_dtype="float64",
    )
    u_c = _tn_climatology_for_sample(u_climatology_data, result.psi_p, time_dim)
    for name in ("fx", "fy"):
        # The existing horizontal routine always normalizes by 1000 hPa.
        result[name] = horizontal[name] * (1000.0 / p_0)
        if min_westerly is not None:
            result[name] = result[name].where(u_c >= min_westerly)
        result[name].attrs = horizontal[name].attrs
    result.attrs["title"] = "Takaya and Nakamura three-dimensional wave activity flux"
    if output_dtype not in ("float32", "float64"):
        raise ValueError("output_dtype must be 'float32' or 'float64'.")
    return result.astype(output_dtype)


def calc_EP_horizontal_flux(
    u_prime_data: xr.DataArray,
    v_prime_data: xr.DataArray,
    time_dim: str = "time",
    lat_dim: str = "lat",
) -> xr.Dataset:
    """
    Calculate horizontal Eliassen–Palm Flux.

    Parameters
    ----------
    u_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The anormaly of zonal wind data.
    v_prime_data: :py:class:`xarray.DataArray<xarray.DataArray>`.
        The anormaly of meridional wind data.
    time_dim: :py:class:`str <str>`, default: `time`.
        The time coordinate dimension name.
    lat_dim: :py:class:`str <str>`, default: `lat`.
        Latitude coordinate dimension name. By default extracting is applied over the `lat` dimension.

    Returns
    -------
    The Eliassen–Palm Flux (:py:class:`xarray.DataArray<xarray.DataArray>`).

    .. seealso::
        - https://www.ncl.ucar.edu/Applications/EPflux.shtml
        - https://renqlsysu.github.io/2020/02/16/wave_activity_flux/
    """
    lat_array = u_prime_data[lat_dim].astype("float64")
    coslat = np.cos(transfer_deg2rad(lat_array))

    fx = 0.5 * (
        (v_prime_data**2).mean(dim=time_dim) - (u_prime_data**2).mean(dim=time_dim)
    )
    fy = -(u_prime_data * v_prime_data).mean(dim=time_dim)

    result = xr.Dataset()
    result["fx"] = fx * coslat
    result["fy"] = fy * coslat
    return result


def calc_monthly_rossby_wave_source(
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    u_climatology_data: xr.DataArray,
    v_climatology_data: xr.DataArray,
    lat_dim: str = "lat",
    omega: float = 7.292e-05,
    R: float = 6371200.0,
) -> xr.DataArray:
    """
    Calculate the Rossby wave source following Sardeshmukh and Hoskins (1988).

    The Rossby wave source (RWS) represents the forcing term for stationary Rossby waves
    and is widely used to diagnose atmospheric teleconnections. It is decomposed into
    five terms representing different physical mechanisms:

    .. math::

        S' = -\\nabla \\cdot (\\mathbf{v}_\\chi \\zeta)' = \\text{term1a} + \\text{term1b} + \\text{term2a} + \\text{term2b} + \\text{term3} + \\text{term4} + \\text{term5}

    where:

    - **term1a**: :math:`-\\bar{\\zeta} \\nabla \\cdot \\mathbf{v}'_\\chi`
      (stretching of climatological vorticity by divergent wind anomalies)
    - **Term1b**: :math:`-f \\nabla \\cdot \\mathbf{v'_\\chi}`
      (Stretching of planetary vorticity by anomalous divergence)
    - **term2a**: :math:`-\\mathbf{v}'_\\chi \\cdot \\nabla\\bar{\\zeta}`
      (advection of climatological vorticity by divergent wind anomalies)
    - **term2b**: :math:`-\\beta v'_\\chi`
      (planetary vorticity advection, where :math:`\\beta = \\partial f/\\partial y`)
    - **term3**: :math:`-\\zeta' \\nabla \\cdot \\bar{\\mathbf{v}}_\\chi`
      (stretching of vorticity anomalies by climatological divergent wind)
    - **term4**: :math:`-\\bar{\\mathbf{v}}_\\chi \\cdot \\nabla\\zeta'`
      (advection of vorticity anomalies by climatological divergent wind)
    - **term5**: :math:`-\\nabla \\cdot (\\mathbf{v}'_\\chi \\zeta')`
      (nonlinear term: divergence of transient vorticity flux)

    where :math:`\\mathbf{v}_\\chi = (u_\\chi, v_\\chi)` is the divergent (irrotational)
    wind component, :math:`\\zeta` is relative vorticity, overbar denotes climatology,
    and prime denotes anomaly.

    .. note::
        - Anomalies are computed as deviations from the monthly climatology.
        - The planetary term (term6) represents the beta effect, which is important for Rossby wave propagation, especially in the tropics and subtropics.
        - Terms 1, 2, and 6 typically dominate the Rossby wave source.
        - Positive RWS indicates a cyclonic wave source; negative indicates anticyclonic.

    Parameters
    ----------
    u_data : :py:class:`xarray.DataArray<xarray.DataArray>` (:math:`\\mathrm{m/s}`)
        Zonal wind component. Must contain a 'time' dimension with monthly or
        higher frequency data. Expected dimensions: ``(time, lat, lon)`` or ``(time, level, lat, lon)``.
    v_data : :py:class:`xarray.DataArray<xarray.DataArray>` (:math:`\\mathrm{m/s}`)
        Meridional wind component. Must have the same dimensions as `u_data`.
    u_climatology_data : :py:class:`xarray.DataArray<xarray.DataArray>` (:math:`\\mathrm{m/s}`)
        Monthly climatology of zonal wind. Expected dimensions: ``(time, lat, lon)``
        or ``(time, level, lat, lon)``, where time/month ranges from 1 to 12.
    v_climatology_data : :py:class:`xarray.DataArray<xarray.DataArray>` (:math:`\\mathrm{m/s}`)
        Monthly climatology of meridional wind. Must have the same dimensions
        as `u_climatology_data`.

    Returns
    -------
    result : :py:class:`xarray.Dataset<xarray.Dataset>`
        Dataset containing the Rossby wave source and its five decomposed terms:

        - **RWS**: Total Rossby wave source (sum of all terms)
        - **term1a**: Vorticity stretching by anomalous divergence
        - **term1b**: Stretching of planetary vorticity by anomalous divergence
        - **term2a**: Vorticity advection by anomalous divergent wind
        - **term2b**: Planetary vorticity advection (beta effect)
        - **term3**: Anomalous vorticity stretching by climatological divergence
        - **term4**: Anomalous vorticity advection by climatological divergent wind
        - **term5**: Nonlinear transient eddy term


        All variables have units of :math:`\\mathrm{s^{-2}}` and retain the spatial/temporal dimensions
        of the input data.

    .. seealso::

        - Sardeshmukh, P. D., & Hoskins, B. J. (1988). The generation of global rotational flow by steady idealized tropical divergence. Journal of the Atmospheric Sciences, 45(7), 1228-1251. https://journals.ametsoc.org/view/journals/atsc/45/7/1520-0469_1988_045_1228_tgogrf_2_0_co_2.xml, https://doi.org/10.1175/1520-0469(1988)045<1228:TGOGRF>2.0.CO;2

    Examples
    --------
    >>> # Calculate monthly climatology
    >>> u_clim = u.groupby('time.month').mean('time')
    >>> v_clim = v.groupby('time.month').mean('time')
    >>>
    >>> # Compute Rossby wave source
    >>> rws_result = calc_rossby_wave_source(u, v, u_clim, v_clim)
    >>>
    >>> # Visualize the dominant terms
    >>> rws_result['RWS'].sel(time='2015-12').plot()
    >>> (rws_result['term1'] + rws_result['term2']).sel(time='2015-12').plot()
    """
    from .spec import (
        calc_relative_vorticity,
        calc_irrotational_component,
        calc_gradient,
    )
    from ..physics.geo import get_coriolis_parameter
    from .utility import transfer_deg2rad

    # planetary vorticity
    lat = v_data[lat_dim]

    # Real Value
    zeta = calc_relative_vorticity(u_data, v_data)
    ir_result = calc_irrotational_component(u_data, v_data)
    uchi = ir_result.uchi
    vchi = ir_result.vchi

    # Climate
    zeta_climate = calc_relative_vorticity(u_climatology_data, v_climatology_data)
    ir_climate_result = calc_irrotational_component(
        u_climatology_data, v_climatology_data
    )
    uchi_climate = ir_climate_result.uchi
    vchi_climate = ir_climate_result.vchi

    # Prime
    zeta_prime = zeta.groupby("time.month") - zeta_climate.groupby("time.month").mean()
    uchi_prime = uchi.groupby("time.month") - uchi_climate.groupby("time.month").mean()
    vchi_prime = vchi.groupby("time.month") - vchi_climate.groupby("time.month").mean()

    # Term1a: \bar{\zeta} \nabla \cdot \bm{v'_\xi}
    uchi_prime_grdx = calc_gradient(uchi_prime)["zonal_gradient"]
    vchi_prime_grdy = calc_gradient(vchi_prime)["meridional_gradient"]
    term1a = -(
        zeta_climate.groupby("time.month").mean()
        * ((uchi_prime_grdx + vchi_prime_grdy).groupby("time.month"))
    )

    # Term1b: -f \nabla \cdot \bm{v'_\chi}
    f = get_coriolis_parameter(lat, omega=omega)
    term1b = -f * (uchi_prime_grdx + vchi_prime_grdy)

    # Term2a: \bm{v'_\xi} \cdot \nabla\bar{\zeta}
    zeta_climate_result = calc_gradient(zeta_climate)
    zeta_climate_grdx = zeta_climate_result["zonal_gradient"]
    zeta_climate_grdy = zeta_climate_result["meridional_gradient"]
    term2a = -(
        uchi_prime.groupby("time.month")
        * zeta_climate_grdx.groupby("time.month").mean()
        + vchi_prime.groupby("time.month")
        * zeta_climate_grdy.groupby("time.month").mean()
    )

    # Term2b: -\beta v'_\chi
    # Calculate beta = df/dy
    # beta = (2 * Omega * cos(phi)) / a
    lat_rad = transfer_deg2rad(lat)
    beta = (2 * omega * np.cos(lat_rad)) / R
    # beta * v'_chi
    term2b = -(beta * vchi_prime)

    # Term3: \zeta' \nabla \cdot \bar{v'_\xi}
    uchi_climate_grdx = calc_gradient(uchi_climate)["zonal_gradient"]
    vchi_climate_grdy = calc_gradient(vchi_climate)["meridional_gradient"]
    term3 = -(
        zeta_prime.groupby("time.month")
        * ((uchi_climate_grdx + vchi_climate_grdy).groupby("time.month").mean())
    )

    # Term4: \bm{\bar{v}_\xi} \cdot \nabla \zeta'
    zeta_prime_result = calc_gradient(zeta_prime)
    zeta_prime_grdx = zeta_prime_result["zonal_gradient"]
    zeta_prime_grdy = zeta_prime_result["meridional_gradient"]
    term4 = -(
        uchi_climate.groupby("time.month").mean()
        * zeta_prime_grdx.groupby("time.month")
        + vchi_climate.groupby("time.month").mean()
        * zeta_prime_grdy.groupby("time.month")
    )

    # Term5: \nabla \cdot (\bm{v}'_\chi \zeta')
    # u'_\chi * \zeta'
    uchi_prime_zeta_prime = uchi_prime * zeta_prime
    # v'_\chi * \zeta'
    vchi_prime_zeta_prime = vchi_prime * zeta_prime
    uchi_prime_zeta_prime_grdx = calc_gradient(uchi_prime_zeta_prime)["zonal_gradient"]
    vchi_prime_zeta_prime_grdy = calc_gradient(vchi_prime_zeta_prime)[
        "meridional_gradient"
    ]
    term5 = -(uchi_prime_zeta_prime_grdx + vchi_prime_zeta_prime_grdy)

    # Compute total RWS
    rws = term1a + term1b + term2a + term2b + term3 + term4 + term5

    # Obtain the dimension order
    dim_array = u_data.dims

    # Create Data Variables
    result = xr.Dataset()

    # Add the data variables for each item
    result["term1a"] = term1a.transpose(*dim_array)
    result["term1b"] = term1b.transpose(*dim_array)
    result["term2a"] = term2a.transpose(*dim_array)
    result["term2b"] = term2b.transpose(*dim_array)
    result["term3"] = term3.transpose(*dim_array)
    result["term4"] = term4.transpose(*dim_array)
    result["term5"] = term5.transpose(*dim_array)
    result["RWS"] = rws.transpose(*dim_array)

    # Set attributes for each variable
    result["term1a"].attrs = {
        "long_name": "RWS Term 1a: Vorticity stretching",
        "description": "Stretching of climatological vorticity by anomalous divergent wind",
        "formula": "-zeta_bar * div(v_chi_prime)",
        "units": "s-2",
        "standard_name": "rossby_wave_source_term1",
    }

    result["term1b"].attrs = {
        "long_name": "RWS Term 1b: Planetary vorticity stretching",
        "formula": "-f * div(v_chi_prime)",
        "units": "s-2",
        "description": "Stretching of planetary vorticity by anomalous divergence",
    }

    result["term2a"].attrs = {
        "long_name": "RWS Term 2a: Vorticity advection",
        "description": "Advection of climatological vorticity gradient by anomalous divergent wind",
        "formula": "-v_chi_prime · grad(zeta_bar)",
        "units": "s-2",
        "standard_name": "rossby_wave_source_term2",
    }

    result["term2b"].attrs = {
        "long_name": "RWS Term 2b: Planetary vorticity advection",
        "description": "Planetary vorticity advection (beta effect): meridional advection of planetary vorticity by anomalous divergent wind",
        "formula": "-beta * v_chi_prime, where beta = df/dy",
        "units": "s-2",
        "standard_name": "rossby_wave_source_planetary",
        "note": "Important for Rossby wave propagation, especially in tropics and subtropics",
    }

    result["term3"].attrs = {
        "long_name": "RWS Term 3: Anomalous vorticity stretching",
        "description": "Stretching of vorticity anomalies by climatological divergent wind",
        "formula": "-zeta_prime * div(v_chi_bar)",
        "units": "s-2",
        "standard_name": "rossby_wave_source_term3",
    }

    result["term4"].attrs = {
        "long_name": "RWS Term 4: Anomalous vorticity advection",
        "description": "Advection of vorticity anomalies by climatological divergent wind",
        "formula": "-v_chi_bar · grad(zeta_prime)",
        "units": "s-2",
        "standard_name": "rossby_wave_source_term4",
    }

    result["term5"].attrs = {
        "long_name": "RWS Term 5: Nonlinear transient term",
        "description": "Divergence of transient vorticity flux (nonlinear eddy term)",
        "formula": "-div(v_chi_prime * zeta_prime)",
        "units": "s-2",
        "standard_name": "rossby_wave_source_term5",
    }

    result["RWS"].attrs = {
        "long_name": "Total Rossby Wave Source",
        "description": "Total Rossby wave source including both relative and planetary vorticity effects",
        "formula": "-div(v_chi * eta)' where eta = zeta + f",
        "units": "s-2",
        "standard_name": "rossby_wave_source_total",
        "references": "Sardeshmukh and Hoskins (1988, JAS)",
        "positive": "cyclonic wave source",
        "negative": "anticyclonic wave source",
    }

    # Add global attributes
    result.attrs.update(
        {
            "title": "Rossby Wave Source Decomposition with Planetary Term",
            "method": "Six-term decomposition (5 relative + 1 planetary) following Sardeshmukh and Hoskins (1988)",
            "anomaly_method": "Deviation from monthly climatology",
            "convention": "Negative divergence convention (sources are positive for cyclonic forcing)",
            "references": "Sardeshmukh, P. D., & Hoskins, B. J. (1988). JAS, 45(7), 1228-1251.",
            "note": "Terms 1, 2, and 6 (planetary) typically dominate the Rossby wave source",
            "earth_radius": f"{R} m",
            "earth_rotation_rate": f"{omega} rad/s",
            "created_with": "easyclimate: calc_rossby_wave_source function",
        }
    )

    return result
