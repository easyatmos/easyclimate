# -*- coding: utf-8 -*-
r"""
.. _tn_waf_3d_example:

TN01 Three-Dimensional Wave Activity Flux
============================================

:func:`easyclimate.calc_TN_wave_activity_3D_flux` calculates the zonal,
meridional and upward components of stationary TN01 wave activity flux
together. A pressure-level map combines horizontal vectors with vertical-flux
shading; a longitude--height section then shows zonal and vertical propagation.

The ERA5 sample data contains February 2025 height anomalies
and the February 1991--2020 basic-state winds and temperature. Height anomalies
are already in **metres** and temperature is in **kelvin**.

Only the four required fields and a buffered domain are stored. The sample
retains native grid spacing: 25 pressure levels, 67 Gaussian latitudes and
320 global longitudes. Derivatives are calculated before any interpolation
for plotting. The supplied ERA5 winds approximate the geostrophic basic
state assumed by TN01.

.. seealso::

    - :ref:`tn_waf_vertical_example`
    - :ref:`max_waf_example`
    - `Nishii's reference programs <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/index.html>`_
    - `Nishii and Nakamura (2004), Figures 2 and 5 <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/paper/QJRMS-NishiiNakamura04.pdf>`_
"""
import cartopy.crs as ccrs
import easyclimate as ecl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr

# %%
# Load the local data and calculate every component
# ------------------------------------------------------
# Keep all neighbouring levels and horizontal points until the calculation
# has finished. The longitudes are global, uniform, and have no repeated
# endpoint, so cyclic differences are enabled. Regional grids use
# ``cyclic_lon=False``.
# The default 1 m/s wind cutoff follows the Fortran reference; set
# ``min_westerly=5.0`` for the monthly NCL mask, or None to disable it.
sample = ecl.open_tutorial_dataset("tn_waf_era5_202502_sample")
sample

# %%
# Configure scientific notation for the contour-plot colorbars.
formatter = ticker.ScalarFormatter(useMathText=True, useOffset=True)
formatter.set_powerlimits((0, 0))

# %%
flux_3d = ecl.calc_TN_wave_activity_3D_flux(
    z_prime_data=sample.z_prime,
    u_climatology_data=sample.u_basic,
    v_climatology_data=sample.v_basic,
    temper_climatology_data=sample.t_basic,
    temper_climatology_data_units = "degK",
    vertical_dim="level",
    vertical_dim_units="hPa",
    cyclic_lon=True,
    min_westerly=1.0,
    min_latitude=10.0,
    output_dtype="float64",
)
flux_3d

# %%
# ``fx``, ``fy`` and ``fz`` are all in m²/s². ``fz`` is positive upward in
# log-pressure height, not in pressure coordinates. ``psi_p`` and ``N2`` are
# also returned. All intermediate operations use float64; set
# ``output_dtype="float32"`` if single-precision output is sufficient.
#
# The three-dimensional function uses the same vertical expression as
# :func:`easyclimate.calc_TN_wave_activity_vertical_flux`. Its horizontal
# differences correspond to the horizontal function with
# ``finite_difference="ncl"`` and the same cyclic-longitude choice. Default
# weak/easterly-wind and tropical masks apply to the three flux components;
# nonpositive stability only masks ``fz``.

# %%
# Horizontal and vertical propagation at 250 hPa
# ------------------------------------------------------
# Literature examples combine horizontal-flux arrows with perturbation
# contours and upward-flux shading (e.g. Nishii and Nakamura, 2004, Figure 2).
# Here all fields are shown at 250 hPa: shading is fz, contours are psi_p in
# millions of m²/s, and arrows are (fx, fy).
#
# Scalar shading stays on the native grid. For evenly placed arrows, interpolate
# the already calculated horizontal flux onto a regular display grid. Do not
# interpolate or thin the inputs before differentiating. White cells remain
# where the flux is undefined or masked; they are not replaced by zeros.
pressure_level = flux_3d.isel(time=0, drop=True).sel(level=250)
shading = ecl.plot.add_lon_cyclic(
    pressure_level.fz, inter=float(sample.lon.diff("lon").isel(lon=0))
)
psi_contours = ecl.plot.add_lon_cyclic(
    pressure_level.psi_p / 1e6, inter=float(sample.lon.diff("lon").isel(lon=0))
)
vectors = pressure_level[["fx", "fy"]].interp(
    lon=np.arange(0, 360, 10), lat=np.arange(20, 81, 5)
)

# %%
fig, ax = plt.subplots(
    figsize=(9, 4),
    subplot_kw={"projection": ccrs.Mercator(central_longitude=180)},
    layout="constrained",
)
ecl.plot.set_geoextent([0, 360, 20, 80], ax=ax)
ax.coastlines(linewidth=0.6)
ax.gridlines(draw_labels=["bottom", "left"], alpha=0.3)
shading.plot.contourf(
    ax=ax,
    levels = np.linspace(-0.05, 0.05, 21),
    cmap="RdBu_r",
    extend="both",
    transform=ccrs.PlateCarree(),
    cbar_kwargs={
        "label": r"Upward WAF $F_z$ (m$^2$ s$^{-2}$)",
        "orientation": "horizontal",
        "pad": 0.09,
        "aspect": 60,
        "shrink": 0.8,
    },
)
psi_contours.plot.contour(
    ax=ax,
    x="lon",
    y="lat",
    levels=[-12, -8, -4, 4, 8, 12],
    colors="grey",
    transform=ccrs.PlateCarree(),
)
arrows = vectors.plot.quiver(
    ax=ax,
    x="lon", y="lat", u="fx", v="fy",
    transform=ccrs.PlateCarree(),
    scale=300,
    width=0.004,
    pivot="middle",
    add_guide=False,
)
ax.quiverkey(
    arrows,
    0.9, 1.04,
    10, "10 m$^2$ s$^{-2}$",
    coordinates="axes"
)
ax.set_title("TN01 WAF at 250 hPa: February 2025", size = 16, pad=25)

# %%
# Zonal and vertical propagation on a uniform display grid
# ------------------------------------------------------------
# Extract a latitude section after the full calculation. The pressure levels
# are irregular, and a log-pressure axis makes even pressure increments look
# uneven. Interpolate the flux in log(p), not pressure, so the arrow rows are
# evenly spaced on the displayed log-pressure axis. Longitude positions are
# also regularly spaced. This interpolation only affects the plotted arrows.
# The scalar shading below retains the original grid.
#
# ``angles="uv"`` expresses directions in display coordinates: positive fz
# points upward even though pressure increases downward. Using ``angles="xy"``
# with raw fz on a pressure axis would give the wrong coordinate interpretation.
# For legibility fz is multiplied by an explicitly stated factor of 100 only
# in the quiver dataset; the calculated fz and its units remain unchanged.
# The reference arrow states the equivalent horizontal and vertical magnitudes.
section = flux_3d.isel(time=0, drop=True).sel(lat=35, method="nearest")
section_latitude = section.lat
shaded_section_vertical = section.sel(level=slice(100, 850))
pressure = np.geomspace(850, 100, 12)

vectors_vertical = (
    section[["fx", "fz"]]
    .assign_coords(level=np.log(section.level.values))
    .interp(level=np.log(pressure), lon=np.linspace(0, 360, 41))
    .assign_coords(level=pressure)
)
vectors_vertical["fz_display"] = vectors_vertical.fz * 100
vectors_vertical

# %%
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

(shaded_section_vertical.psi_p).plot.contourf(
    ax=ax,
    x="lon",
    y="level",
    levels=np.linspace(-2e7, 2e7, 21),
    cmap="RdBu_r",
    extend="both",
    cbar_kwargs={
        "label": r"Streamfunction anomaly (m$^2$ s$^{-1}$)",
        "pad": 0.03,
        "format": formatter
    },
)

arrows = vectors_vertical.plot.quiver(
    ax=ax,
    x="lon", y="level", u="fx", v="fz_display",
    angles="uv",
    scale=200,
    width=0.003,
    pivot="middle",
    add_guide=False,
)
ax.quiverkey(
    arrows,
    0.72, 1.035,
    5, "$F_x=5$ or $F_z=0.05$ m$^2$ s$^{-2}$",
    coordinates="axes",
    labelpos="E",
    fontproperties={"size": 12},
)

ecl.plot.set_p_format_axis(ax=ax, axis="y", axis_limits=(850, 100), ticker_step=100)
ecl.plot.set_lon_format_axis(ax = ax, axis = "x")

ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 60))
ax.set_xlabel("Longitude (degrees east)")
ax.set_ylabel("Pressure (hPa)")
ax.set_title(
    f"TN01 zonal–vertical WAF at {section_latitude:.1f}°N (vertical arrows ×100)",
    pad=40, size = 16
)
