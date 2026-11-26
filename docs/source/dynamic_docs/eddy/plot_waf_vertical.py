# -*- coding: utf-8 -*-
r"""
.. _tn_waf_vertical_example:

TN01 Wave Activity Vertical Flux
============================================

The vertical component of Takaya--Nakamura wave activity flux describes
stationary-wave propagation in log-pressure height. This example uses
:func:`easyclimate.calc_TN_wave_activity_vertical_flux` to diagnose February
2025 relative to the February climatology for 1991--2020.

.. seealso::

    - :ref:`tn_waf_3d_example`
    - `Nishii's reference programs <https://www.atmos.rcast.u-tokyo.ac.jp/nishii/programs/index.html>`_
"""
from pathlib import Path

import easyclimate as ecl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %%
# Load the local sample
# --------------------------
# The input file contains only four fields: geopotential **height
# anomaly in metres**, basic-state zonal and meridional wind in m/s, and
# basic-state absolute temperature in K. The ERA5 geopotential was converted
# to height before saving the sample. Do not divide ``z_prime`` by gravity again.
#
# The sample retains native N80 spacing: 25 pressure levels (70--925 hPa),
# 67 Northern Hemisphere Gaussian latitudes, and 320 global longitudes.
# Only the required variables and a domain with derivative buffers are stored.
# Do not thin pressure or horizontal points before calculating the derivatives.
# The winds are the supplied ERA5 climatological winds, used as an approximation
# to the geostrophic basic state required by TN01.
sample = ecl.open_tutorial_dataset("tn_waf_era5_202502_sample")
sample


# %%
# Calculate the vertical flux on the full sample grid
# ------------------------------------------------------
# Keep the pressure, latitude and longitude dimensions during calculation:
# both horizontal and vertical neighbours are required. Select a pressure
# level or a cross-section **after** evaluating the flux.
#
# ``cyclic_lon=True`` is appropriate here because longitude covers the whole
# globe without a duplicated 0/360-degree endpoint. Use False for a region.
#
# The stationary-wave interpretation of the TN01 flux may become unreliable
# in weak-westerly regions, particularly near critical layers. The
# ``min_westerly`` parameter masks flux where the basic-state zonal wind U
# falls below the specified threshold, including easterly regions.
#
# The physical constants match the double-precision monthly NCL example.
# The default 1 m/s wind cutoff follows Nishii's Fortran implementation;
# set ``min_westerly=5.0`` to reproduce the monthly NCL wind mask.
# These are implementation-specific cutoffs, not universal critical wind
# speeds prescribed by TN01. Setting ``min_westerly=None`` disables this
# additional mask but does not remove the underlying physical limitations.
vertical_flux = ecl.calc_TN_wave_activity_vertical_flux(
    z_prime_data=sample.z_prime,
    u_climatology_data=sample.u_basic,
    v_climatology_data=sample.v_basic,
    temper_climatology_data=sample.t_basic,
    temper_climatology_data_units = "degK",
    vertical_dim="level",
    vertical_dim_units="hPa",
    cyclic_lon=True,
    min_westerly=1.0,
    min_latitude=10.0
)

vertical_flux

# %%
# Interpret the output
# --------------------------
#
# - ``fz``: upward wave activity flux, in m²/s²; negative values indicate
#   downward flux in log-pressure height, not pressure-coordinate velocity.
# - ``psi_p``: perturbation streamfunction, in m²/s.
# - ``N2``: squared buoyancy frequency in log-pressure height, in s⁻².
#
# The coordinate is :math:`z=-H\ln(p/p_0)`, with :math:`H = 8000` m and :math:`p_0 = 1000` hPa.
# Inspect stability alongside the flux, because ``fz`` contains ``1/N2``.
# Default masks remove weak/easterly basic-state zonal winds, tropical
# latitudes and nonpositive stability. Missing flux values are not zeros.
#
# A longitude--pressure section is selected only after evaluating the flux.
# Shading uses the actual pressure levels; no interpolation is needed to draw
# nonuniformly spaced scalar cells with xarray. The colour ranges below are
# fixed plotting ranges, with colourbar extensions for values beyond them.
section = (
    vertical_flux.isel(time=0, drop=True)
    .sel(lat=35.0, method="nearest")
    .sel(level=slice(100, 850))
)
section_latitude = float(section.lat)
section_latitude

# %%
fig, ax = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")

section.fz.plot.contourf(
    ax=ax[0],
    levels = np.linspace(-0.08, 0.08, 21),
    x="lon",
    y="level",
    yscale="log",
    cmap="RdBu_r",
    extend="both",
    cbar_kwargs={
        "label": r"Upward WAF $F_z$ (m$^2$ s$^{-2}$)",
        "ticks": [-0.08, -0.04, 0, 0.04, 0.08],
        "pad": 0.03,
    },
)

(section.N2 * 1e4).plot.contourf(
    ax=ax[1],
    levels = np.linspace(0, 3, 21),
    x="lon",
    y="level",
    yscale="log",
    cmap="jet",
    extend="max",
    cbar_kwargs={
        "label": r"$N^2$ ($10^{-4}$ s$^{-2}$)",
        "pad": 0.03,
    },
)

for axi in ax.flat:
    ecl.plot.set_p_format_axis(ax=axi, axis="y", axis_limits=(850, 100), ticker_step=100)
    ecl.plot.set_lon_format_axis(ax = axi, axis = "x")
    axi.set_xlim(0, 360)
    axi.set_xticks(np.arange(0, 361, 60))
    axi.set_xlabel("Longitude (degrees east)")
    axi.set_ylabel("Pressure (hPa)")
    axi.set_title("")
    axi.grid(alpha=0.2)

ax[0].set_title(f"Vertical TN01 flux at {section_latitude:.1f}°N")
ax[1].set_title("Basic-state stability")
fig.suptitle("February 2025 (1991–2020 basic state)", size = 16)
