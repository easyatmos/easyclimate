# -*- coding: utf-8 -*-
"""
Regridding
==========

Regrid zonal wind data from one regular latitude-longitude grid to another.
"""

import easyclimate as ecl
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %%
# Reading example raw grid data
u_data = ecl.tutorial.open_tutorial_dataset("uwnd_202201_mon_mean").sortby("lat").uwnd
u_data

# %%
# Define the target grid (only for **latitude/longitude and regular grids**)
target_grid = xr.DataArray(
    dims=("lat", "lon"),
    coords={
        "lat": np.arange(-89, 89, 6) + 1 / 1.0,
        "lon": np.arange(-180, 180, 6) + 1 / 1.0,
    },
)

# %%
# :py:func:`easyclimate.interp.interp_mesh2mesh <easyclimate.interp.interp_mesh2mesh>` performs a regridding operation.
#
# .. seealso::
#
#   https://github.com/EXCITED-CO2/xarray-regrid
regriding_data = ecl.interp.interp_mesh2mesh(u_data, target_grid)
regriding_data

# %%
# Plotting differences before and after interpolation
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

u_data.sel(level=500).isel(time=0).plot(ax=ax[0])
ax[0].set_title("Before", size=20)

regriding_data.sel(level=500).isel(time=0).plot(ax=ax[1])
ax[1].set_title("After", size=20)
