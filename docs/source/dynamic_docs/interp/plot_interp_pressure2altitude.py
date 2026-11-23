# -*- coding: utf-8 -*-
"""
Interpolation from pressure layers to altitude layers
=====================================================

Interpolate a zonal wind profile from pressure levels to a regular altitude grid.
"""

import easyclimate as ecl
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %%
# Suppose the following data are available

# fmt: off
z_data = xr.DataArray(
    np.array([
        214.19354, 841.6774, 1516.871, 3055.7097, 4260.5806,
        5651.4194, 7288.032, 9288.193, 10501.097, 11946.71,
        13762.322, 16233.451, 18370.902, 20415.227, 23619.033,
        26214.322, 30731.807,
    ]),
    dims=("level"),
    coords={
        "level": np.array([
            1000.0, 925.0, 850.0, 700.0, 600.0,
            500.0, 400.0, 300.0, 250.0, 200.0,
            150.0, 100.0, 70.0, 50.0, 30.0,
            20.0, 10.0,
        ])
    },
)

uwnd_data = xr.DataArray(
    np.array([
        -2.3200073, -3.5700073, -2.5800018, 8.080002, 14.059998,
        22.119995, 33.819992, 49.339996, 57.86, 64.009995,
        62.940002, 49.809998, 31.160004, 16.59999, 10.300003,
        10.459991, 9.880005,
    ]),
    dims=("level"),
    coords={
        "level": np.array([
            1000.0, 925.0, 850.0, 700.0, 600.0,
            500.0, 400.0, 300.0, 250.0, 200.0,
            150.0, 100.0, 70.0, 50.0, 30.0,
            20.0, 10.0,
        ])
    },
)
# fmt: on

# %%
# Then we need to interpolate the zonal wind data (located on the isobaric surface) to the altitude layers.

target_heights = np.linspace(0, 10000, 101)

result = ecl.interp.interp1d_vertical_pressure2altitude(
    z_data=z_data,
    variable_data=uwnd_data,
    target_heights=target_heights,
    vertical_input_dim="level",
    vertical_output_dim="altitude",
)
result

# %%
# Now we can check the interpolation results.
plt.plot(z_data[:9], uwnd_data[:9], "o", label="Original")
plt.plot(result.altitude, result.data, label="Interpolated")
plt.xlabel("Altitude [m]")
plt.ylabel("Zonal Wind [m/s]")
plt.legend()
