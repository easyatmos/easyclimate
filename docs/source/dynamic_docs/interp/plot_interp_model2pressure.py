# -*- coding: utf-8 -*-
"""
Interpolation from model layers to pressure layers
==================================================

Interpolate a zonal wind profile from model levels to pressure levels.
"""

import easyclimate as ecl
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %%
# Suppose the following data are available

# fmt: off
uwnd_data = xr.DataArray(
    np.array([
        -15.080393, -10.749071, -4.7920494, -2.3813725, -1.4152431,
        -0.6465206, -7.8181705, -14.718096, -14.65539, -14.948015,
        -13.705519, -11.443476, -8.865583, -7.9528713, -8.329103,
        -7.2445316, -6.7150173, -5.5189686, -4.139448, -3.2731838,
        -2.2931194, -1.0041752, -1.8983078, -2.3674374, -2.8203106,
        -3.2940865, -3.526329, -3.8654022, -4.164995, -4.2834396,
        -4.2950516, -4.3438225, -4.3716908, -4.7688255, -4.6155453,
        -4.5528393, -4.4831676, -4.385626, -4.2950516, -4.0953217,
    ]),
    dims=("model_level",),
    coords={
        "model_level": np.array([
            36, 44, 51, 56, 60, 63, 67, 70, 73, 75,
            78, 81, 83, 85, 88, 90, 92, 94, 96, 98,
            100, 102, 104, 105, 107, 109, 111, 113, 115, 117,
            119, 122, 124, 126, 129, 131, 133, 135, 136, 137,
        ])
    },
)

p_data = xr.DataArray(
    np.array([
        2081.4756, 3917.6995, 6162.6455, 8171.3506, 10112.652,
        11811.783, 14447.391, 16734.607, 19317.787, 21218.21,
        24357.875, 27871.277, 30436.492, 33191.027, 37698.96,
        40969.438, 44463.73, 48191.92, 52151.29, 56291.098,
        60525.63, 64770.367, 68943.69, 70979.66, 74908.17,
        78599.67, 82012.95, 85122.69, 87918.29, 90401.77,
        92584.94, 95338.72, 96862.08, 98165.305, 99763.38,
        100626.21, 101352.69, 101962.28, 102228.875, 102483.055,
    ]),
    dims=("model_level",),
    coords={
        "model_level": np.array([
            36, 44, 51, 56, 60, 63, 67, 70, 73, 75,
            78, 81, 83, 85, 88, 90, 92, 94, 96, 98,
            100, 102, 104, 105, 107, 109, 111, 113, 115, 117,
            119, 122, 124, 126, 129, 131, 133, 135, 136, 137,
        ])
    },
)
# fmt: on

# %%
# Now we interpolate the data located on the mode plane to the isobaric plane.
# Note that the units of the given isobaric surface are consistent with `pressure_data`.

result = ecl.interp.interp1d_vertical_model2pressure(
    pressure_data=p_data,
    variable_data=uwnd_data,
    vertical_input_dim="model_level",
    vertical_output_dim="plev",
    vertical_output_level=np.array(
        [100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 20000, 10000]
    ),
)
result

# %%
# Simply calibrate the interpolation effect.

fig, ax = plt.subplots()
ax.plot(p_data, uwnd_data, label="Original")
ax.plot(result.plev, result.data, "o", label="Interpolated")
ax.invert_xaxis()
ax.set_xlabel("Pressure [Pa]")
ax.set_ylabel("Zonal Wind [m/s]")
plt.legend()
