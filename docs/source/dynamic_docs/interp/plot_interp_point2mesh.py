# -*- coding: utf-8 -*-
"""
Interpolation from points to grid
=================================

Interpolate surface pressure observations to a regular grid using Barnes interpolation.
"""

import easyclimate as ecl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %%
# Open sample surface pressure data for the European region
#
#
# .. warning::
#
#       Although the original version only support specific geographical domain and projection (as in Zürcher, B. K., 2023), which is fixed to the European latitudes and Lambert conformal projection and cannot be freely chosen.
#
#       However, with moderate modifications, it now supports interpolation across global regions.
#       Except for method ``"optimized_convolution_S2"`` in the function :py:func:`easyclimate.interp.interp_spatial_barnesS2 <easyclimate.interp.interp_spatial_barnesS2>`, which for objective reasons can only support interpolation for a single hemisphere (either northern or southern) and cannot exceed 180 degrees across the entire interpolation longitude range.


station_data = ecl.open_tutorial_dataset("PressQFF_202007271200_872.csv")
print(station_data)

# %%
# :py:func:`easyclimate.interp.interp_spatial_barnes <easyclimate.interp.interp_spatial_barnes>` enables interpolation from site data to grid point data.
#
# .. seealso::
#
#     - https://github.com/MeteoSwiss/fast-barnes-py
#     - Zürcher, B. K.: Fast approximate Barnes interpolation: illustrated by Python-Numba implementation fast-barnes-py v1.0, Geosci. Model Dev., 16, 1697–1711, https://doi.org/10.5194/gmd-16-1697-2023, 2023.
meshdata_numba = ecl.interp.interp_spatial_barnes(
    station_data,
    "qff",
    lon_dim="lon",
    lat_dim="lat",
    grid_res_deg=0.25,
    sigma_deg=0.5,
    influence_radius_deg=None,  # Default = 4*sigma
    mask_radius_deg=None,  # Default = influence radius
    method="optimized_convolution",
    buffer_deg=5.0,
)
meshdata_numba

# %%
# Or use functions :py:func:`easyclimate.interp.interp_spatial_barnes_rs <easyclimate.interp.interp_spatial_barnes_rs>` accelerated by Rust
meshdata_rust = ecl.interp.interp_spatial_barnes_rs(
    station_data,
    "qff",
    lon_dim="lon",
    lat_dim="lat",
    grid_res_deg=0.25,
    sigma_deg=0.5,
    influence_radius_deg=None,  # Default = 4*sigma
    mask_radius_deg=None,  # Default = influence radius
    method="optimized_convolution",
    buffer_deg=5.0,
)
meshdata_rust

# %%
# Plotting interpolated grid point data and corresponding station locations
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})

ax.gridlines(draw_labels=["bottom", "left"], color="grey", alpha=0.5, linestyle="--")
ax.coastlines(edgecolor="black", linewidths=0.5)
ax.set_extent([-30, 50, 32, 75])

# Draw interpolation results
meshdata_numba.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"location": "bottom", "aspect": 50, "shrink": 0.9},
    cmap="RdBu_r",
    levels=21,
)

# Draw observation stations
ax.scatter(
    station_data["lon"], station_data["lat"], s=1, c="r", transform=ccrs.PlateCarree()
)
ax.set_title("Method: Numba, optimized_convolution")

# %%
# Next, we consider the results obtained using Rust.
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})

ax.gridlines(draw_labels=["bottom", "left"], color="grey", alpha=0.5, linestyle="--")
ax.coastlines(edgecolor="black", linewidths=0.5)
ax.set_extent([-30, 50, 32, 75])

# Draw interpolation results
meshdata_rust.plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"location": "bottom", "aspect": 50, "shrink": 0.9},
    cmap="RdBu_r",
    levels=21,
)

# Draw observation stations
ax.scatter(
    station_data["lon"], station_data["lat"], s=1, c="r", transform=ccrs.PlateCarree()
)
ax.set_title("Method: Rust, optimized_convolution")

# %%
# Now, when we compare the computational results of the two, we can see that only the machine precision differs.
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})

ax.gridlines(draw_labels=["bottom", "left"], color="grey", alpha=0.5, linestyle="--")
ax.coastlines(edgecolor="black", linewidths=0.5)
ax.set_extent([-30, 50, 32, 75])

# Draw interpolation results
(meshdata_rust - meshdata_numba).plot.contourf(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"location": "bottom", "aspect": 50, "shrink": 0.9},
    cmap="RdBu_r",
)

# Draw observation stations
ax.scatter(
    station_data["lon"], station_data["lat"], s=1, c="r", transform=ccrs.PlateCarree()
)
ax.set_title("Method: Rust, optimized_convolution")
