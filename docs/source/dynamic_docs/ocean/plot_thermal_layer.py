# -*- coding: utf-8 -*-
"""
Thermocline Depth and 20°C Isotherm Depth
=========================================

The thermocline core is the depth of strongest vertical cooling. For depth
increasing downward, it corresponds to the most negative vertical temperature
gradient. D20 is the depth at which the profile reaches 20°C. It is often used
as a proxy for the thermocline core in the equatorial Pacific, where the 20°C
isotherm often lies in the main thermocline.

This example compares two methods for each diagnostic on the same monthly
potential temperature field from SODA 3.4.2 in August 2020.

.. seealso::

    - Yang, H., & Wang, F. (2009). Revisiting the Thermocline Depth in the
      Equatorial Pacific. https://doi.org/10.1175/2009JCLI2836.1
    - Carton, J. A., Chepurin, G. A., & Chen, L. (2018). SODA3: A New Ocean
      Climate Reanalysis. https://doi.org/10.1175/JCLI-D-18-0149.1
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

import easyclimate as ecl

# %%
# Load the August 2020 temperature sample
# ---------------------------------------
# This example uses a losslessly compressed, single-month sample with the
# original global grid and all 50 depth levels. The depth coordinate is in
# meters and increases downward.
#
# :func:`easyclimate.open_tutorial_dataset` downloads the sample on first use
# and reuses the cached copy on subsequent runs.
#
# .. tip::
#
#    :download:`Download the full 2020 SODA temperature dataset <https://huggingface.co/datasets/shenyulu/easyclimate/resolve/main/tutorial_data/temp_soda3.4.2_mn_ocean_reg_2020_EN4.nc>`
#
#    After downloading the full file, open all 12 months with:
#
#    .. code-block:: python
#
#        import xarray as xr
#
#        dataset = xr.open_dataset(
#            "temp_soda3.4.2_mn_ocean_reg_2020_EN4.nc", chunks="auto"
#        )
#        temperature_2020 = dataset.temp.rename(st_ocean="depth")

with ecl.open_tutorial_dataset("sample_soda_temperature_202008") as dataset:
    temperature = dataset.temp.isel(time=0, drop=True)
    temperature = temperature.rename(st_ocean="depth").load()

# %%
# Compare the shallowest and deepest 20°C crossings
# -------------------------------------------------
# :func:`easyclimate.field.ocean.calc_D20_depth` uses linear interpolation
# between adjacent valid levels that straddle 20°C. Exact hits use the sampled
# depth. Profiles with no crossing or exact hit return NaN and appear white.
#
# * ``method="shallowest"`` (default) selects the first crossing from the surface.
# * ``method="deepest"`` selects the last crossing. Temperature inversions can
#   lead to multiple crossings in a profile.
#
# The two panels share a color scale. Each red ring in the right panel marks
# a grid cell whose temperature profile reaches 20°C at more than one distinct
# depth. The left panel shows the shallowest of those depths and the right panel
# shows the deepest. The blue diamond marks the profile plotted below.

d20_shallow = ecl.field.ocean.calc_D20_depth(temperature, method="shallowest")
d20_deep = ecl.field.ocean.calc_D20_depth(temperature, method="deepest")
d20_deep.plot()

# %%
geographic = ccrs.PlateCarree()


def plot_maps(fields, titles, *, label, cmap, vmin, vmax, extend="neither"):
    fig, axes = ecl.plot.quick_draw_spatial_basemap(
        2, 1, figsize=(6, 6), central_longitude=180
    )
    for ax, field, title in zip(axes, fields, titles):
        ax.set_extent([0.1, 359.9, -82, 89], crs=geographic)
        ax.add_feature(cfeature.LAND, facecolor="#dddddd", zorder=1)
        mesh = field.plot.pcolormesh(
            ax=ax,
            transform=geographic,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )
        ax.set_title(title)
    fig.colorbar(
        mesh,
        ax=axes,
        orientation="vertical",
        pad=0.05,
        label=label,
        extend=extend,
        shrink=0.9,
    )
    return fig, axes


# %%
fig, axes = plot_maps(
    (d20_shallow, d20_deep),
    ("D20 · Shallowest crossing", "D20 · Deepest crossing"),
    label="Depth (m)",
    cmap="viridis_r",
    vmin=0,
    vmax=600,
)

depth_difference = d20_deep - d20_shallow
different = (depth_difference > 0).transpose("lat", "lon")
lat_index, lon_index = np.where(different.values)
axes[1].scatter(
    different.lon.values[lon_index],
    different.lat.values[lat_index],
    transform=geographic,
    s=24,
    facecolors="none",
    edgecolors="red",
    zorder=3,
    label="Multiple 20°C depths",
)

profile_index = depth_difference.argmax(dim=["lat", "lon"])
profile = temperature.isel(profile_index)
axes[1].scatter(
    float(profile.lon),
    float(profile.lat),
    transform=geographic,
    marker="D",
    s=26,
    color="#225a9c",
    edgecolors="white",
    zorder=4,
    label="Profile shown below",
)
axes[1].legend(loc="lower left", fontsize=8)

# %%
# Inspect a profile with multiple 20°C crossings
# ----------------------------------------------
# The blue diamond marks the grid cell with the largest separation between
# the two D20 estimates in this example. Small dots show the sampled SODA
# levels, and the dashed vertical line shows 20°C. The colored markers show
# the two interpolated depths. The depth axis increases downward.

shallow_depth = float(d20_shallow.isel(profile_index))
deep_depth = float(d20_deep.isel(profile_index))
last_level = profile.get_index("depth").searchsorted(deep_depth, side="right")
upper_profile = profile.isel(depth=slice(None, last_level + 1))


# %%
fig, ax = plt.subplots(figsize=(3, 6), layout="constrained")

ax.plot(
    upper_profile.values,
    upper_profile.depth.values,
    color="#364152",
    marker=".",
    label="SODA temperature profile",
)
ax.axvline(20, color="0.45", linestyle="--", label="20°C")

ax.scatter(
    20,
    shallow_depth,
    color="#007b72",
    zorder=3,
    label=f"Shallowest: {shallow_depth:.1f} m",
)
ax.scatter(
    20,
    deep_depth,
    color="#c44e10",
    zorder=3,
    label=f"Deepest: {deep_depth:.1f} m",
)

latitude = float(profile.lat)
longitude = (float(profile.lon) + 180) % 360 - 180
lat_label = f"{abs(latitude):g}°{'N' if latitude >= 0 else 'S'}"
lon_label = f"{abs(longitude):g}°{'E' if longitude >= 0 else 'W'}"

ax.set_title(f"Temperature profile at ({lat_label}, {lon_label})")
ax.set_xlabel("Potential temperature (°C)")
ax.set_ylabel("Depth (m)")

ax.invert_yaxis()
ax.grid(alpha=0.25)
ax.legend(
    loc="center", bbox_to_anchor=(0.5, -0.33), shadow=False, ncol=1, frameon=False
)

# %%
# Compare two estimates of the maximum cooling depth
# --------------------------------------------------
# :func:`easyclimate.field.ocean.calc_seawater_thermocline_depth` selects the
# strongest cooling rate in each profile.
#
# * ``method="layer"`` (default) computes the temperature drop divided by the
#   depth spacing between adjacent valid levels. It returns the midpoint of
#   the interval with the largest cooling rate.
# * ``method="centered"`` estimates the derivative at each sampled level,
#   accounting for unequal spacing. It uses one-sided differences at the
#   ends of the depth coordinate and returns a sampled depth.
#
# Both methods search the full water column and return NaN when there is no
# positive cooling rate. At high latitudes the strongest cooling in a
# profile can occur below 1,000 m. The upper end of the colorbar includes
# values beyond the plotted scale.

thermocline_layer = ecl.field.ocean.calc_seawater_thermocline_depth(
    temperature, method="layer"
)
thermocline_centered = ecl.field.ocean.calc_seawater_thermocline_depth(
    temperature, method="centered"
)
thermocline_centered.plot()

# %%
plot_maps(
    (thermocline_layer, thermocline_centered),
    ("Thermocline · Layer midpoint", "Thermocline · Centered derivative"),
    label="Depth (m)",
    cmap="viridis_r",
    vmin=0,
    vmax=1000,
    extend="max",
)

# %%
# Compare differences in depth
# ----------------------------
# In the left panel, positive values mean the centered method places the
# thermocline deeper than the layer method. In the right panel, positive
# values mean the 20°C isotherm is deeper than the layer-based thermocline.
# The latter difference is shown where both depths are defined.

# sphinx_gallery_thumbnail_number = 5
plot_maps(
    (thermocline_centered - thermocline_layer, d20_shallow - thermocline_layer),
    ("Centered − layer thermocline", "Shallowest D20 − layer thermocline"),
    label="Depth difference (m)",
    cmap="RdBu_r",
    vmin=-150,
    vmax=150,
    extend="both",
)
