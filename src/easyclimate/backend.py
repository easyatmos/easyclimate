"""Lazy access to optional compiled and Python backends."""

import platform
import os
import site
import warnings
from importlib import import_module
from importlib.util import find_spec
from threading import RLock
import sys
from pathlib import Path


# Detect the current operating system
CURRENT_PLATFORM = platform.system()
SUPPORTED_BACKEND_PLATFORMS = ("Windows", "Linux", "Darwin")

_WINDOWS_DLL_DIRECTORY_HANDLES = []


def _register_windows_dll_directories():
    if CURRENT_PLATFORM != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    candidate_dirs = []
    prefixes = [
        sys.prefix,
        sys.base_prefix,
        os.environ.get("CONDA_PREFIX"),
    ]
    for prefix in prefixes:
        if prefix:
            candidate_dirs.append(Path(prefix) / "Library" / "bin")

    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        site_dirs.append(site.getusersitepackages())
    except AttributeError:
        pass

    for site_dir in site_dirs:
        site_path = Path(site_dir)
        candidate_dirs.extend(
            [
                site_path / "easyclimate_backend.libs",
                site_path / "easyclimate_backend" / ".libs",
                site_path / "Library" / "bin",
            ]
        )

    oneapi_root = os.environ.get("ONEAPI_ROOT")
    if oneapi_root:
        candidate_dirs.append(Path(oneapi_root) / "compiler" / "latest" / "bin")

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        candidate_dirs.append(
            Path(program_files_x86) / "Intel" / "oneAPI" / "compiler" / "latest" / "bin"
        )

    seen = set()
    for candidate_dir in candidate_dirs:
        candidate_dir = candidate_dir.resolve()
        if candidate_dir in seen or not candidate_dir.is_dir():
            continue
        seen.add(candidate_dir)
        try:
            _WINDOWS_DLL_DIRECTORY_HANDLES.append(
                os.add_dll_directory(str(candidate_dir))
            )
        except OSError:
            pass


_BACKENDS = {
    "wk_analysis": ("easyclimate_backend.wk_spectra.wk_analysis", None),
    "matsuno_plot": ("easyclimate_backend.wk_spectra.matsuno_plot", None),
    "wave_signif": ("easyclimate_backend.wavelet.waveletFunctions", "wave_signif"),
    "wavelet": ("easyclimate_backend.wavelet.waveletFunctions", "wavelet"),
    "spharm": ("easyclimate_backend.pyspharm.spharm", None),
    "VectorWind": ("easyclimate_backend.windspharm.xarray", "VectorWind"),
    "aeronoskin": ("easyclimate_backend.aerobulk.mod_aerobulk_wrap_noskin", None),
    "aeroskin": ("easyclimate_backend.aerobulk.mod_aerobulk_wrap_skin", None),
    "human_index_mod": ("easyclimate_backend.heat_stress.human_index_mod", None),
    "human_index_mod_old": (
        "easyclimate_backend.heat_stress.human_index_mod_old",
        None,
    ),
    "_ecl_redfit": ("easyclimate_backend.redfit._ecl_redfit", None),
    "_ecl_redfit_x": ("easyclimate_backend.redfit._ecl_redfit_x", None),
    "_vinth2p_dp": ("easyclimate_backend.vinth2p._vinth2p_dp", "vinth2p"),
    "_vinth2p_ecmwf": ("easyclimate_backend.vinth2p._vinth2p_ecmwf", "vinth2pecmwf"),
    "_vintp2p_ecmwf": ("easyclimate_backend.vinth2p._vintp2p_ecmwf", "vintp2pecmwf"),
    "dvibeta_ncl": ("easyclimate_backend.vibeta._vibeta_dp", "dvibeta"),
    "ddvfidf_ncl": ("easyclimate_backend.rvdv._rvdv", "ddvfidf"),
    "dvrfidf_ncl": ("easyclimate_backend.rvdv._rvdv", "dvrfidf"),
    "_wet_bulb_temperature": (
        "easyclimate_backend.wet_bulb._wet_bulb_temperature",
        None,
    ),
    "calc_wet_bulb_temperature_rs": (
        "easyclimate_rust._easyclimate_rust",
        "calc_wet_bulb_temperature",
    ),
    "calc_sphere_laplacian_numpy_rs": (
        "easyclimate_rust._easyclimate_rust",
        "calc_sphere_laplacian_numpy",
    ),
    "calc_sphere_laplacian_conservative_numpy_rs": (
        "easyclimate_rust._easyclimate_rust",
        "calc_sphere_laplacian_conservative_numpy",
    ),
    "calc_detrend_spatial_3d_rs": (
        "easyclimate_rust._easyclimate_rust",
        "calc_detrend_spatial_3d",
    ),
    "calc_detrend_spatial_3d_chunked_rs": (
        "easyclimate_rust._easyclimate_rust",
        "calc_detrend_spatial_3d_chunked",
    ),
    "calc_detrend_spatial_flexible_rs": (
        "easyclimate_rust._easyclimate_rust",
        "calc_detrend_spatial_flexible",
    ),
    "interp1d_linear_core_rs": (
        "easyclimate_rust._easyclimate_rust",
        "interp1d_linear_core",
    ),
    "interp1d_linear_2d_rs": (
        "easyclimate_rust._easyclimate_rust",
        "interp1d_linear_2d",
    ),
    "interp1d_linear_3d_rs": (
        "easyclimate_rust._easyclimate_rust",
        "interp1d_linear_3d",
    ),
    "interp1d_linear_4d_rs": (
        "easyclimate_rust._easyclimate_rust",
        "interp1d_linear_4d",
    ),
    "barnes_rs": ("easyclimate_rust._easyclimate_rust", "barnes"),
    "barnesS2_rs": ("easyclimate_rust._easyclimate_rust", "barnes_s2"),
    "radius_mask_2d_rs": ("easyclimate_rust._easyclimate_rust", "radius_mask_2d"),
    "barnes_numba": ("easyclimate_rust.fastbarnes.interpolation", "barnes"),
    "barnes_S2_numba": ("easyclimate_rust.fastbarnes.interpolationS2", "barnes_S2"),
    "kdtree_fastbarnes": ("easyclimate_rust.fastbarnes.util", "kdtree"),
    "ddvfidf_rs": ("easyclimate_rust._easyclimate_rust", "ddvfidf"),
    "dvrfidf_rs": ("easyclimate_rust._easyclimate_rust", "dvrfidf"),
    "ddvfidf_batch_rs": ("easyclimate_rust._easyclimate_rust", "ddvfidf_batch"),
    "dvrfidf_batch_rs": ("easyclimate_rust._easyclimate_rust", "dvrfidf_batch"),
    "dvibeta_rs": ("easyclimate_rust._easyclimate_rust", "dvibeta"),
    "dvibeta_batch": ("easyclimate_rust._easyclimate_rust", "dvibeta_batch"),
    "dvibeta_batch_sum_norm": (
        "easyclimate_rust._easyclimate_rust",
        "dvibeta_batch_sum_norm",
    ),
    "VectorWind_rs": ("spectralkernel_rs.ws_xarray", "VectorWind"),
    "spectral_tool_rs": ("spectralkernel_rs.tool", None),
    "Spharmt_rs": ("spectralkernel_rs.spharmt_rs", "Spharmt_rs"),
    "ddvfidf_batch_noreyon_rs": (
        "easyclimate_rust._easyclimate_rust",
        "ddvfidf_batch_noreyon",
    ),
    "dvrfidf_batch_noreyon_rs": (
        "easyclimate_rust._easyclimate_rust",
        "dvrfidf_batch_noreyon",
    ),
    "dvibeta_batch_noreyon": (
        "easyclimate_rust._easyclimate_rust",
        "dvibeta_batch_noreyon",
    ),
    "dvibeta_batch_sum_norm_noreyon": (
        "easyclimate_rust._easyclimate_rust",
        "dvibeta_batch_sum_norm_noreyon",
    ),
    "gill_simulate": ("easyclimate_rust._easyclimate_rust", "gill_simulate"),
    "xarray_regrid": ("easyclimate_backend.xarray_regrid", None),
}

_WRF_EXPORTS = (
    "xarray_enabled",
    "disable_xarray",
    "enable_xarray",
    "cartopy_enabled",
    "disable_cartopy",
    "enable_cartopy",
    "basemap_enabled",
    "disable_basemap",
    "enable_basemap",
    "pyngl_enabled",
    "enable_pyngl",
    "disable_pyngl",
    "set_cache_size",
    "get_cache_size",
    "omp_enabled",
    "ALL_TIMES",
    "Constants",
    "ConversionFactors",
    "ProjectionTypes",
    "default_fill",
    "OMP_SCHED_STATIC",
    "OMP_SCHED_DYNAMIC",
    "OMP_SCHED_GUIDED",
    "OMP_SCHED_AUTO",
    "destagger",
    "getvar",
    "xy",
    "interp1d",
    "interp2dxy",
    "interpz3d",
    "slp",
    "tk",
    "td",
    "rh",
    "uvmet",
    "smooth2d",
    "cape_2d",
    "cape_3d",
    "cloudfrac",
    "ctt",
    "dbz",
    "srhel",
    "udhel",
    "avo",
    "pvo",
    "eth",
    "wetbulb",
    "tvirtual",
    "omega",
    "pw",
    "DiagnosticError",
    "omp_set_num_threads",
    "omp_get_num_threads",
    "omp_get_max_threads",
    "omp_get_thread_num",
    "omp_get_num_procs",
    "omp_in_parallel",
    "omp_set_dynamic",
    "omp_get_dynamic",
    "omp_set_nested",
    "omp_get_nested",
    "omp_set_schedule",
    "omp_get_schedule",
    "omp_get_thread_limit",
    "omp_set_max_active_levels",
    "omp_get_max_active_levels",
    "omp_get_level",
    "omp_get_ancestor_thread_num",
    "omp_get_team_size",
    "omp_get_active_level",
    "omp_in_final",
    "omp_init_lock",
    "omp_init_nest_lock",
    "omp_destroy_lock",
    "omp_destroy_nest_lock",
    "omp_set_lock",
    "omp_set_nest_lock",
    "omp_unset_lock",
    "omp_unset_nest_lock",
    "omp_test_lock",
    "omp_test_nest_lock",
    "omp_get_wtime",
    "omp_get_wtick",
    "interplevel",
    "vertcross",
    "interpline",
    "vinterp",
    "xy_to_ll",
    "ll_to_xy",
    "xy_to_ll_proj",
    "ll_to_xy_proj",
    "viewitems",
    "viewkeys",
    "viewvalues",
    "isstr",
    "py2round",
    "py3range",
    "ucode",
    "to_np",
    "extract_global_attrs",
    "is_standard_wrf_var",
    "extract_dim",
    "extract_vars",
    "extract_times",
    "combine_files",
    "npbytes_to_str",
    "is_moving_domain",
    "is_staggered",
    "get_left_indexes",
    "iter_left_indexes",
    "get_right_slices",
    "get_proj_params",
    "from_args",
    "args_to_list",
    "arg_location",
    "psafilepath",
    "get_id",
    "from_var",
    "combine_dims",
    "either",
    "get_iterable",
    "IterWrapper",
    "is_coordvar",
    "latlon_coordvars",
    "is_mapping",
    "has_time_coord",
    "is_multi_file",
    "is_multi_time_req",
    "get_coord_pairs",
    "is_time_coord_var",
    "geo_bounds",
    "get_cartopy",
    "get_basemap",
    "get_pyngl",
    "cartopy_xlim",
    "cartopy_ylim",
    "latlon_coords",
    "ll_points",
    "pairs_to_latlon",
    "GeoBounds",
    "NullGeoBounds",
    "WrfProj",
    "NullProjection",
    "LambertConformal",
    "Mercator",
    "PolarStereographic",
    "LatLon",
    "RotatedLatLon",
    "getproj",
    "CoordPair",
    "to_xy_coords",
    "cache_item",
    "get_cached_item",
)

_BACKENDS.update({name: ("easyclimate_backend.wrf", name) for name in _WRF_EXPORTS})

_PACKAGES = {
    "easyclimate_backend": "easyclimate-backend",
    "easyclimate_rust": "easyclimate-rust",
    "spectralkernel_rs": "spectralkernel-rs",
}
_LOAD_LOCK = RLock()
_dll_directories_registered = False


class BackendUnavailableError(ImportError):
    """An optional backend is missing, incompatible, or cannot be loaded."""


def _warn_missing_backends():
    # Only inspect top-level packages: finding a dotted module may import its parent.
    missing = []
    for module, package in _PACKAGES.items():
        if module in sys.modules:
            installed = sys.modules[module] is not None
        else:
            try:
                installed = find_spec(module) is not None
            except (ImportError, ValueError):
                installed = False
        if not installed:
            missing.append(package)
    if missing:
        warnings.warn(
            "Optional EasyClimate backends are not installed: "
            + ", ".join(missing)
            + ". Other functionality remains available. Install with: python -m pip install "
            + " ".join(missing)
            + ". Backend loading is deferred until use.",
            UserWarning,
            stacklevel=2,
        )


def require(name):
    """Load and cache a backend symbol, with actionable dependency errors.

    Successful loads are cached; failed loads can be retried. Exceptions raised
    by the computation itself are never intercepted here.
    """
    global _dll_directories_registered
    if name not in _BACKENDS:
        raise AttributeError(f"Unknown EasyClimate backend symbol: {name!r}")
    with _LOAD_LOCK:
        if name in globals():
            return globals()[name]
        module_name, attribute = _BACKENDS[name]
        package = _PACKAGES[module_name.split(".")[0]]
        if not _dll_directories_registered:
            _register_windows_dll_directories()
            _dll_directories_registered = True
        try:
            module = import_module(module_name)
            value = module if attribute is None else getattr(module, attribute)
        except ModuleNotFoundError as exc:
            root_module = module_name.split(".")[0]
            if exc.name == root_module:
                message = (
                    f"Backend symbol '{name}' requires {package}. "
                    f"Install it in the current Python environment: python -m pip install {package}"
                )
            else:
                message = (
                    f"Cannot load backend symbol '{name}' from {package}: "
                    f"module '{exc.name}' is missing. Check {package}'s installation "
                    f"and its dependencies. Original error: {exc}"
                )
            raise BackendUnavailableError(message) from exc
        except (ImportError, OSError, AttributeError, ValueError) as exc:
            raise BackendUnavailableError(
                f"Cannot load backend symbol '{name}' from {package}. "
                "Check that its version, Python/NumPy ABI and native runtime libraries "
                f"are compatible with this environment. Original error: {exc}"
            ) from exc
        globals()[name] = value
        return value


def is_available(name):
    """Check a specific capability on demand, including binary loadability."""
    try:
        require(name)
    except BackendUnavailableError:
        return False
    return True


def __getattr__(name):
    # Backward compatibility for callers importing individual backend symbols.
    if name == "RUST_AVAILABLE":
        return is_available("calc_detrend_spatial_3d_rs")
    return require(name)


def __dir__():
    return sorted(set(globals()) | set(_BACKENDS) | {"RUST_AVAILABLE"})


_warn_missing_backends()
