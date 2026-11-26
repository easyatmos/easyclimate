"""Optional backends must never be required to import EasyClimate."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def backend():
    path = Path(__file__).parents[1] / "src/easyclimate/backend.py"
    spec = importlib.util.spec_from_file_location("_test_lazy_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_without_any_optional_backend():
    script = r"""
import importlib.abc
import importlib.util
import sys
import warnings

roots = {"easyclimate_backend", "easyclimate_rust", "spectralkernel_rs"}
attempts = []
class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in roots:
            attempts.append(fullname)
            raise ModuleNotFoundError("Blocked optional backend", name=fullname)

original_find_spec = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    if name in roots:
        return None
    return original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = find_spec
sys.meta_path.insert(0, BlockBackends())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import easyclimate as ecl
assert not attempts, attempts
assert not roots.intersection(sys.modules)
messages = [str(w.message) for w in caught if "Optional EasyClimate backends" in str(w.message)]
assert len(messages) == 1, messages
assert all(p in messages[0] for p in ("easyclimate-backend", "easyclimate-rust", "spectralkernel-rs"))
assert "getvar" in dir(ecl.wrf)
assert callable(ecl.wrf.open_wrf_data)
from easyclimate import backend
for name, package in [("dvrfidf_rs", "easyclimate-rust"), ("spharm", "easyclimate-backend"), ("VectorWind_rs", "spectralkernel-rs")]:
    try:
        backend.require(name)
    except ImportError as exc:
        assert "python -m pip install " + package in str(exc), str(exc)
    else:
        raise AssertionError(name)
try:
    ecl.wrf.getvar
except ImportError as exc:
    assert "easyclimate-backend" in str(exc)
else:
    raise AssertionError("WRF backend unexpectedly loaded")
import numpy as np
import xarray as xr
data = xr.DataArray(np.arange(24.).reshape(6, 2, 2), dims=("time", "lat", "lon"))
result = ecl.calc_detrend_spatial_fast(data, method="auto")
assert result.shape == data.shape
assert np.allclose(result, 0, atol=1e-10)
try:
    ecl.calc_detrend_spatial_fast(data, method="rust")
except ImportError as exc:
    assert "python -m pip install easyclimate-rust" in str(exc), str(exc)
else:
    raise AssertionError("Explicit Rust selection must explain the missing dependency")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loading_is_cached_and_capabilities_are_independent(backend, monkeypatch):
    function = lambda: 42
    calls = []

    def load(name):
        calls.append(name)
        if name.startswith("spectralkernel_rs"):
            raise OSError("DLL load failed")
        return SimpleNamespace(dvrfidf=function)

    monkeypatch.setattr(backend, "import_module", load)
    assert not backend.is_available("VectorWind_rs")
    assert backend.require("dvrfidf_rs") is function
    assert backend.dvrfidf_rs() == 42
    assert backend.require("dvrfidf_rs") is function
    assert calls.count("easyclimate_rust._easyclimate_rust") == 1


@pytest.mark.parametrize(
    "error",
    [
        ImportError("ABI mismatch"),
        OSError("DLL load failed"),
        AttributeError("missing symbol"),
        ValueError("numpy.dtype size changed"),
    ],
)
def test_load_failure_preserves_cause(backend, monkeypatch, error):
    def fail(name):
        raise error

    monkeypatch.setattr(backend, "import_module", fail)
    with pytest.raises(backend.BackendUnavailableError) as caught:
        backend.require("dvrfidf_rs")
    assert caught.value.__cause__ is error
    assert str(error) in str(caught.value)
    assert "compatible" in str(caught.value)


def test_missing_transitive_dependency_is_not_misreported(backend, monkeypatch):
    def fail(name):
        raise ModuleNotFoundError("No module named 'numba'", name="numba")

    monkeypatch.setattr(backend, "import_module", fail)
    with pytest.raises(backend.BackendUnavailableError, match="numba") as caught:
        backend.require("barnes_numba")
    assert "python -m pip install easyclimate-rust" not in str(caught.value)


def test_failed_load_can_be_retried(backend, monkeypatch):
    def fail(name):
        raise ModuleNotFoundError(name="easyclimate_rust")

    monkeypatch.setattr(backend, "import_module", fail)
    assert not backend.is_available("dvrfidf_rs")
    function = lambda: None
    monkeypatch.setattr(
        backend, "import_module", lambda name: SimpleNamespace(dvrfidf=function)
    )
    assert backend.require("dvrfidf_rs") is function


@pytest.mark.parametrize(
    "name",
    [
        "wk_analysis",
        "matsuno_plot",
        "spharm",
        "aeronoskin",
        "aeroskin",
        "human_index_mod",
        "human_index_mod_old",
        "_ecl_redfit",
        "_ecl_redfit_x",
        "_wet_bulb_temperature",
        "spectral_tool_rs",
        "xarray_regrid",
    ],
)
def test_module_exports_are_loaded_as_modules(backend, monkeypatch, name):
    module = SimpleNamespace()
    imported = []

    def load(module_name):
        imported.append(module_name)
        return module

    monkeypatch.setattr(backend, "import_module", load)
    assert backend.require(name) is module
    assert imported == [backend._BACKENDS[name][0]]


def test_unknown_symbol(backend):
    with pytest.raises(AttributeError):
        backend.require("unknown_symbol")


def test_computation_error_is_not_wrapped(backend, monkeypatch):
    error = ValueError("invalid input shape")

    def compute():
        raise error

    monkeypatch.setattr(
        backend, "import_module", lambda name: SimpleNamespace(dvrfidf=compute)
    )
    with pytest.raises(ValueError) as caught:
        backend.require("dvrfidf_rs")()
    assert caught.value is error


def test_presence_check_does_not_load_backends(backend, monkeypatch):
    inspected = []

    def find(name):
        inspected.append(name)
        return object()

    def load(name):
        pytest.fail("Presence checks must not import optional backends")

    for name in backend._PACKAGES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(backend, "find_spec", find)
    monkeypatch.setattr(backend, "import_module", load)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend._warn_missing_backends()
    assert not caught
    assert set(inspected) == set(backend._PACKAGES)
