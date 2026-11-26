"""WRF interfaces; compiled backend exports are resolved on first access."""

from .. import backend as _backend
from .interface import *
from .interface import __all__ as _interface_exports

__all__ = list(_backend._WRF_EXPORTS) + _interface_exports


def __getattr__(name):
    if name not in _backend._WRF_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _backend.require(name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
