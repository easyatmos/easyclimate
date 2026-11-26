from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("easyclimate")
except PackageNotFoundError:
    __version__ = "0+unknown"


def show_versions() -> str:
    return __version__
