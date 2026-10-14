import importlib
import importlib.util
import platform
import sys

LIBRARIES = ("torch", "transformers", "peft", "datasets", "accelerate")


def diagnostics():
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "backend_version": _backend_version(),
        "libraries": {name: _library_info(name) for name in LIBRARIES},
        "device": _device_info(),
    }


def _backend_version():
    from . import __version__

    return __version__


def _library_info(name):
    from . import _versions

    minimum = _versions.MINIMUMS.get(name)
    if importlib.util.find_spec(name) is None:
        return {"installed": False, "version": None, "minimum": minimum, "outdated": False}
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        return {
            "installed": True,
            "version": version,
            "minimum": minimum,
            "outdated": _versions.is_outdated(name, version),
        }
    except Exception as error:
        return {
            "installed": False,
            "version": None,
            "minimum": minimum,
            "outdated": False,
            "error": str(error),
        }


def _device_info():
    try:
        import torch
    except Exception:
        return None

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        recommended = "cuda"
    elif mps_available:
        recommended = "mps"
    else:
        recommended = "cpu"

    return {
        "cuda": {
            "available": cuda_available,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        },
        "mps": {
            "available": mps_available,
            "built": torch.backends.mps.is_built(),
        },
        "recommended": recommended,
    }
