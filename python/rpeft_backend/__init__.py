"""Private Python backend for the RPEFT Ruby gem.

Imported through PyCall; not a public API. Every facade function takes
a JSON string and returns a JSON string, so only plain strings cross
the PyCall boundary.

This module must stay importable without torch installed: all heavy
imports live inside function bodies, and diagnostics() reports what is
missing instead of failing.
"""

import json
import math
import os
import sys

__version__ = "0.2.0"


def _dumps(value):
    # json.dumps writes NaN and Infinity, which are not JSON and which the Ruby side cannot parse.
    # A diverged run reports such losses, so they cross as null rather than failing the whole call.
    return json.dumps(_finite(value), allow_nan=False)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _repair_sys_executable():
    # PyCall embeds Python inside the Ruby process, so sys.executable
    # points at the Ruby binary. multiprocessing (used by datasets for
    # its resource tracker) would then spawn Ruby instead of Python.
    # Repair it from the PYTHON environment variable that PyCall honors.
    python = os.environ.get("PYTHON")
    if not python or not os.path.isfile(python):
        return
    if "python" in os.path.basename(sys.executable):
        return

    import multiprocessing

    sys.executable = python
    multiprocessing.set_executable(python)


_repair_sys_executable()


def train_lora(config_json):
    from . import _training, _versions

    _versions.require_minimums()

    return _dumps(_training.train_lora(json.loads(config_json)))


def merge_lora(config_json):
    from . import _merge, _versions

    _versions.require_minimums()

    return _dumps(_merge.merge_lora(json.loads(config_json)))


def generate(config_json):
    from . import _generate, _versions

    _versions.require_minimums()

    return _dumps(_generate.generate(json.loads(config_json)))


def detect_device(config_json="{}"):
    from . import _device

    config = json.loads(config_json)
    return _dumps(_device.resolve_device(config.get("device")))


def diagnostics():
    from . import _diagnostics

    return _dumps(_diagnostics.diagnostics())
