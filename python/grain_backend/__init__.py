"""Private Python backend for the Grain Ruby gem.

Imported through PyCall; not a public API. Every facade function takes
a JSON string and returns a JSON string, so only plain strings cross
the PyCall boundary.

This module must stay importable without torch installed: all heavy
imports live inside function bodies, and diagnostics() reports what is
missing instead of failing.
"""

import json
import os
import sys

__version__ = "0.1.0"


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
    from . import _training

    return json.dumps(_training.train_lora(json.loads(config_json)))


def merge_lora(config_json):
    from . import _merge

    return json.dumps(_merge.merge_lora(json.loads(config_json)))


def generate(config_json):
    from . import _generate

    return json.dumps(_generate.generate(json.loads(config_json)))


def detect_device(config_json="{}"):
    from . import _device

    config = json.loads(config_json)
    return json.dumps(_device.resolve_device(config.get("device")))


def diagnostics():
    from . import _diagnostics

    return json.dumps(_diagnostics.diagnostics())
