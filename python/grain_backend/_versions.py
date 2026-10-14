import importlib.metadata
import re

from ._errors import BackendError

# transformers 4.56 is the first release that accepts the dtype argument the backend passes, and it loads
# checkpoints and .bin model files only with torch 2.6 or later.
MINIMUMS = {"torch": "2.6", "transformers": "4.56"}


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_outdated(name, version):
    minimum = MINIMUMS.get(name)
    if minimum is None or version is None:
        return False
    return _parse(version) < _parse(minimum)


def require_minimums():
    outdated = [
        "%s %s (needs %s or later)" % (name, version, MINIMUMS[name])
        for name in MINIMUMS
        for version in [installed_version(name)]
        if is_outdated(name, version)
    ]
    if outdated:
        raise BackendError("dependency", "outdated Python packages: %s" % ", ".join(outdated))


def _parse(version):
    # Local suffixes such as "2.6.0+cpu" and pre-release tags are ignored; only the release numbers count.
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()
