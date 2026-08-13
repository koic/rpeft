from ._errors import BackendError, stage

_KNOWN_DEVICES = ("cuda", "mps", "cpu")


def resolve_device(override=None):
    with stage("device"):
        import torch

        if override in (None, "", "auto"):
            device = _detect(torch)
            reason = "cpu fallback" if device == "cpu" else "%s available" % device
        else:
            device = _validate(torch, override)
            reason = "explicit override"

        return {
            "device": device,
            "dtype": _dtype_for(torch, device),
            "reason": reason,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }


def _detect(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate(torch, requested):
    if requested not in _KNOWN_DEVICES:
        raise BackendError("device", "unknown device: %s" % requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise BackendError("device", "cuda requested but not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise BackendError("device", "mps requested but not available")
    return requested


def _dtype_for(torch, device):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    # fp16/bf16 on MPS are numerically fragile; v0.1 sticks to float32
    # everywhere except bf16-capable CUDA.
    return "float32"


def empty_cache(device):
    import torch

    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
