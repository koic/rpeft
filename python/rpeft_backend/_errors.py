import contextlib


class BackendError(RuntimeError):
    """Error whose message carries a machine-readable stage token.

    The Ruby side parses the "[stage:...]" prefix to pick the matching
    Ruby exception class; see lib/rpeft/backend/error_translation.rb.
    """

    def __init__(self, stage, message):
        super().__init__("[stage:%s] %s" % (stage, message))
        self.stage = stage


@contextlib.contextmanager
def stage(name):
    """Tag any exception escaping this block with the given stage name."""
    try:
        yield
    except BackendError:
        raise
    except Exception as error:
        raise BackendError(name, "%s: %s" % (type(error).__name__, error)) from error
