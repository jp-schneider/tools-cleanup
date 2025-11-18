try:
    import torch
except ImportError:
    torch = None  # type: ignore
    pass


class TorchDeterministicAlgorithms():
    """Context manager for changing the torch deterministic algorithms behavior."""

    def __init__(self, only_deterministic: bool = True, warn_only: bool = True):
        """Creates the context manager for changing the torch deterministic algorithms behavior.

        Parameters
        ----------
        only_deterministic : bool
            If True, only deterministic algorithms will be used. If False, non-deterministic algorithms are allowed.

        warn_only : bool
            If True, only warnings will be shown instead of raising errors when non-deterministic algorithms are used.

        """
        if torch is None:
            raise ImportError(
                "torch is not installed, cannot use TorchDeterministicWarning context manager.")
        self.only_deterministic = only_deterministic
        self.warn_only = warn_only
        self._old_deterministic_value = None
        self._old_warn_value = None

    def __enter__(self):
        self._old_deterministic_value = torch._C._get_deterministic_algorithms()
        self._old_warn_value = torch._C._get_deterministic_algorithms_warn_only()
        torch._C._set_deterministic_algorithms(
            self.only_deterministic, warn_only=self.warn_only)

    def __exit__(self, type, value, traceback):
        torch._C._set_deterministic_algorithms(
            self._old_deterministic_value, warn_only=self._old_warn_value)
        return False
