from typing import Any

from tools.transforms.transform import Transform


class FittableTransform(Transform):
    """Abstract class for fittable transform.
    This transform can be fitted to the data. Can only be executed after fitting."""

    def __init__(self,
                 auto_fit: bool = False,
                 persistent: bool = True
                 ) -> None:
        """Constructor for FittableTransform.

        Parameters
        ----------
        auto_fit : bool, optional
            If the fit method should be called automatically when invoking transform without a prior fit., by default False
        persistent : bool, optional
            If the transform should be persistent after fitting, by default True. If False, the transform will be reset after each transform call.
        """
        super().__init__()
        self.fitted = False
        self.auto_fit = auto_fit
        self.persistent = persistent
        self.need_reset = False

    def fit(self, *args, **kwargs) -> None:
        """Abstract method for fitting the transform to the data.

        Parameters
        ----------
        *args : Any
            Any positional arguments.
        **kwargs : Any
            Any keyword arguments.
        """
        self.fitted = True
        if not self.persistent:
            self.need_reset = True

    def transform(self, *args, **kwargs) -> Any:
        """Transforms the data.

        Parameters
        ----------
        *args : Any
            Any positional arguments.
        **kwargs : Any
            Any keyword arguments.

        Returns
        -------
        Any
            The transformed data.
        """
        if not self.fitted:
            if self.auto_fit:
                self.fit(*args, **kwargs)
            else:
                raise RuntimeError(
                    "Transform must be fitted before it can be executed.")
        if not self.persistent and self.need_reset:
            self.fitted = False
            self.need_reset = False
        pass

    def fit_transform(self, *args, **kwargs) -> Any:
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
