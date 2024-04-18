from typing import Any
from abc import ABC, abstractmethod


class Transform():
    """Base class for all transforms.
    """
    
    @abstractmethod
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
        pass
    
    
    def __call__(self, *args, **kwargs) -> Any:
        """Calls the transform.

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
        return self.transform(*args, **kwargs)
