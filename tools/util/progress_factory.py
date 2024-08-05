from typing import Dict, Optional
import weakref
from tqdm.auto import tqdm
from uuid import uuid4
import inspect


class ProgressElement(tqdm):

    _tag: Optional[str]
    """Name tag for the progress element."""

    _is_reusable: bool
    """Whether the progress element is reusable."""

    def __init__(self,
                 *args,
                 tag: Optional[str] = None,
                 is_reusable: bool = False,
                 **kwargs):
        self._tag = tag
        self._is_reusable = is_reusable
        if self._tag is None and self._is_reusable:
            self._tag = str(uuid4())
        super().__init__(*args, **kwargs)


def free_items(progress_elements: Dict[str, ProgressElement]):
    """Free up memory by deleting progress elements."""
    for tag, element in progress_elements.items():
        try:
            element.close()
        except Exception as e:
            pass
        del element
    progress_elements.clear()


def get_progress_bar_argument(func: callable) -> Optional[str]:
    """Get whether a function accepts a progress bar argument.

    Parameters
    ----------
    func : callable
        Function to check for progress bar argument.

    Returns
    -------
    Optional[str]
        The name of the argument if it exists, otherwise None.
    """
    params = inspect.signature(func).parameters
    # Check if any argument is named "progress_bar"
    if "progress_bar" in params:
        return "progress_bar"
    # Check if any argument is named "pb"
    if "pb" in params:
        return "pb"
    # Check if has kwargs
    if params.get("kwargs", None) is not None:
        return "progress_bar"
    # Invalid
    return None


def get_progress_factory_argument(func: callable) -> Optional[str]:
    """Get whether a function accepts a progress factory argument.

    Parameters
    ----------
    func : callable
        Function to check for progress factory argument.

    Returns
    -------
    Optional[str]
        The name of the argument if it exists, otherwise None.
    """
    params = inspect.signature(func).parameters
    # Check if any argument is named "progress_factory"
    if "progress_factory" in params:
        return "progress_factory"
    # Check if any argument is named "pf"
    if "pf" in params:
        return "pf"
    # Check if has kwargs
    if params.get("kwargs", None) is not None:
        return "progress_factory"
    # Invalid
    return None


class ProgressFactory:

    _elements: dict[str, ProgressElement]
    """Dictionary of progress elements."""

    def __init__(self):
        self._elements = {}
        self._finalizer = weakref.finalize(self, free_items, self._elements)

    def _get_or_create(self, *args, tag: str, is_reusable: bool, **kwargs) -> Optional[ProgressElement]:
        exists = self._elements.get(tag, None) if is_reusable else None
        if exists is not None:
            # Reset the progress bar
            exists.reset(total=kwargs.get("total", None))
            if "desc" in kwargs:
                exists.set_description(kwargs["desc"])
            exists.refresh()
            return exists
        else:
            # Create a new progress bar
            elem = ProgressElement(
                *args, tag=tag, is_reusable=is_reusable, **kwargs)
            if is_reusable:
                self._elements[tag] = elem
            return elem

    def close(self):
        self._finalizer()

    @property
    def is_closed(self):
        return not self._finalizer.alive

    def tqdm(self,
             *args,
             tag: Optional[str] = None,
             is_reusable: bool = False,
             **kwargs) -> ProgressElement:
        """Return a new or existing progress element.


        Parameters
        ----------
        tag : Optional[str], optional
            Tag which can be used to reuse arbitrary progress bars, by default None
        is_reusable : bool, optional
            If the bar should be reusable, by default False

        Returns
        -------
        ProgressElement
            The progress element
        """
        _kwargs = dict(kwargs)
        if is_reusable:
            _kwargs["leave"] = True
        element = self._get_or_create(
            *args, tag=tag, is_reusable=is_reusable, **_kwargs)
        return element
