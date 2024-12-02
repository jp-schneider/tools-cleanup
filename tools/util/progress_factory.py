from typing import Dict, Optional, Iterable
import weakref
from tqdm.auto import tqdm
from uuid import uuid4
import inspect
from tqdm.contrib import DummyTqdmFile
import contextlib
import sys

ORIGINAL_STDOUT = None
"""Variable to store the original stdout. When using std_out_err_redirect_tqdm context manager."""

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
        global ORIGINAL_STDOUT
        self._tag = tag
        self._is_reusable = is_reusable
        if self._tag is None and self._is_reusable:
            self._tag = str(uuid4())
        if ORIGINAL_STDOUT is not None:
            kwargs["file"] = ORIGINAL_STDOUT
            kwargs["dynamic_ncols"] = True
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

    def bar(self,
            iterable: Optional[Iterable] = None,
            total: Optional[int] = None,
            desc: Optional[str] = None,
            tag: Optional[str] = None,
            is_reusable: bool = False,
            delay: float = 2.,
            **kwargs) -> ProgressElement:
        """Return a new or existing progress element.

        Alias for tqdm.

        iterable : Optional[Iterable], optional
            Iterable to iterate over, by default None

        total : Optional[int], optional
            Total number of iterations, by default None

        desc : Optional[str], optional
            Description of the progress bar, by default None

        tag : Optional[str], optional
            Tag which can be used to reuse arbitrary progress bars, by default None

        is_reusable : bool, optional
            If the bar should be reusable, by default False

        delay : float, optional
            Delay in seconds before which needs to expire before a bar is displayed, by default 1.

        kwargs : dict
            Additional arguments to pass to tqdm

        Returns
        -------
        ProgressElement
            The progress element
        """
        return self.tqdm(iterable, total=total, desc=desc, tag=tag, is_reusable=is_reusable, delay=delay, **kwargs)


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """Redirect stdout to tqdm.write().

    Make sure to use this context manager with the `with` statement and to use the `as` keyword to store the original stdout within the
    ORIGINAL_STDOUT Global variable.

    with std_out_err_redirect_tqdm() as orig_stdout:
        global ORIGINAL_STDOUT
        ORIGINAL_STDOUT = orig_stdout
        # Your code here
        ...

    Yields
    -------
    Tuple[io.TextIOWrapper, io.TextIOWrapper]
        The original stdout and stderr.

    Raises
    ------
    Exception
        Any exception that occurs during the redirection.

    """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    finally:
        sys.stdout, sys.stderr = orig_out_err

