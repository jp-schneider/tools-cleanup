import inspect
from typing import Optional
from tools.dataset.base_dataset import BaseDataset
from tools.util.progress_factory import ProgressFactory, get_progress_bar_argument, get_progress_factory_argument
from tqdm.auto import tqdm
from tools.util.reflection import class_name


def accepts_index_argument(func: callable) -> bool:
    """Check if a function accepts an index argument.

    Parameters
    ----------
    func : callable
        Function to check for index argument.

    Returns
    -------
    bool
        True if the function accepts an index argument, otherwise False.
    """
    return "index" in inspect.signature(func).parameters


class DatasetProcessor:
    """Base class for dataset processors."""

    _progress_factory: Optional[ProgressFactory]
    """Progress factory for the dataset processor."""

    _progress_bar: bool
    """If a progress bar should be shown."""

    _dataset: BaseDataset
    """Dataset to process."""

    _func: callable
    """Function to process the dataset."""

    _func_kwargs: dict
    """Keyword arguments for the function."""

    _process_text: str
    """Text to show during processing."""

    def __init__(self,
                 dataset: BaseDataset,
                 func: callable,
                 progress_bar: bool = True,
                 progress_factory: Optional[ProgressFactory] = None,
                 process_text: str = "Processing dataset",
                 **kwargs
                 ) -> None:
        self._progress_bar = progress_bar
        if progress_bar and progress_factory is None:
            progress_factory = ProgressFactory()
        if dataset is None:
            raise ValueError("Dataset cannot be None.")
        if func is None:
            raise ValueError("Function cannot be None.")
        self._progress_factory = progress_factory
        self._dataset = dataset
        self._func = func
        self._process_text = process_text
        self._func_kwargs = kwargs

    def process(self):
        """Process the dataset."""
        it = range(len(self._dataset))
        pbar = None
        if self._progress_bar:
            pbar = self._progress_factory.tqdm(desc=self._process_text, is_reusable=True, total=len(
                it), tag="DatasetProcessor_" + class_name(self._func))

        accepts_index = accepts_index_argument(self._func)
        accepts_progress_bar_str = get_progress_bar_argument(self._func)
        accepts_progress_factory_str = get_progress_factory_argument(
            self._func)

        for i in it:
            args = dict()
            if accepts_index:
                args["index"] = i
            if accepts_progress_bar_str is not None:
                args[accepts_progress_bar_str] = self._progress_bar
            if accepts_progress_factory_str is not None:
                args[accepts_progress_factory_str] = self._progress_factory
            if self._func_kwargs is not None:
                other = self._func_kwargs.copy()
                if accepts_progress_bar_str is not None:
                    other.pop(accepts_progress_bar_str, None)
                if accepts_progress_factory_str is not None:
                    other.pop(accepts_progress_factory_str, None)
                if accepts_index:
                    other.pop("index", None)
                args.update(other)

            # Get i th item
            item = self._dataset[i]
            # Call the function
            self._func(item, **args)
            # Update progress bar
            if pbar is not None:
                pbar.update(1)

    def __call__(self, **kwargs):
        self.process()
