
from typing import TypeVar, Generic, Optional, List, Union, Callable
from tools.util.typing import NOTSET, _NOTSET
from functools import partial
T = TypeVar('T')


class LazyItem(Generic[T]):
    """A wrapper around a callable that lazily evaluates its value.
    The callable is only executed when the value is accessed for the first time.

    Parameters
    ----------
    func : callable[[], T]
        A function that returns the value.

    value : Union[T, _NOTSET], optional
        An optional precomputed value. If provided, the function will not be called.
        By default, NOTSET.

    """

    def __init__(self, func: Callable[[], T], value: Union[T, _NOTSET] = NOTSET) -> None:
        self._func = func
        self._value = value
        self._evaluated = value != NOTSET
        self._apply_func: List[Callable[[T], None]] = []

    def _retrieve(self) -> T:
        v = self._func()
        for func in self._apply_func:
            func(v)
        self._apply_func.clear()
        return v

    @property
    def value(self) -> T:
        """Gets the value, evaluating the function if necessary."""
        if not self._evaluated:
            self._value = self._retrieve()
            self._evaluated = True
        return self._value

    def apply(self, func: Callable[[T], None]) -> None:
        """Applies a function to the value when it is evaluated.

        Parameters
        ----------
        func : callable[[T], None]
            A function that takes the value and returns None.
        """
        if self._evaluated:
            func(self._value)
        else:
            self._apply_func.append(func)


class LazyList(Generic[T]):
    """A list that lazily evaluates its items.
    Items are only computed when accessed, allowing for efficient use of resources.

    Parameters
    ----------
    func : callable[[int], T]
        A function that takes an index and returns the item at that index.
    length : int
        The length of the list.
    """

    def __init__(self, func: Callable[[int], T], length: int) -> None:
        self._func = func
        self._length = length
        self._values: List[LazyItem[T]] = [
            LazyItem(partial(func, i)) for i in range(length)]
        self._get_lazy_item = False

    @property
    def get_lazy_item(self) -> bool:
        """If True, accessing an item returns the LazyItem instead of the value."""
        return self._get_lazy_item

    @get_lazy_item.setter
    def set_lazy_item(self, value: bool) -> None:
        """Sets whether to return the LazyItem instead of the value."""
        self._get_lazy_item = value

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> T:
        if not self._get_lazy_item:
            return self._values[index].value
        else:
            return self._values[index]

    def __iter__(self):
        for item in self._values:
            yield item.value

    def __repr__(self) -> str:
        return f"LazyList(length={self._length})"

    def __setitem__(self, index: int, value: T) -> None:
        self._values[index] = LazyItem(lambda: value, value)

    def apply(self, func: Callable[[T], None]) -> None:
        """Applies a function to all items when they are evaluated.

        Will apply immediately to already evaluated items, and store the function to apply to future items uppon retrieval.

        Parameters
        ----------
        func : callable[[T], None]
            A function that takes an item and returns None.
        """
        for item in self._values:
            item.apply(func)

    def to_list(self) -> List[T]:
        return [item.value for item in self._values]

    def extend(self, values: List[T]) -> None:
        for value in values:
            self._values.append(LazyItem(lambda: value, value))
        self._length += len(values)

    def append(self, value: T) -> None:
        self._values.append(LazyItem(lambda: value, value))
        self._length += 1

    def insert(self, index: int, value: T) -> None:
        self._values.insert(index, LazyItem(lambda: value, value))
        self._length += 1

    def clear(self) -> None:
        self._values.clear()
        self._length = 0

    def pop(self, index: int = -1) -> T:
        item = self._values.pop(index)
        self._length -= 1
        return item.value

    def remove(self, value: T) -> None:
        for i, item in enumerate(self._values):
            if item.value == value:
                self._values.pop(i)
                self._length -= 1
                return
        raise ValueError("Value not found in LazyList")
