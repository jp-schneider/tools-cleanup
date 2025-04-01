from typing import TypeVar, Generic

T = TypeVar('T')


class Container(Generic[T]):
    """Container class to store a single value - for call by reference."""

    _value: T
    """Inner value of the container"""

    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        """Get the value of the container."""
        return self._value

    @value.setter
    def value(self, value: T):
        """Set the value of the container."""
        self._value = value
