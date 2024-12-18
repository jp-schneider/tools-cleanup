from typing import Sized, Generator
from functools import wraps
from typing import Any, Callable


class SizedGenerator(Sized, Generator):
    """Weakly sized generator. This is a generator that has a __len__ method"""

    def __init__(self, generator: Generator[Any, None, None], size: int):
        """Create a new SizedGenerator object.

        Parameters
        ----------
        generator : Generator[Any, None, None]
            The generator to wrap.
        size : int
            The size of the generator.
        """
        self.generator = generator
        self.size = size

    def __iter__(self):
        return self.generator

    def __len__(self):
        return self.size

    def __next__(self):
        return next(self.generator)

    def send(self, value: Any):
        return self.generator.send(value)

    def throw(self, type: Any, value: Any = None, traceback: Any = None):
        return self.generator.throw(type, value, traceback)

    def close(self):
        return self.generator.close()


def sized_generator() -> SizedGenerator:
    """
    Decorator to create a SizedGenerator from a generator.

    A SizedGenerator is a generator that has a __len__ method.
    The decorator can be used on any generator that returns the size as the first element, followed by the actual items.

    The decorator will remove the size from the generator and wrap it in a SizedGenerator so subsequent calls to len() will work.
    And the generator can be used as a normal generator.

    Returns
    -------
    SizedGenerator
        Sized generator object.
    """
    def decorator(function: Callable[[Any], Generator[Any, Any, Any]]) -> Callable[[Any], Generator[Any, Any, Any]]:
        @wraps(function)
        def wrapper(*args, **kwargs):
            gen = function(*args, **kwargs)
            size = next(gen)
            if not isinstance(size, int) or size < 0:
                raise ValueError(
                    "First element of sized generator must return the size as an int which must be >= 0.")
            return SizedGenerator(gen, size)
        return wrapper
    return decorator
