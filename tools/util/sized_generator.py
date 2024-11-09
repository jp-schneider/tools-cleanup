from collections.abc import Sized, Generator
from typing import Any


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
