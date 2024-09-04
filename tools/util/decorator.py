from functools import wraps
from typing import Any, Callable


def placeholder(
    *decorator_args: Any, **decorator_kwargs: Any
):
    """Placeholder decorator.
    This decorator can be used as a placeholder for a decorator that is not yet implemented or not used.
    """
    # type: ignore
    def decorator(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapper
    return decorator
