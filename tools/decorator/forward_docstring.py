from functools import wraps
from typing import Any, Callable, Union
from tools.util.reflection import dynamic_import

# def forward_docstring(
#     docstring_source: Union[Any, str]
# ):
#     patched_desc = None
#     if isinstance(docstring_source, str):
#         imp = dynamic_import(docstring_source)
#         patched_desc = imp.__doc__
#     else:
#         patched_desc = docstring_source.__doc__
#     def decorator(function: Callable[[Any], Any]) -> Callable[[Any], Any]:
#         function.__doc__ = patched_desc
#         @wraps(function)
#         def wrapper(*args, **kwargs):
#             return function(*args, **kwargs)
#         return wrapper
#     return decorator
