from typing import Any, Optional, Type, Union, Set, Dict, List
from types import ModuleType, FunctionType
import importlib

IMPORT_CACHE = {}

ALIAS_TYPE_CACHE: Dict[str, Type] = {}
TYPE_ALIAS_CACHE: Dict[Type, List[str]] = {}

class _NOALIAS:
    pass

NOALIAS = _NOALIAS()

def _get_alias(cls_or_obj: Union[object, Type]) -> Union[str, List[str], _NOALIAS]:
    if not isinstance(cls_or_obj, (Type, FunctionType)):
        cls_or_obj = type(cls_or_obj)
    if hasattr(cls_or_obj, '__type_alias__'):
        alias = cls_or_obj.__type_alias__
        if isinstance(alias, (list, tuple, str, set)):
            if isinstance(alias, str):
                return [alias]
            return list(alias)
        else:
            raise ValueError("Alias must be a string or a list of strings.")
    return NOALIAS

def _register_alias(cls_or_obj: Union[object, Type]) -> None:
    alias = _get_alias(cls_or_obj)
    if alias != NOALIAS and isinstance(alias, str):
        alias = {alias}
    if alias != NOALIAS and alias is not None and any((a in ALIAS_TYPE_CACHE) for a in alias):
        # Check if alias in use
        registered_aliases = ", ".join([a for a in alias if a in ALIAS_TYPE_CACHE])
        used_for = ", ".join([get_type_string(ALIAS_TYPE_CACHE[a]) for a in alias if a in ALIAS_TYPE_CACHE])
        raise ValueError(f"Alias {alias} already used for {used_for}!")
    # Register alias
    TYPE_ALIAS_CACHE[cls_or_obj] = alias
    if alias != NOALIAS:
        for a in alias:
            ALIAS_TYPE_CACHE[a] = cls_or_obj
    return alias


def get_alias(cls_or_obj: Union[object, Type]) -> Optional[List[str]]:
    """Get the alias of a class or object.
    An alias is a string or a list of strings, which can be used to identify a class or object.
    The alias is used for serialization and deserialization of objects.

    The alias can be set by the `__type_alias__` attribute of a class or object.

    Parameters
    ----------
    cls_or_obj : Union[object, Type]
        Class or object to get its alias.

    Returns
    -------
    Optional[List[str]]
        List of alias strings or None if no alias is set.

    Raises
    ------
    ValueError
        If invalid type is passed.
    """
    if not isinstance(cls_or_obj, (Type, FunctionType)):
        cls_or_obj = type(cls_or_obj)
    if not isinstance(cls_or_obj, (Type, FunctionType)):
        raise ValueError("cls_or_obj must be a type or a function.")
    alias = TYPE_ALIAS_CACHE.get(cls_or_obj, None)
    if alias is None:
        alias = _register_alias(cls_or_obj)
    if alias == NOALIAS:
        return None
    return alias

def get_alias_type(alias: str) -> Optional[Type]:
    """Get the type which is registered for the given alias or None.

    Parameters
    ----------
    alias : str
        The alias to get the type for.

    Returns
    -------
    Optional[Type]
        Type which is registered for the alias or None if not found.
    """
    return ALIAS_TYPE_CACHE.get(alias, None)

def dynamic_import(class_or_method: str) -> Any:
    """
    Imports a class, method or module based on a full import string.
    Example: For importing this method the string would be:

    `tools.util.reflection.dynamic_import`  

    So this method dynamically does the following:
    >>> from tools.util.reflection import dynamic_import

    Also modules can be imported with a string like:

    `tools`

    Meaning:

    >>> import tools

    Warning
    ----------
    This method is not safe for user input, as it can import any module or class, 
    which is available in the current environment. Be sure that the input is safe as
    it can lead to security issues and code execution.
    If only safe packages are installed, this method should be safe to use.

    Parameters
    ----------
    class_or_method : str
        The fully qualifing string to import. 
        Working strings can be retrieved with ``class_name`` function.

    Returns
    -------
    Any
        The imported module / type.

    Raises
    ------
    ImportError
        If the import fails.
    """
    value = IMPORT_CACHE.get(class_or_method, None)
    if value is not None:
        return value
    alias_type = get_alias_type(class_or_method)
    # If found alias type, use it, otherwise use the class_or_method and try to import it
    if alias_type is not None:
        return alias_type
    components = class_or_method.split('.')
    if len(components) > 1:
        # Class / type import, trim the class
        module = components[:-1]
    else:
        module = components
    try:
        mod = importlib.import_module(".".join(module))
    except (NameError, ModuleNotFoundError, ImportError) as err:
        raise ImportError(f"Could not import: {class_or_method} \
                          due to an {err.__class__.__name__} does \
                          the Module / Type exists and is it installed?") from err
    if len(components) == 1:
        # Import was a module import only, return it directly
        return mod
    attribute = getattr(mod, components[-1])
    IMPORT_CACHE[class_or_method] = attribute
    return attribute

def get_type_string(cls_or_obj: Union[object, Type]) -> str:
    """
    Returns the type string of the current class or object as string with namespace.

    Parameters
    ----------
    cls_or_obj : Union[object, Type]
        Class or object to get its fully qualified name.

    Returns
    -------
    str
        The fully qualified name.
    """
    if isinstance(cls_or_obj, (Type, FunctionType)): # Types and functions can be imported via their name
        return cls_or_obj.__module__ + '.' + cls_or_obj.__name__
    return cls_or_obj.__class__.__module__ + '.' + cls_or_obj.__class__.__name__

def class_name(cls_or_obj: Union[object, Type], use_alias: bool = True) -> str:
    """
    Returns the class name of the current class or object as string with namespace.

    Parameters
    ----------
    cls_or_obj : Union[object, Type]
        Class or object to get its fully qualified name.

    Returns
    -------
    str
        The fully qualified name.
    """
    if use_alias:
        alias = get_alias(cls_or_obj)
        if alias is not None:
            return alias[0]
    return get_type_string(cls_or_obj)
