import inspect
from typing import Any, Callable, Optional, Tuple, Type, Union, Set, Dict, List
from types import ModuleType, FunctionType
import importlib

from tools.util.typing import NOTSET, PATHNONE, MISSING
from collections.abc import Sequence

IMPORT_CACHE = {}

ALIAS_TYPE_CACHE: Dict[str, Type] = {}
LEGACY_TYPE_CACHE: Dict[str, Type] = {}
LEGACY_MODULE_CACHE: Dict[str, ModuleType] = {}
TYPE_ALIAS_CACHE: Dict[Type, List[str]] = {}


class _NOALIAS:
    pass


NOALIAS = _NOALIAS()

WARN_ON_LEGACY_IMPORT = True

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
    # Make sure rule registry is loaded
    alias = _get_alias(cls_or_obj)
    if alias != NOALIAS and isinstance(alias, str):
        alias = {alias}
    if alias != NOALIAS and alias is not None and any((a in ALIAS_TYPE_CACHE) for a in alias):
        # Check if alias in use
        registered_aliases = ", ".join(
            [a for a in alias if a in ALIAS_TYPE_CACHE])
        used_for = ", ".join([get_type_string(ALIAS_TYPE_CACHE[a])
                             for a in alias if a in ALIAS_TYPE_CACHE])
        raise ValueError(
            f"Alias {alias} already used for {used_for} while trying to register {class_name(cls_or_obj)}!")
    # Register alias
    TYPE_ALIAS_CACHE[cls_or_obj] = alias
    if alias != NOALIAS:
        for a in alias:
            ALIAS_TYPE_CACHE[a] = cls_or_obj
    return alias

def _register_legacy_import(legacy_import: str, current_import: str) -> None:
    """
    Registers a legacy import string to the current import string.

    Adds an alias import string for the current import string, such that after renaming or moving a class,
    the legacy import string can still be used to import the class.

    Parameters
    ----------
    legacy_import : str
        The legacy import string.

    current_import : str
        The current import string.
    """
    cls_or_obj = dynamic_import(current_import)
    # Check if cls is type or module
    if isinstance(cls_or_obj, (ModuleType)):
        # Type is module
        # Check if already registered
        if legacy_import in LEGACY_MODULE_CACHE:
            existing_module = LEGACY_MODULE_CACHE[legacy_import]
            if existing_module != cls_or_obj:
                raise ValueError(
                    f"Legacy import {legacy_import} already registered for {existing_module.__name__} while trying to register {cls_or_obj.__name__}!")
            # Already registered, do nothing
            return
        # Register legacy import as alias
        LEGACY_MODULE_CACHE[legacy_import] = cls_or_obj
    else:
        # Type is class or function  
        # Check if already registered
        if legacy_import in LEGACY_TYPE_CACHE:
            existing_cls_or_obj = LEGACY_TYPE_CACHE[legacy_import]
            if existing_cls_or_obj != cls_or_obj:
                raise ValueError(
                    f"Legacy import {legacy_import} already registered for {class_name(existing_cls_or_obj)} while trying to register {class_name(cls_or_obj)}!")
            # Already registered, do nothing
            return
        # Register legacy import as alias
        LEGACY_TYPE_CACHE[legacy_import] = cls_or_obj

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

def register_type():
    """
    Register a type for serialization and deserialization.
    """
    def decorator(_type: Type) -> Type:
        _ = get_alias(_type)
        return _type
    return decorator


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
    from tools.serialization.rules.json_serialization_rule_registry import JsonSerializationRuleRegistry
    _ = JsonSerializationRuleRegistry.instance()
    val = ALIAS_TYPE_CACHE.get(alias, None)
    # Check legacy imports
    if val is None:
        val = LEGACY_TYPE_CACHE.get(alias, None)
        if val is not None:
            global WARN_ON_LEGACY_IMPORT
            if WARN_ON_LEGACY_IMPORT:
                # Warn about legacy import usage
                from tools.util.format import get_frame_summary
                from tools.logger.logging import logger
                frame_summary = get_frame_summary(2)
                logger.warning(f"Legacy import string '{alias}' used in {frame_summary.filename}:{frame_summary.lineno} for type '{class_name(val, use_alias=False)}'. Please update to the new import string.")
    return val

def get_legacy_replace_module(alias: str, depth: int = 0, full_alias: Optional[str] = None) -> str:
    """Get the module which is was regstered to repalce the given alias.

    Parameters
    ----------
    alias : str
        The alias to get the module for.

    Returns
    -------
    Optional[ModuleType]
        Module which is registered for the alias or None if not found.
    """
    if full_alias is None:
        full_alias = alias
    val = LEGACY_MODULE_CACHE.get(alias, None)
    if val is not None:
        global WARN_ON_LEGACY_IMPORT
        if WARN_ON_LEGACY_IMPORT:
            # Warn about legacy import usage
            from tools.util.format import get_frame_summary
            from tools.logger.logging import logger
            frame_summary = get_frame_summary(2 + depth)
            logger.warning(f"Legacy import string '{alias}' used in {frame_summary.filename}:{frame_summary.lineno} for module '{val.__name__}'. Please update to the new import string. Full import alias was: '{full_alias}'.")
        alias = class_name(val, use_alias=False)
    else:
        pass
    splits = alias.split('.')
    if len(splits) > 1:
        # Try parent modules
        parent_alias = '.'.join(splits[:-1])
        parent_alias = get_legacy_replace_module(parent_alias, depth + 1, full_alias=full_alias)
        current_alias = splits[-1]
        return f"{parent_alias}.{current_alias}"
    else:
        return alias

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

    success = False
    check_with_legacy = False

    while not success or (not success and check_with_legacy):
        try:
            mod = importlib.import_module(".".join(module))
            success = True
        except (NameError, ModuleNotFoundError, ImportError) as err:
            if check_with_legacy:
                # Try legacy import replacement here
                raise ImportError(f"Could not import: {class_or_method} \
                                due to an {err.__class__.__name__} does \
                                the Module / Type exists and is it installed?") from err
            else:
                check_with_legacy = True
                legacy_module_str = get_legacy_replace_module(".".join(module) + (("." +components[-1]) if len(components) > len(module) else ""), depth=0)
                new_components = legacy_module_str.split('.')
                if len(new_components) > 1:
                    new_modules = new_components[:-1]
                else:
                    new_modules = new_components
                if ".".join(new_modules) == ".".join(module):
                    # No change in legacy module, stop here
                    raise ImportError(f"Could not import: {class_or_method} \
                                    due to an {err.__class__.__name__} does \
                                    the Module / Type exists and is it installed?") from err
                else:
                    # Try with replaced legacy module
                    module = new_modules
                    components = new_components

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
    if isinstance(cls_or_obj, (Type, FunctionType)):  # Types and functions can be imported via their name
        return cls_or_obj.__module__ + '.' + cls_or_obj.__name__
    elif isinstance(cls_or_obj, ModuleType):  # Modules have no class name
        return cls_or_obj.__name__
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


def propagate_init_kwargs(cls_or_obj: Type, type_in_mro: Type, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Decides which kwarguements can be propageted to the next type in the MRO with the init method.

    Parameters
    ----------
    cls_or_obj : Type
        The actual class or object to propagate the arguments from.
    type_in_mro : Type
        The current type in the MRO to which wants to decide which arguments to propagate.
    kwargs : Dict[str, Any]
        The kwargs to propagate.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple of two dictionaries, the first dictionary contains the accepted parameters, the second dictionary contains the left parameters / invalid parameters.
    """
    if len(kwargs) == 0:
        # No arguments to propagate
        return kwargs, dict()
    if not isinstance(cls_or_obj, Type):
        cls_or_obj = type(cls_or_obj)
    if not isinstance(cls_or_obj, Type):
        raise ValueError("cls_or_obj must be a type or instance.")
    if not isinstance(type_in_mro, Type):
        raise ValueError("type_in_mro must be a type.")
    # Get MRO of the class
    mro = cls_or_obj.mro()
    if type_in_mro not in mro:
        raise ValueError(f"type_in_mro must be in the MRO of cls_or_obj.")
    # Get the index of the type in the MRO
    index = mro.index(type_in_mro)
    # Get the init method of the type after the type_in_mro
    mro_type = mro[index + 1] if index + 1 < len(mro) else None
    if mro_type is None or mro_type == object:
        # Return empty kwargs to propagate no arguments as its the last type in mro
        return dict(), kwargs

    # Get the signature of the init method
    sig = inspect.signature(mro_type.__init__)
    # Get the parameters of the init method
    param_dict = sig.parameters
    params = param_dict.values()
    param_keys = param_dict.keys()
    # If params is empty, return empty kwargs to propagate no arguments
    if len(params) == 0:
        return dict(), kwargs

    accepted_params = dict()
    left_params = dict(kwargs)

    has_kwargs = False
    # Check if kwargs are accepted, if so, all parameters are accepted
    has_kwargs = list(params)[-1].kind == inspect.Parameter.VAR_KEYWORD
    if has_kwargs:
        return kwargs, dict()

     # Accepted parameters can be of the following kinds
    allowed_kinds = {inspect.Parameter.KEYWORD_ONLY,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_KEYWORD}

    # Decide where to loop over, kwargs or params based on the length
    if len(params) > len(kwargs):
        # Loop over the parameters
        for param in params:
            if param.kind not in allowed_kinds:
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_kwargs = True
                break
            if param.name in left_params:
                accepted_params[param.name] = left_params.pop(param.name)
    else:
        # Loop over the kwargs
        for param_name, value in kwargs.items():
            if param_name in param_keys:
                if param_dict[param_name].kind not in allowed_kinds:
                    continue
                if param_dict[param_name].kind == inspect.Parameter.VAR_KEYWORD:
                    has_kwargs = True
                    break
                accepted_params[param_name] = value
                # Remove the parameter from the left params
                left_params.pop(param_name)
    if has_kwargs:
        return kwargs, dict()
    return accepted_params, left_params


def _get_attribute(obj: Any, path: str, default: Any = NOTSET) -> Any:
    if isinstance(obj, dict):
        # Check if path is a key in the dictionary
        return obj.get(path, default)
    if path.isnumeric():
        # Check if path is a number, if so check if obj is a Sequnece
        if not isinstance(obj, Sequence):
            raise ValueError(
                f"Object {obj} is not a sequence, but path {path} is a number.")
        index = int(path)
        if index >= len(obj):
            return NOTSET
        return obj[index]
    return getattr(obj, path, default)


def _set_attribute(obj: Any, path: str, value: Any) -> Any:
    if isinstance(obj, dict):
        # Check if path is a key in the dictionary
        if value == NOTSET:
            oval = obj.pop(path, None)
            return oval
        else:
            old_value = obj.get(path, NOTSET)
            obj[path] = value
            return old_value
    if path.isnumeric():
        # Check if path is a number, if so check if obj is a Sequnece
        if not isinstance(obj, Sequence):
            raise ValueError(
                f"Object {obj} is not a sequence, but path {path} is a number.")
        index = int(path)
        old_value = NOTSET
        if index >= len(obj) and value != NOTSET:
            raise ValueError(
                f"Index {index} is out of bounds for object {obj}.")
        if value == NOTSET:
            if index < len(obj):
                # Remove the value from the sequence
                old_value = obj.pop(index)
            return old_value
        else:
            old_value = obj[index] if index < len(obj) else NOTSET
        obj[index] = value
        return old_value
    old_value = getattr(obj, path, NOTSET)
    setattr(obj, path, value)
    return old_value


def _get_nested_value(obj: Any, path: str, default: Any = NOTSET) -> Any:
    if obj is None and len(path) > 0:
        return PATHNONE
    if '.' in path:
        path, rest = path.split('.', 1)
        return _get_nested_value(_get_attribute(obj, path, default), rest, default)
    else:
        return _get_attribute(obj, path, default)


def _set_nested_value(obj: Any, path: str, value: Any) -> Any:
    if value == PATHNONE:
        return
    if '.' in path:
        path, rest = path.split('.', 1)
        current_obj = _get_attribute(obj, path, MISSING)
        if current_obj == MISSING:
            raise ValueError(f"Object {obj} does not have attribute {path}.")
        return _set_nested_value(current_obj, rest, value)
    else:
        return _set_attribute(obj, path, value)


def set_nested_value(obj: Any, path: str, value: Any) -> Any:
    """Set a nested value in an object by a path / chain of attributes to the value.

    Can also be used to delete a value by setting it to NOTSET.
    Supports also lists, and aribitrary sequences which can be indexed by a number.
    E.g. "a.b.c.0.d.e".

    Parameters
    ----------
    obj : Any
        The object to set the value in.
    path : str
        The path to the value, e.g. "a.b.c".
    value : Any
        The value to set.
        If the value is NOTSET, the attribute is deleted.

    Returns
    -------
    Any
        The old value at the path or NOTSET if the path did not exist.
    """
    return _set_nested_value(obj, path, value)


def get_nested_value(obj: Any, path: str, default: Any = NOTSET) -> Any:
    """Get a nested value in an object by a path / chain of attributes to the value.

    Can be used to get values from dictionaries, lists, and aribitrary sequences which can be indexed by a number.
    E.g. "a.b.c.0.d.e".

    Parameters
    ----------
    obj : Any
        The object to get the value from.
    path : str
        The path to the value, e.g. "a.b.c".

    default : Any, optional
        The default value if the path does not exist, by default NOTSET.

    Returns
    -------
    Any
        The value at the path or the default value.

        Returns PATHNONE if the path does not completely exist.
        Returns NOTSET if the last attribute does not exist.
    """
    return _get_nested_value(obj, path, default)


def check_package(package_name: str, silent: bool = True) -> Optional[ModuleType]:
    """Checks if a package is installed.

    Parameters
    ----------
    package_name : str
        The name of the package to check.

    silent : bool, optional
        If set, no exception is raised if the package is not installed, by default True
        If false, an ImportError or is raised if the package is not installed.

    Returns
    -------
    Optional[ModuleType]
        The module if the package is installed, otherwise None.

    """
    try:
        module = importlib.import_module(package_name)
        return module
    except (ImportError, ModuleNotFoundError) as e:
        if not silent:
            raise e
        return None


def check_fnc_supported_args(
    func_or_method: Callable[[Any], Any],
    args: Dict[str, Any],
    kwargs_as_supported: bool = True,
    **kwargs: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Checks if the arguments are supported by the function / method.

    Parameters
    ----------
    func_or_method : Callable[[Any], Any]
        The function or method to check the arguments for.

    args : Dict[str, Any]
        The arguments to check.

    kwargs_as_supported : bool, optional
        If kwargs are supported, by default True
        When kwargs are supported, all arguments are accepted.
        E.g. if the function has a **[name] parameter.

    kwargs : Any
        Additional keyword arguments to pass to the function. Treated the same as args.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple of two dictionaries, the first dictionary contains the accepted parameters, the second dictionary contains the left parameters / invalid parameters.
    """
    successfull = dict()
    left = dict(args)
    left.update(**kwargs)
    if hasattr(func_or_method, '__qualname__') and func_or_method.__qualname__ == 'object.__init__':
        # Top level object init method, return all arguments, its the base class, and does not have any parameters
        return dict(), left
    pparams = dict(inspect.signature(func_or_method).parameters)
    if len(pparams) == 0:
        return dict(), left
    if kwargs_as_supported:
        # Check if kwargs are supported
        if any([p.kind == inspect.Parameter.VAR_KEYWORD for p in pparams.values()]):
            return left, dict()
    if len(pparams) < len(left):
        # Loop over the parameters
        for param in pparams.values():
            if param.name in dict(left):
                successfull[param.name] = left.pop(param.name)
    else:
        # Loop over the kwargs
        for param_name, value in dict(left).items():
            if param_name in pparams:
                successfull[param_name] = value
                # Remove the parameter from the left params
                left.pop(param_name)
    return successfull, left
