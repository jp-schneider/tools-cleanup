from typing import Dict, Any, List, Tuple, Optional, Set, Union
from enum import Enum
from collections.abc import MutableMapping
from tools.util.reflection import class_name
from tools.util.typing import NOCHANGE, MISSING, CYCLE, _NOCHANGE, _MISSING, _CYCLE


def dict_diff(base: Dict[str, Any], cmp: Dict[str, Any]) -> Union[Dict[str, Any], _NOCHANGE]:
    """Returns the difference between two dictionaries.

    Parameters
    ----------
    base : Dict[str, Any]
        The base dictionary.
    cmp : Dict[str, Any]
        The dictionary which will be compared with the base dictionary.

    Returns
    -------
    Dict[str, Any]
        The difference between the two dictionaries.
    """

    result = dict()
    all_keys = set(base.keys()).union(set(cmp.keys()))
    for k in all_keys:
        if k not in base:
            result[k] = cmp[k]
        elif k not in cmp:
            result[k] = MISSING
        else:
            chg = changes(base[k], cmp[k])
            if chg != NOCHANGE:
                result[k] = chg
    if len(result) == 0:
        return NOCHANGE
    return result


def object_diff(base: Any, cmp: Any) -> Dict[str, Any]:
    """Returns the difference between two objects.

    Parameters
    ----------
    base : Any
        The base object.
    cmp : Any
        The object which will be compared with the base object.

    Returns
    -------
    Dict[str, Any]
        The difference between the two objects.
    """

    result = dict()
    if not hasattr(base, "__dict__") or not hasattr(cmp, "__dict__"):
        raise ValueError("Both objects must have a __dict__ attribute.")
    base_dict = base.__dict__
    cmp_dict = cmp.__dict__
    dd = dict_diff(base_dict, cmp_dict)
    if not isinstance(dd, _NOCHANGE) and dd != NOCHANGE:
        result.update(dd)
    if type(base) != type(cmp):
        result["__class__"] = class_name(cmp)
    return result


def list_diff(base: List[Any], cmp: List[Any]) -> Union[List[Any], _NOCHANGE]:
    """Returns the difference between two lists.

    Parameters
    ----------
    base : List[Any]
        The base list.
    cmp : List[Any]
        The list which will be compared with the base list.

    Returns
    -------
    List[Any]
        The difference between the two lists.
    """
    result = list()
    length = max(len(base), len(cmp))
    for i in range(length):
        if i >= len(cmp):
            result.append(MISSING)
        elif i >= len(base):
            result.append(cmp[i])
        else:
            result.append(changes(base[i], cmp[i]))
    if all([x == NOCHANGE for x in result]):
        return NOCHANGE
    return result


def tuple_diff(base: tuple, cmp: tuple) -> Union[tuple, _NOCHANGE]:
    """Returns the difference between two tuples.

    Parameters
    ----------
    base : tuple
        The base tuple.
    cmp : tuple
        The tuple which will be compared with the base tuple.

    Returns
    -------
    tuple
        The difference between the two tuples.
    """
    result = list_diff(list(base), list(cmp))
    if isinstance(result, _NOCHANGE) or result == NOCHANGE:
        return NOCHANGE
    return tuple(result)


def changes(base: Any, cmp: Any) -> Any:
    """Returns the changed items of cmp object wrt base.

    Parameters
    ----------
    base : Any
        The base dictionary.
    cmp : Any
        The dictionary which will be compared with the base dictionary.

    Returns
    -------
    Any
        The difference between the two objects. Or NOCHANGE if there are equal
    """

    result = None
    if (base is None and cmp is not None) or (base is not None and cmp is None):
        result = cmp
    if issubclass(type(base), type(cmp)):
        if isinstance(base, dict):
            result = dict_diff(base, cmp)
        elif isinstance(base, Enum):
            if base != cmp:
                result = cmp
            else:
                result = NOCHANGE
        elif isinstance(base, list):
            result = list_diff(base, cmp)
        elif isinstance(base, tuple):
            result = tuple_diff(base, cmp)
        elif hasattr(base, "__dict__") and hasattr(cmp, "__dict__"):
            result = object_diff(base, cmp)
        else:
            if base != cmp:
                result = cmp
            else:
                result = NOCHANGE
    elif isinstance(base, object) and isinstance(cmp, object) and hasattr(base, "__dict__") and hasattr(cmp, "__dict__"):
        result = object_diff(base, cmp)
    else:
        result = cmp
    return result


def flatten(
    dictionary: Dict[str, Any],
    separator: str = '_',
    prefix: str = '',
    keep_empty: bool = False,
) -> Dict[str, Any]:
    """Flattens a dictionary.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to flatten.
    separator : str, optional
        The seperator between nested items, by default '_'
    prefix : str, optional
        Prefix, this is an internal value but can also be used to set a prefix on each entry, by default ''
    keep_empty : bool, optional
        If empty entries should be kept, by default False

    Returns
    -------
    Dict[str, Any]
        The flattend dictionary.
    """
    items = []
    if len(dictionary) == 0 and keep_empty:
        items.append((prefix, None))
    for key, value in dictionary.items():
        new_key = prefix + separator + key if prefix else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, separator=separator,
                         prefix=new_key, keep_empty=keep_empty).items())  # type: ignore
        else:
            items.append((new_key, value))
    return dict(items)


def filter(dictionary: Dict[str, Any], allowed_keys: Dict[str, Any]) -> Dict[str, Any]:
    """Filters a dictionary by allowed keys. In A nested way.

    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to filter.
    allowed_keys : Dict[str, Any]
        The allowed keys.

    Returns
    -------
    Dict[str, Any]
        The filtered dictionary.
    """
    result = dict()
    for key, value in dictionary.items():
        if key in allowed_keys:
            if isinstance(value, MutableMapping):
                # If child is a dict, filter it too but only if there are child keys
                if len(allowed_keys[key]) > 0:
                    result[key] = filter(value, allowed_keys[key])
                else:
                    result[key] = value
            else:
                result[key] = value
    return result


def nested_keys(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Returns all nested keys of a dictionary, whereby the value is a dictionary containing subkeys or is empty if there are no subkeys."""
    result = dict()
    for key, value in dictionary.items():
        if isinstance(value, MutableMapping):
            result[key] = nested_keys(value)
        else:
            result[key] = dict()
    return result


def combine_nested_keys(dictionaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combines nested keys of multiple dictionaries. On all hierarchies of the dict.
    The result contains all keys which exists in at least one dictionary.
    Accordingly its the union of all keys.

    Parameters
    ----------
    dictionaries : List[Dict[str, Any]]
        List of dictionaries with nested keys to combine.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary of all keys.
    """
    result = dict()
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key not in result:
                result[key] = value
            else:
                if isinstance(value, MutableMapping):
                    result[key] = combine_nested_keys([result[key], value])
                else:
                    result[key] = value
    return result


def compute_diff_string(
    obj: Dict[str, Any],
    default_values: Dict[str, Any] = None,
    key_alias: Optional[Dict[str, str]] = None,
    value_alias: Optional[Dict[str, str]] = None,
    value_separator: str = ":",
    item_separator: str = "_",
    nested_keys: Optional[Dict[str, Any]] = None,
    specified_markers: bool = True
) -> str:
    """Computes a diff string based on the given object and default values.
    This string will show the declaration differences of the current obj w.r.t the default values.

    Parameters
    ----------
    default_values : Dict[str, Any], optional
        Default values to compare against, by default None
    key_alias : Optional[Dict[str, str]], optional
        Alias name for the key. This should be the mapping of the old key name to the new one.
        As the default values can be a nested dict, this nesting can be adressed by concatenating the keys with the separator '__'.
        So the key_alias itself is a flat dict, by default None
    value_alias : Optional[Dict[str, str]], optional
        This can be used to alter the value of the differing key by simple substitution.
        Like 'False': 'No', by default None
        , by default None
    value_separator : str, optional
        How the value should be seperated from the key in the result string, by default ":"
    item_separator : str, optional
        How individual items should be separated, by default "_"
    nested_keys : Optional[Dict[str, Any]], optional
        If only a subset of the keys should be used for the diff, by default None
    specified_markers : bool, optional
        If the diff string should contain markers for specified values and changed values, by default True
    Returns
    -------
    str
        The diffstring with all altered parameters and their values.
    """
    from tools.util.diff import nested_keys as get_nested_keys

    def _process_value(v: Any, default_value: Optional[Any],
                       value_alias: Optional[Dict[str, str]] = None
                       ) -> Any:
        if isinstance(v, dict) or isinstance(default_value, dict):
            ret = dict()
            keys = []
            if v is not None:
                keys = list(v.keys())
            if default_value is not None:
                keys += list(default_value.keys())
            keys = set(keys)
            for k in keys:
                _v = v.get(k, None) if v is not None else None
                _def = default_value.get(
                    k, None) if default_value is not None else None
                name = k
                if _v is None and _def is None:
                    continue
                ret[name] = _process_value(_v, _def, value_alias=value_alias)
            return ret
        else:
            has_changes = None
            if v is not None:
                if default_value is not None:
                    has_changes = not (v == default_value)
                v_str = str(v)
                if value_alias is not None:
                    v_str = value_alias.get(v_str, v_str)
                if has_changes is not None:
                    v_str = (v_str + ("*" if specified_markers else "")
                             ) if has_changes else v_str
                else:
                    v_str = v_str + ("?" if specified_markers else "")
                return v_str
            elif default_value is not None:
                v_str = str(default_value)
                if value_alias is not None:
                    v_str = value_alias.get(v_str, v_str)
                if specified_markers:
                    v_str += "-"
                return v_str
            else:
                raise NotImplementedError(
                    "Either v, or default should not be None, there is an error.")

    own = obj
    if nested_keys is not None:
        own = filter(own, nested_keys)
        default_values = filter(default_values, nested_keys)
    else:
        altered = changes(default_values, own)
        ntk = get_nested_keys(altered)
        own = filter(own, ntk)
        default_values = filter(default_values, ntk)

    deep = _process_value(own, default_values, value_alias)
    flattened_dict = flatten(deep, separator="__")
    if key_alias is not None:
        ret = {}
        for k, v in flattened_dict.items():
            new_key = key_alias.get(k, k)
            ret[new_key] = v
        flattened_dict = ret
    comp = [k + value_separator + v for k, v in flattened_dict.items()]
    return item_separator.join(comp)
