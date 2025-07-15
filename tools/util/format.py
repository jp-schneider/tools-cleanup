import copy
from enum import Enum
import math
import os
import re
from datetime import timedelta
from string import Template
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union
from tools.logger.logging import logger
import inspect
from pandas import Series
from tools.util.typing import VEC_TYPE
from tools.error import ArgumentNoneError
from tools.error.argument_none_type_suggestion_error import ArgumentNoneTypeSuggestionError
from tools.util.path_tools import format_os_independent
from tools.util.typing import DEFAULT, MISSING
from tools.util.reflection import dynamic_import, get_nested_value, set_nested_value
from traceback import FrameSummary, extract_stack
from types import FrameType, TracebackType
import pandas as pd
from dataclasses import dataclass, field

CAMEL_SEPERATOR_PATTERN = re.compile(
    r'((?<!^)(?<!_))((?=[A-Z][a-z])|((?<=[a-z])(?=[A-Z])))')
UPPER_SNAKE_PATTERN = re.compile(r'^([A-Z]+_?)*([A-Z]+)$')
UPPER_PATTERN = re.compile(r'^([A-Z]+)$')

REGEX_ISO_8601_PATTERN = r'^(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?(Z|[+-](?:2[0-3]|[01][0-9]):[0-5][0-9])?$'
REGEX_ISO_COMPILED = re.compile(REGEX_ISO_8601_PATTERN)


@dataclass
class FormatVariable():
    """Dataclass for a format variable."""

    variable: str
    """The variable name."""

    value: str
    """The unformatted value of the variable."""

    formatter: Optional[str] = field(default=None)
    """The formatter for the variable."""

    localizer: Optional[str] = field(default=None)
    """The localizer for the variable."""


def to_snake_case(input: str) -> str:
    """Converts a upper snake case, or camel case pattern to lower snake case.

    Parameters
    ----------
    input : str
        The input string to convert.

    Returns
    -------
    str
        The converted string

    Raises
    ------
    ArgumentNoneError
        If input is None.
    ValueError
        If input is not a string.
    """
    if input is None:
        raise ArgumentNoneError("input")
    if not isinstance(input, str):
        raise ValueError(
            f"Type of input should be string but is: {type(input).__name__}")
    if ('_' not in input or not UPPER_SNAKE_PATTERN.match(input)) and not UPPER_PATTERN.match(input):
        return CAMEL_SEPERATOR_PATTERN.sub('_', input).lower()
    else:
        return input.lower()


def snake_to_upper_camel(input: str, sep: str = "") -> str:
    """Converts a snake '_' pattern to a upper camel case pattern.

    Parameters
    ----------
    input : str
        The input string to convert.
    sep : str, optional
        Seperator for individual words., by default ""

    Returns
    -------
    str
        The altered string.
    """
    words = [x.capitalize() for x in input.split("_")]
    return sep.join(words)


def str_to_bool(value: str) -> bool:
    """Converts a string to a boolean.

    Parameters
    ----------
    value : str
        The value to convert.

    Returns
    -------
    bool
        The converted value.

    Raises
    ------
    ValueError
        If the value is not a valid boolean string.
    """
    if value.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif value.lower() in ("no", "false", "f", "0", "n", ""):
        return False
    else:
        raise ValueError(f"Invalid boolean string: {value}")


class TimeDeltaTemplate(Template):
    """Class for formating timedelta with strftime like syntax"""
    delimiter = "%"


def strfdelta(delta: timedelta, format: str = "%D days %H:%M:%S.%f") -> str:
    """Formats the timedelta.
    Supported substitudes:

    %D - Days
    %H - Hours
    %M - Minutes
    %S - Seconds
    %f - Microseconds

    Parameters
    ----------
    delta : timedelta
        The timedelta to format
    format : str
        The format string.

    Returns
    -------
    str
        The formatted string.
    """
    d = {"D": delta.days}
    d["H"], rem = divmod(delta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    d["f"] = delta.microseconds
    t = TimeDeltaTemplate(format)
    d = {k: f'{v:02d}' for k, v in d.items()}
    return t.substitute(**d)


def preceding_zeros_format(max_value: int) -> str:
    """Determines the number of preceding zeros for a number.

    Parameters
    ----------
    max_value : int
        The maximum value.

    Returns
    -------
    str
        The format string.
    """
    return f"{{:0{len(str(max_value))}d}}"


def destinctive_number_float_format(values: Series,
                                    max_decimals: int = 10,
                                    use_scientific_format: Optional[bool] = None,
                                    distinctive_digits_for_scientific_format: int = 5
                                    ) -> str:
    """Evaluates based on the number of destinctive digits the best format for a float.

    Parameters
    ----------
    values : Series
        The values to evaluate.
    max_decimals : int, optional
        The upper limit for decimal digits, by default 10
    use_scientific_format : Optional[bool], optional
        If the scientific format should be used or not, by default decision will be based
        on how many distinctive decimal places are needed., by default None
    distinctive_digits_for_scientific_format : int, optional
        The number of destinctive digits for the scientific format, by default 4
        If a number needs more or equal destinctive digits to judge wether there are different
         it will be formatted in scientific format.
    Returns
    -------
    str
        The string format.
    """
    destinctive = False
    values = copy.deepcopy(values)

    # Ignore Nan

    values = values.dropna()

    def _count_leading_zeros(value: float):
        if abs(value) >= 1 or value == 0:
            return 0
        post = str(value).split('.')[1]
        i = 0
        while post[i] == '0':
            i += 1
        return i

    leading_zeros = min(values.apply(lambda x: _count_leading_zeros(x)))
    max_leading_zeros = max(values.apply(lambda x: _count_leading_zeros(x)))

    unshifted = copy.deepcopy(values)
    unshifted_i = 0

    _exp = dict()

    # Move large numbers behind the comma
    for i, _v in values.items():
        exp = 0
        num = _v
        if num == 0:
            exp = 1
            num = 0.
            values[i] = num
            _exp[i] = exp
            continue
        while abs(num) >= 10:
            num = num / 10
            exp += 1
        while abs(num) < 1:
            num = num * 10
            exp -= 1
        values[i] = num
        _exp[i] = exp

    i = 0
    float_destinctive = False
    while (not destinctive or not float_destinctive) and i < max_decimals:
        if not destinctive:
            _v = values * 10**i
            _v = _v.apply(math.floor)
            _cmp = Series({k: val * (10 ** _exp[k]) for k, val in _v.items()})
            if len(_cmp.unique()) == len(values.unique()):
                destinctive = True
            else:
                i += 1
        if len(set([round(x, unshifted_i) for x in unshifted])) != len(unshifted.unique()):
            unshifted_i += 1
        else:
            float_destinctive = True

    _exp_max = max(list(_exp.values()))
    _exp_min = min(list(_exp.values()))
    _exp_mean = sum(list(_exp.values())) / len(_exp.values())

    if ((_exp_max - _exp_min) >= 3 or _exp_mean < -2) and use_scientific_format is None:
        use_scientific_format = True
    else:
        use_scientific_format = False
    num_destinctive_digits = i if use_scientific_format else unshifted_i
    return f"{{:.{num_destinctive_digits}{'e' if use_scientific_format else 'f'}}}"


def latex_postprocessor(text: str,
                        replace_underline: bool = True,
                        replace_bfseries: bool = True,
                        replace_text_decoration_underline: bool = True,
                        replace_booktabs_rules: bool = True
                        ) -> str:
    """Postprocesses a latex string.
    Can applied to pandas to latex commands to fix incorrect latex syntax.

    Parameters
    ----------
    text : str
        The string which shoud be altered.
    replace_underline : bool, optional
        If underlines should be replaced. Will replace all and does not care for math mode., by default True
    replace_bfseries : bool, optional
        Replace a bfseries which should be a textbf, by default True

    Returns
    -------
    str
        The processed string.
    """
    # Pattern
    UNDERSCORE_IN_TEXT = r"(?<=([A-z0-9\_]))\_(?=[A-z0-9\_])"
    BF_SERIES = r"(\\bfseries)( )(?P<text>[A-z0-9.\-\_\+]+)( )"
    TEXT_DECO_UNDERLINE = r"(\\text-decorationunderline)( )(?P<text>[A-z0-9.\-\_\+]+)( )"

    if replace_underline:
        text = re.sub(UNDERSCORE_IN_TEXT, r"\_", text)
    if replace_bfseries:
        text = re.sub(BF_SERIES, r"\\textbf{\g<text>}", text)
    if replace_text_decoration_underline:
        text = re.sub(TEXT_DECO_UNDERLINE, r"\\underline{\g<text>}", text)
    if replace_booktabs_rules:
        text = text.replace("\\toprule", "\\hline")
        text = text.replace("\\midrule", "\\hline")
        text = text.replace("\\bottomrule", "\\hline")
    return text


E = TypeVar('E', bound=Enum)


def parse_enum(cls: Type[E], value: Any) -> E:
    """Parses a value to an enum.
    Simple helper function to parse a value to an enum.

    Parameters
    ----------
    cls : Type[E]
        Type of the enum.
    value : Any
        Value to parse.

    Returns
    -------
    E
        The parsed enum value.

    Raises
    ------
    ValueError
        If the value is not of the correct type or cannot be parsed.
    """
    if not issubclass(cls, Enum):
        raise ValueError(
            f"Type of cls should be an Enum but is: {type(cls).__name__}")
    if isinstance(value, cls):
        return value
    elif isinstance(value, (str, int)):
        return cls(value)
    else:
        raise ValueError(
            f"Type of value for creating: {cls.__name__} should be either string or int but is: {type(value).__name__}")


def parse_type(_type_or_str: Union[Type, str],
               parent_type: Optional[Union[Type, Tuple[Type, ...]]] = None,
               instance_type: Optional[Type] = None,
               variable_name: Optional[str] = None,
               default_value: Optional[Any] = None,
               handle_invalid: Literal["set_default",
                                       "raise", "set_none"] = "raise",
               handle_not_a_class: Literal["ignore", "raise"] = "raise"
               ) -> Type:
    """Parses a type from a string or type.
    Optionally includes checks for beeing a subclass of a parent type or an instance of a type.

    Parameters
    ----------
    _type_or_str : Union[Type, str]
        The type or string to parse.
    parent_type : Optional[Union[Type, Tuple[Type, ...]]], optional
        If the type is subclass of any of the given types, by default None
    instance_type : Optional[Type], optional
        If the type is an instance of some type, by default None
    variable_name : Optional[str], optional
        The name of the variable. Can be used to further specify the error message, by default None
    default_value : Optional[Any], optional
        The default value which should be used when parsing fails and the handle invalid mode is set default, by default None
    handle_invalid : Literal[&quot;set_default&quot;, &quot;raise&quot;, &quot;set_none&quot;], optional
        How an invalid type or error during parsing should be handled, by default "raise"
        "raise" - Raises an error
        "set_default" - Sets the default value
        "set_none" - Sets None as return value
        Both non raise options will log a warning.
    handle_not_a_class : Literal[&quot;ignore&quot;, &quot;raise&quot;], optional
        How to handle if the type is not a class, by default "raise"
        "ignore" - Will ignore the error and check for isinstance
        "raise" - Will raise an error

    Returns
    -------
    Type
        The parsed and checked type.
    """
    def handle_error(error_message: str, exception: Optional[Exception] = None) -> Any:
        prefix = f"Failing to parse type of variable {variable_name}." if variable_name is not None else "Failing to parse type."
        if handle_invalid == "raise":
            if exception is not None:
                raise ValueError(error_message) from exception
            else:
                raise ValueError(error_message)
        elif handle_invalid == "set_default":
            logger.warning(prefix + error_message +
                           f" Setting default value: {default_value}")
            return default_value
        elif handle_invalid == "set_none":
            logger.warning(prefix + error_message + " Setting None")
            return None
        else:
            raise ValueError(
                prefix + f"Invalid handle_invalid: {handle_invalid}")
    _parsed_type = None
    if isinstance(_type_or_str, str):
        try:
            _parsed_type = dynamic_import(_type_or_str)
        except ImportError as e:
            return handle_error(f"Could not import: {_type_or_str} due to an {e.__class__.__name__} does the Module / Type exists and is it installed?", e)
    elif isinstance(_type_or_str, type):
        _parsed_type = _type_or_str
    else:
        return handle_error(f"Type of _type_or_str should be either string or type but is: {type(_type_or_str).__name__}")
    check_instance = True
    if parent_type is not None:
        if inspect.isclass(_parsed_type):
            if not issubclass(_parsed_type, parent_type):
                return handle_error(f"Type: {_parsed_type.__name__} is not a subclass of (any): {parent_type}")
            check_instance = False
        else:
            if handle_not_a_class == "raise":
                return handle_error(f"Type: {_parsed_type.__name__} is not a class.")
            elif handle_not_a_class == "ignore":
                check_instance = True
            else:
                raise ValueError(
                    f"Invalid handle_not_a_class: {handle_not_a_class}")

    if instance_type is not None and check_instance:
        if not isinstance(_parsed_type, instance_type):
            return handle_error(f"Type: {_parsed_type.__name__} is not an instance of: {instance_type.__name__}")
    return _parsed_type


def _format_value(
        value: Any,
        format: Optional[str] = None,
        default_formatters: Optional[Dict[Type, Callable[[Any], str]]] = None) -> str:
    if format is not None and len(format) > 0:
        if "{" in format and "}" in format:
            return format.format(value)
        else:
            return f"{{{format}}}".format(value)
    if default_formatters is None:
        default_formatters = _get_default_formatters()
    fmt = default_formatters.get(type(value), str)
    return fmt(value)


def _get_default_formatters() -> Dict[Type, Callable[[Any], str]]:
    return {
        int: lambda x: f"{x:d}",
        float: lambda x: f"{x:.3f}",
        bool: lambda x: f"{x}",
        timedelta: lambda x: strfdelta(x, "%D days %H:%M:%S") if x.days > 0 else strfdelta(x, "%H:%M:%S"),
        Series: lambda x: f"{x.to_string()}",
        dict: lambda x: ", ".join([f"{_format_value(k)}: {_format_value(v)}" for k, v in x.items()]),
        list: lambda x: ", ".join([_format_value(v) for v in x]),
        tuple: lambda x: ", ".join([_format_value(v) for v in x]),
        type(None): lambda x: "None"
    }


def parse_format_string(format_string: str,
                        obj_list: List[Any],
                        index_variable: str = "index",
                        allow_invocation: bool = False,
                        additional_variables: Optional[Dict[str, Any]] = None,
                        default_formatters: Optional[Dict[Type, Callable[[
                            Any], str]]] = DEFAULT,
                        default_key_formatters: Optional[Dict[str, Callable[[
                            Any], str]]] = None,
                        index_offset: int = 0,
                        found_variables: Optional[List[List[FormatVariable]]] = None) -> List[str]:
    """Formats content of a list of objects with a format string for each object.


    Parameters
    ----------
    format_string : str, optional
        A custom format string to create the string, whereby each variable is substitudet by the corresponding value of the object.

        Every variable in the format string will be replaced by the corresponding value of the config.
        Format specified as {variable:formatter} can be used the format the variable with normal string formatting.

        It also supports Environment variables, which can be used with {ENV:variable[:formatter]}.
        If the environment variable does not exist, an ValueError will be raised.

        Every property of the obj can be used as a variable, in addition to `index_variable` which is the index of the obj in the provided list.

    obj_list : List[Any]
        List of objects to format.

    index_variable : str, optional
        The variable name for the index of the object in the list, by default "index"

    allow_invocation : bool, optional
        If a variable is a function, it can be invoked when set to True, by default False
        Only supports functions without arguments.

    additional_variables : Optional[Dict[str, Any]], optional
        Additional variables which can be used in the format string, by default None
        These are used for all objects in the list.

    default_formatters : Optional[Dict[Type, Callable[[Any], str]]], optional
        Default formatters for the variables, by default DEFAULT
        Defines values for the types which are not specified in the format string. E.g. for int, float, bool, timedelta, Series, dict, list, tuple and None.
        If set to DEFAULT, the default formatters will be used, which are defined in _get_default_formatters().
        Only applied if no format is specified in the format string and no default key formatters are specified.

    default_key_formatters : Optional[Dict[str, Callable[[Any], str]]], optional
        Default formatters for specific keys, e.g. variable names if not specified, by default None
        Takes precedence over the default_formatters but is only applied if no format is specified in the format string.

    index_offset : int, optional
        Offset for the index, by default 0
        If obj_list is a subset of a larger list, the offset can be used to adjust the index.

    found_variables : Optional[List[List[Dict[str]]]], optional
        If set to a list, this will be filled with the found variables for each object, by default None
        For each object in the list, a list of dictionaries will be created, which contains the found variables in order.
        [[{"localizer": str, "variable": str, "formatter": str, "value": Any}, ... per found variable], ... per object]

    Returns
    -------
    List[str]
        Formatted strings.
    """
    pattern = re.compile(
        r"\{((?P<localizer>(ENV)|(env))\:)?(?P<variable>[A-z0-9\_]+)(?P<formatter>:[0-9A-z\.\,]+)?\}")

    matches = [m.groupdict() for m in pattern.finditer(format_string)]
    # Check which variables should included in the format string
    variables = re.findall(pattern, format_string)

    keys = [variable[0] for variable in variables]
    formats = [variable[1] if len(
        variable) > 1 else None for variable in variables]
    loc_key_formats = [(x.get("localizer", None), x.get(
        "variable"), x.get("formatter", None)) for x in matches]

    if additional_variables is None:
        additional_variables = dict()

    if default_formatters is None:
        default_formatters = dict()
    elif default_formatters == DEFAULT:
        default_formatters = _get_default_formatters()
    else:
        pass
    if default_key_formatters is None:
        default_key_formatters = dict()

    results = []
    for i, obj in enumerate(obj_list):
        name = format_string
        replacements = []
        for loc, key, format in loc_key_formats:
            value = MISSING
            if index_variable is not None and key == index_variable:
                value = i
                if index_offset is not None and index_offset > 0:
                    value += index_offset
            else:
                if loc is not None:
                    if loc.lower() != "env":
                        raise ValueError(
                            f"Unknown localizer: {loc} for key: {key} in format string: {format_string}")
                    # Try to find the value in the environment
                    value = os.environ.get(key, MISSING)
                    if value == MISSING:
                        # Check if variable is in the additional variables
                        if key in additional_variables:
                            value = additional_variables[key]
                    if value == MISSING:
                        raise ValueError(
                            f"Environment variable '{key}' does not exist, but was specified in the format string {format_string}.")
                else:
                    # Default localizers
                    try:
                        value = getattr(obj, key)
                    except AttributeError:
                        if isinstance(obj, dict) and key in obj:
                            value = obj[key]
                        elif key in additional_variables:
                            value = additional_variables[key]

            if value == MISSING:
                raise AttributeError(
                    f"Object does not have a property: {key}")

            if value is not None and callable(value) and allow_invocation:
                value = value()

            

            use_format = format
            if use_format is None and key in default_key_formatters:
                # Use the default key formatter if no format is specified
                use_format = default_key_formatters[key]

            _formatted_value = _format_value(
                value, format=use_format, default_formatters=default_formatters)
            if format is None:
                format = ""
            if loc is None:
                loc = ""
            else:
                loc = loc + ":"
            name = name.replace(
                "{" + loc + key + format + "}", _formatted_value)
            replacements.append(FormatVariable(
                localizer=loc, variable=key, formatter=use_format if use_format is not None else "", value=value))
        results.append(name)
        if found_variables is not None:
            found_variables.append(replacements)
    return results


def raise_on_none(obj: Any, shadow_function_in_exception_trace: bool = True) -> Any:
    """Checks if an object is not None and returns it unchanged.

    And raises an error if it is.

    As this checks the traceback for the variable name, it is not recommended to use this in performance critical code, when Nones are frequent,
    and rather more on parsing user input.


    Parameters
    ----------
    obj : Any
        The object to check.

    shadow_function_in_exception_trace : bool, optional
        If the function should be shadowed in the exception trace, by default True
        If set to True, the function will not appear in the exception trace, but the caller of the function.

    Raises
    ------
    ArgumentNoneError
        If the object is None.

    ArgumentNoneTypeSuggestionError
        If the object is None and the type was type hinted.
        Can only find the type if a type hint was used and the function can be imported - this is possible if its a standalone function or constructor of a class.
    """
    ex = None
    if obj is None:
        pattern = r"raise_on_none\((obj( )+=( )+)?(?P<var_name>[A-z0-9_]+)(?P<other_args>(,( )*[A-z0-9_])+)?\)"
        # We are lazy and trying to get the variable name from the stacktrace, but for beeing precise we need https://peps.python.org/pep-0657/ starting from 3.11
        tb = extract_stack()
        # Use the second last element which is the function / code executing the raise_on_none and extract the variable name
        frame = tb[-2]
        line = frame.line
        # Use a regex to extract the variable name
        matches = list(re.finditer(pattern, line))
        if len(matches) == 0:
            # Warn if we cannot extract the variable name
            logger.warning(
                f"Could not extract variable name in raise_on_none. In lineno: {frame.lineno} of file: {frame.filename} with line: {line}.")
            ex = ArgumentNoneError("obj")
        else:
            # If we have exactly one match, we can extract the variable name
            var_name = matches[0]["var_name"]
            # if we have more than one match, display a warning and use the first one
            if len(matches) > 1:
                logger.warning(
                    f"Multiple matches for variable name in raise_on_none: {matches} result may be incorrect. In lineno: {frame.lineno} of file: {frame.filename} multiple statements are used.")
            # Get one frame up and try to extract the function call.
            func_pattern = r"(?P<function_name>[A-z0-9_]+)\((?P<func_args>[A-z0-9,=\*_ ]*)\)"
            frame_func_invocation = tb[-3]
            func_invocations = list(re.finditer(
                func_pattern, frame_func_invocation.line))

            if len(func_invocations) == 0 or len(func_invocations) > 1:
                # Okay no idea on type hints, so we just raise the error
                ex = ArgumentNoneError(var_name)
            else:
                # Exactly one function call, so we can try to import it and get the type hints
                try:
                    func_name = func_invocations[0]["function_name"]
                    # Get the filename as module name
                    module_name_proto = frame_func_invocation.filename.split(".")[
                        0]
                    # Replace any path prefix which is known from the module_name_proto, use the longest

                    import sys
                    index = -1
                    length = -1
                    for prefix in sys.path:
                        if prefix in module_name_proto:
                            if len(prefix) > length:
                                index = module_name_proto.index(prefix)
                                length = len(prefix)
                    if index != -1:
                        prefix = sys.path[index]
                        module_name_proto = module_name_proto.replace(
                            prefix, "")
                        module_name_proto = module_name_proto.strip(
                            os.path.sep)

                    # Format os_indenpendent
                    module_name_proto = format_os_independent(
                        module_name_proto)
                    # Replace / with .
                    module_name = module_name_proto.replace("/", ".")

                    # Import the function
                    func = dynamic_import(module_name + "." + func_name)

                    # Get the type hints
                    signature = inspect.signature(func)
                    var = signature.parameters.get(var_name)
                    if var is None:
                        raise ArgumentNoneError(var_name)
                    if var.annotation == inspect._empty:
                        ex = ArgumentNoneError(var_name)
                    else:
                        annotation = var.annotation
                        ex = ArgumentNoneTypeSuggestionError(
                            var_name, annotation)
                except Exception:
                    # If we cannot import the function, we just raise the error
                    ex = ArgumentNoneError(var_name)
        if shadow_function_in_exception_trace:
            tb = _custom_traceback(1)
            raise ex.with_traceback(tb)
        else:
            raise ex
    else:
        return obj


def get_frame_summary(stack_position: int = 0) -> FrameSummary:
    """Gets information of the current call stack.

    Parameters
    ----------
    stack_position : int, optional
        Position in the call to get the frame from, by default 0
        0 is the call to get_frame_summary, 1 is the caller of the function, 2 is the caller of the caller and so on.

    Returns
    -------
    FrameSummary
        The frame summary object.
    """

    tb = extract_stack()
    position = -1 - (stack_position + 1)
    return tb[position]


def _custom_traceback(stack_position: int = 0) -> TracebackType:
    """Creates a custom traceback.

    Will return a traceback object which can be used to raise an error with a custom traceback.

    Parameters
    ----------
    stack_position : int, optional
        Position in the call stack where the exception should appear, by default 0
        0 is the current function, 1 is the caller of the function, 2 is the caller of the caller and so on.

    Returns
    -------
    TracebackType
        Traceback object.
    """
    # Stack position defines how many frames we go up
    import sys
    stack_position = abs(stack_position)
    try:
        raise ValueError()
    except ValueError:
        traceback = sys.exc_info()[2]
        frame = traceback.tb_frame.f_back  # One iter
        try:
            for _ in range(stack_position):
                frame = frame.f_back
        except Exception:
            pass
        tb = TracebackType(tb_next=None,
                           tb_frame=frame,
                           tb_lasti=frame.f_lasti,
                           tb_lineno=frame.f_lineno)
        return tb


def get_leading_zeros_format(max_number: int) -> str:
    """Gets the format string for leading zeros for integers

    Parameters
    ----------
    max_number : int
        The maximum number which can appear.

    Returns
    -------
    str
        The format string. e.g. 05d for 5 leading zeros. if max_number is up to 99999.
    """
    return f"0{len(str(max_number))}d"


def get_leading_zeros_format_string(max_number: int) -> str:
    """Gets the format string for leading zeros for strings, directly usable in format strings."""
    return f"{{:{get_leading_zeros_format(max_number)}}}"


def format_dataframe_string(df: pd.DataFrame) -> str:
    """Formats a dataframe into a string representation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to format.

    Returns
    -------
    str
        The formatted string.
    """
    return '\t' + df.to_string().replace('\n', '\n\t')


def consecutive_indices_string(x: VEC_TYPE, slice_sep: str = "-", item_sep: str = ",", item_format: Optional[str] = None) -> str:
    """Formats a 1D tensor of (consecutive) indices into a string representation.

    Simplifies the representation of consecutive indices, or repeating orders or numbers by grouping them together.

    Indices of similar step size are grouped together. The output is a string
    where each item is a string of the form "StartSlice[-EndSlice-StepSize]".
    StartSlice is the starting index of the group, EndSlice is the ending index, both are inclusive.
    If the group size is smaller than 3, there will be no grouping and the indices will be printed as single items.
    If only one element is present, it will be simply printed as a single item.

    Example:
    x = [0, 1, 2, 3, 5, 7, 9, 11, 15, 16, 19]
    consecutive_indices_string(x) -> "0-3-1,5-11-2,15,16,19"

    x = [0]
    consecutive_indices_string(x) -> "0"

    x = [0, 1]
    consecutive_indices_string(x) -> "0,1,1"

    x = []
    consecutive_indices_string(x) -> ""

    Parameters
    ----------
    x : VEC_TYPE
        A 1D array of indices.
    slice_sep : str, optional
        Seperator for slices, by default "-"
    item_sep : str, optional
        Seperator for items, by default ","
    item_format : Optional[str], optional
        Format string for each item, by default None
        If None, the default format is "{:d}".
        Formatting applied to the StartSlice, EndSlice and StepSize.
    Returns
    -------
    str
        String representation of the input tensor.
    """
    from tools.transforms.to_numpy import numpyify
    x = numpyify(x)
    if item_format is None:
        item_format = "{:d}"
    if "int" not in str(x.dtype):
        raise ValueError("Input must be an integer array.")
    if len(x.shape) > 1:
        raise ValueError("Input must be a 1D array.")
    if len(x) == 2:
        return f"{item_format.format(x[0])},{item_format.format(x[0])},{item_format.format(1)}"
    if len(x) == 1:
        return f"{item_format.format(x[0])}"
    if len(x) == 0:
        return ""
    grad = x[1:] - x[:-1]
    rets = []

    def _append(l, start, end, step):
        if start == end:
            l.append(f"{item_format.format(start)}")
        elif (end - start) == step:
            l.append(f"{item_format.format(start)}")
            l.append(f"{item_format.format(end)}")
        else:
            l.append(
                f"{item_format.format(start)}{slice_sep}{item_format.format(end)}{slice_sep}{item_format.format(step)}")

    istart = 0
    cstart = x[0]
    cend = None
    cstep = None
    while istart < len(x):
        cend = x[istart]
        if cstep is None:
            cstep = grad[istart]
        else:
            if istart == len(grad) or cstep != grad[istart]:
                _append(rets, cstart, cend, cstep)
                if istart == len(grad):
                    break
                istart += 1
                cstart = x[istart]
                if istart == len(grad):
                    _append(rets, cstart, cend, cstep)
                    break
                cstep = grad[istart]
            else:
                pass
        istart += 1
    return item_sep.join(rets)
