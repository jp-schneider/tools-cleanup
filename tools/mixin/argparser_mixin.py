import argparse
import inspect
import logging
from argparse import ArgumentParser
from dataclasses import MISSING, Field
import pathlib
from typing import Any, Dict, List, Optional, Type, get_args
from tools.util.typing import is_list_type
from simple_parsing.docstring import get_attribute_docstring
from typing_inspect import is_literal_type, is_optional_type, is_tuple_type, is_classvar, is_union_type
import os
from tools.error import UnsupportedTypeError, IgnoreTypeError
from enum import Enum
from tools.logger.logging import logger
from tools.serialization.json_convertible import JsonConvertible

WARNING_ON_UNSUPPORTED_TYPE = True
"""If true, a warning will be printed if a type is not supported."""

import sys
vinfo = sys.version_info
IS_ABOVE_3_9 = (vinfo.major >= 3 and vinfo.minor > 8) or vinfo.major > 3

def set_warning_on_unsupported_type(warning: bool) -> None:
    """Sets the warning on unsupported type.

    Parameters
    ----------
    warning : bool
        If true, a warning will be printed if a type is not supported.
    """
    global WARNING_ON_UNSUPPORTED_TYPE
    WARNING_ON_UNSUPPORTED_TYPE = warning


class ParseEnumAction(argparse.Action):
    """Custom action to parse enum values from the command line."""

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        self.enum_type = kwargs.pop("enum_type", None)
        super().__init__(option_strings, dest, **kwargs)
        if self.enum_type is None:
            raise ValueError("enum_type must be specified")

    def __call__(self, parser, namespace, values, option_string=None):
        v = None
        if isinstance(values, str):
            v = self.enum_type(values)
        elif isinstance(values, list):
            v = [self.enum_type(x) for x in values]
        else:
            raise ValueError(
                f"Unsupported type of values: {type(values).__name__}")
        setattr(namespace, self.dest, v)


class ArgparserMixin:
    """Mixin wich provides functionality to construct a argparser for a
    dataclass type and applies its data."""

    @classmethod
    def _map_type_to_parser_arg(cls, field: Field, _type: Optional[Type] = None, is_optional: bool = False) -> Dict[str, Any]:
        """Mapping field types to argparse arguments.
        Parameters
        ----------
        field : Field
            The field which should be mapped.
        _type : Optional[Type]
            Alterating field type on recursive calls, default None.

        is_optional : bool
            If the type is optional, this is used to set the required flag to False.
            Dont need to be handled here, just for information.

        Returns
        -------
        Dict[str, Any]
            kwargs for the argparse add argument call.
        Raises
        ------
        UnsupportedTypeError
            If the type is not supported for comparison.
        """
        if not _type:
            _type = field.type
        if isinstance(_type, Type) and issubclass(_type, bool):
            # Check default and switch accordingly
            if not IS_ABOVE_3_9 or not is_optional:
                if not field.default:
                    return dict(action="store_true")
                else:
                    return dict(action="store_false", name_prefix="no_")
            else:
                return dict(action=argparse.BooleanOptionalAction)
        elif isinstance(_type, Type) and issubclass(_type, os.PathLike):
            # If type is a PathLike, we assume its a string.
            return dict(type=str)
        elif isinstance(_type, Type) and issubclass(_type, (str, int, float)):
            return dict(type=_type)
        elif is_literal_type(_type):
            arg = get_args(_type)
            ret = dict()
            if len(arg) > 0:
                ret["choices"] = list(arg)
                ret["type"] = type(arg[0])
            return ret
        elif _type == type(None):
            raise IgnoreTypeError()
        elif ((isinstance(_type, Type) and
              (issubclass(_type, list) or issubclass(_type, tuple)))
              or is_tuple_type(_type) or is_list_type(_type)):
            # Handling list or tuples the same way
            # Limitation: Lists can have an arbitrary amount of arguments.
            args = dict()
            arg = get_args(_type)
            if is_tuple_type(_type):
                # If a typing tuple, the number types can be directly inferred
                args["nargs"] = len(arg)
            elif (isinstance(_type, Type) and issubclass(_type, tuple)):
                # Empty tuple would not make sense so limit it to 1...n
                args["nargs"] = "+"
            elif (isinstance(_type, List) and issubclass(_type, list)) or is_list_type(_type):
                # For list empty would be ok.
                args["nargs"] = "*"
            else:
                raise UnsupportedTypeError(
                    f"Dont know how to handle type: {_type} of field: {field.name}.")
            if len(arg) > 0:
                args["type"] = arg[0]
            return args
        elif isinstance(_type, Type) and issubclass(_type, Enum):
            choices = [x.value for x in _type]
            # Get type of choice
            _arg_type = type(next((x for x in choices), str))
            return dict(type=_arg_type,
                        enum_type=_type,
                        choices=choices,
                        action=ParseEnumAction)
        elif is_union_type(_type):
            # Check if contains None
            _new_types = get_args(_type)
            any_matched = False
            args = dict()
            for _new_type in _new_types:
                try:
                    args = cls._map_type_to_parser_arg(field, _new_type)
                    any_matched = True
                    break
                except (IgnoreTypeError, UnsupportedTypeError):
                    continue
            if is_optional_type(_type):
                args["required"] = False
            if any_matched:
                return args
            else:
                raise UnsupportedTypeError(
                    f"Dont know how to handle type: {_type} of field: {field.name}.")
        elif is_optional_type(_type):
            # Unpack optional type.
            _new_type = get_args(_type)[0]
            args = cls._map_type_to_parser_arg(field, _new_type, is_optional=True)
            # Because its optional, make it non required
            args["required"] = False
            return args
        elif is_classvar(_type):
            raise IgnoreTypeError()
        else:
            raise UnsupportedTypeError(
                f"Dont know how to handle type: {_type} of field: {field.name}.")

    @classmethod
    def _get_parser_arg_value(cls, field: Field, value: Any, _type: Optional[Type] = None) -> Any:
        if not _type:
            _type = field.type
        if (isinstance(_type, Type) and issubclass(_type, (str, int, float, bool))) or value is None:
            return value  # Simple types
        elif is_literal_type(_type):
            arg = get_args(_type)
            ret = dict()
            if len(arg) > 0:
                if value not in arg:
                    raise ValueError(
                        f"{value} is not value supported for literal type: {_type}")
                return value
            else:
                raise ValueError(
                    f"Could not specify {value} for literal type: {_type}")
        elif ((isinstance(_type, Type) and
              (issubclass(_type, list) or issubclass(_type, tuple)))
              or is_tuple_type(_type) or is_list_type(_type)):
            if is_tuple_type(_type) or (isinstance(_type, Type) and issubclass(_type, tuple)):
                return tuple(value)  # Be shure that value is a tuple
            else:
                # For list empty would be ok.
                return list(value)
        elif is_optional_type(_type):
            # Unpack optional type.
            _new_type = get_args(_type)[0]
            return cls._get_parser_arg_value(field, value=value, _type=_new_type)
        elif is_union_type(_type):
            # Check if contains None
            _new_types = get_args(_type)
            any_matched = False
            args = dict()
            for _new_type in _new_types:
                try:
                    val = cls._get_parser_arg_value(field, value, _new_type)
                    any_matched = True
                    break
                except (IgnoreTypeError, UnsupportedTypeError):
                    continue
            if any_matched:
                return val
            else:
                raise UnsupportedTypeError(
                    f"Dont know how to handle type: {_type} of field: {field.name}.")
        elif isinstance(_type, Type) and issubclass(_type, Enum):
            return value
        else:
            raise UnsupportedTypeError(
                f"Dont know how to handle type: {_type} of field: {field.name}.")

    @classmethod
    def _get_parser_members(cls) -> List[Field]:
        """Returning the parser members which are in the dataclass.
        """
        # Get all dataclass properties
        members = inspect.getmembers(cls)
        all_fields: List[Field] = list(
            next((x[1] for x in members if x[0] == '__dataclass_fields__'), dict()).values())

        # Non private fields
        fields = [x for x in all_fields if not x.name.startswith(
            '_') and x.name not in cls.argparser_ignore_fields()]
        return fields

    @classmethod
    def argparser_ignore_fields(cls) -> List[str]:
        """Can be derived to ignore custom fields and not apply them in the argparser.

        Returns
        -------
        List[str]
            List of fields to ignore.
        """
        return [

        ]

    @classmethod
    def  get_parser(cls, parser: Optional[ArgumentParser] = None, sep: str = "-") -> ArgumentParser:
        """Creates / fills an Argumentparser with the fields of the current class.
        Inheriting class must be a dataclass to get annotations and fields.
        By default only puplic field are used (=field with a leading underscore "_" are ignored.)
        Parameters
        ----------
        parser : Optional[ArgumentParser]
            An existing argument parser. If not specified a new one will be created. Defaults to None.

        sep : str
            Separator for the argument name, will replace the "_" of config classes. Defaults to "-".

        Returns
        -------
        ArgumentParser
            The filled argument parser.
        """
        # Create parser if None
        if not parser:
            parser = argparse.ArgumentParser(
                description=f'Default argument parser for {cls.__name__}',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        fields = cls._get_parser_members()

        for field in fields:
            name = field.name.replace("_", sep)
            try:
                args = cls._map_type_to_parser_arg(field)
            except IgnoreTypeError as ig:
                continue
            except UnsupportedTypeError as err:
                msg = f"Could not create parser arg for field: {field.name} due to a {type(err).__name__} \n {str(err)}"
                if WARNING_ON_UNSUPPORTED_TYPE:
                    logging.warning(msg)
                else:
                    logging.debug(msg)
                continue
            # default value
            if field.default != MISSING:
                args["default"] = field.default
            # docstring
            name_prefix = args.pop("name_prefix", "")
            if name_prefix:
                name_prefix = name_prefix.replace("_", sep)
            args["help"] = str(get_attribute_docstring(
                cls, field_name=field.name).docstring_below)
            parser.add_argument("--" + name_prefix + name, **args)

        return parser

    @classmethod
    def from_parsed_args(cls, parsed_args: Any) -> 'ArgparserMixin':
        """Creates an ArgparserMixin object from parsed_args which is the result
        of the argparser.parse_args() method.
        Parameters
        ----------
        parsed_args : Any
            The parsed arguments.
        Returns
        -------
        ArgparserMixin
            The instance of the object filled with cli data.
        """
        fields = cls._get_parser_members()
        # Look for matching fieldnames
        ret = dict()
        for field in fields:
            if hasattr(parsed_args, field.name):
                value = getattr(parsed_args, field.name)
                ret[field.name] = cls._get_parser_arg_value(field, value)
        return cls(**ret)

    @classmethod
    def get_field_name_with_prefix_for_argparser(cls, field: Field) -> str:
        """Returns for a given field with an eventually prefix"""
        name = field.name
        if field.type is None:
            return name
        is_optional = False
        _type = field.type
        if is_optional_type(_type):
            is_optional = True
            _type = get_args(field.type)[0]
        if isinstance(_type, Type) and issubclass(_type, bool):
            # Check default and switch accordingly
            if not IS_ABOVE_3_9 or not is_optional:
                if field.default:
                    return "no_" + name
        return name
    
    def apply_parsed_args(self, parsed_args: Any) -> None:
        """Applies parsed_args, which is the result
        of the argparser.parse_args() method, to an existing object.
        But only if its different to the objects default.
        Parameters
        ----------
        parsed_args : Any
            The parsed arguments.
        """
        fields = type(self)._get_parser_members()
        # Look for matching fieldnames
        ret = dict()
        for field in fields:
            name = self.get_field_name_with_prefix_for_argparser(field)
            if hasattr(parsed_args, name):
                value = getattr(parsed_args, name)
                if ((field.default != MISSING and (value is not None and value != field.default)) or
                        (field.default_factory != MISSING and (
                            value is not None and value != field.default_factory()))
                        or (field.default == MISSING and field.default_factory == MISSING)):
                    new_value = self._get_parser_arg_value(field, value)
                    old_value = getattr(self, field.name)
                    if self.allow_cli_config_overwrite(field, old_value, new_value):
                        setattr(self, field.name, new_value)
                    else:
                        logger.info(
                            f"Value for field {field.name} was not overwritten.")

    def allow_cli_config_overwrite(self, field: Field, config_value: Any, cli_value: Any) -> bool:
        """
        Method will be called for each field which was provided as a cli argument.
        The default behavior is to allow the cli argument to overwrite the value in the current config object.

        Parameters
        ----------
        field : Field
            The field in the config which was provided as a cli argument.
            Name of the property can be accessed via field.name

        config_value : Any
            The current value of the field in the config object. This value will be overwritten if the method returns True.

        cli_value : Any
            The value which was provided as a cli argument to the field.

        Returns
        -------
        bool
            If the cli config should be allowed to overwrite the config object.
            False if the value should not be overwritten and discarded.
        """
        return True

    @classmethod
    def parse_args(cls,
                   parser: ArgumentParser = None,
                   add_config_path: bool = True,
                   sep: str = "-",
                   ) -> "ArgparserMixin":
        """Parses the arguments from the command line and returns a config object.

        Parameters
        ----------
        parser : ArgumentParser
            Predefined parser object.

        add_config_path : bool, optional
            If the parse args method should consider a --config_path argument, by default True

        sep : str
            Separator for the argument name, will replace the "_" of config classes. Defaults to "-".

        Returns
        -------
        ArgparserMixin
            The instance of the object filled with cli data.
        """
        from tools.logger.logging import logger
        parser = cls.get_parser(parser, sep=sep)

        if add_config_path:
            parser.add_argument("--config-path", type=str,
                                help="Path to an initial config file. Other arguments supplied may overwrite the values in the config file.",
                                default=None, required=False)

        args = parser.parse_args()

        config: ArgparserMixin = None
        if add_config_path and args.config_path:
            args.config_path = args.config_path.strip("\"").strip("\'")
            config = JsonConvertible.load_from_file(args.config_path)
            if not isinstance(config, cls):
                logger.warning(
                    f"Loaded config from file is not of type {cls.__name__}. But {type(config).__name__}.")
            config.apply_parsed_args(args)
        else:
            config = cls.from_parsed_args(args)
        return config
