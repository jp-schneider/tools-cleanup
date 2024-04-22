from typing import Any, Dict, List, Optional
import logging
try:
    import pandas as pd
    from pandas import Series, DataFrame
except (NameError, ImportError, ModuleNotFoundError):
    Series = object
    DataFrame = object
    pd = object
    pass  # Ignore exception on import

from tools.util.format import to_snake_case
from tools.util.object_factory import ObjectFactory
from tools.util.typing import DEFAULT


class SeriesConvertibleMixin:
    """Mixin to indicate that the object can be converted to a pandas series and vice versa."""

    @classmethod
    def get_properties(cls) -> List[str]:
        """Get the properties of the class.

        Returns
        -------
        List[str]
            The list of properties.
        """
        import dataclasses
        import inspect
        if dataclasses.is_dataclass(cls):
            return list((x.name for x in dataclasses.fields(cls)))
        else:
            if hasattr(cls, '__annotations__'):
                return list(cls.__annotations__.keys())
            return [x for x, y in inspect.getmembers(cls) if not x.startswith('_') and not inspect.isfunction(y)]

    @classmethod
    def get_empty_data_frame(cls) -> DataFrame: # type: ignore
        """Get an empty data frame with the properties of the object.

        Returns
        -------
        DataFrame
            The empty data frame.
        """
        return DataFrame(columns=cls.get_properties())

    @classmethod
    def default_key_mapping(cls) -> Dict[str, str]:
        """Default key mapping for the object.
        Can be used to convert key names.

        Returns
        -------
        Dict[str, str]
            Mapping of keys.
        """
        return None

    def to_series(self) -> Series:  # type: ignore
        """Converts the object to a series.
        Uses by default all entries in __dict__

        Returns
        -------
        Series
            The created series.
        """
        return Series(dict(vars(self)))

    @classmethod
    def from_data_frame(cls,
                        df: DataFrame,  # type: ignore
                        key_mapping: Optional[Dict[str, str]] = DEFAULT,
                        allow_dynamic_args: bool = True, **kwargs) -> List[Any]:
        """Generates a list of instances from a dataframe.

        Parameters
        ----------
        df : DataFrame
            The dataframe to create the items from.
        allow_dynamic_args : bool, optional
            If dynamic args should be allowed when creating the object, by default True

        Returns
        -------
        List[Any]
            The list with instances.
        """
        if df is None:
            return []
        if key_mapping is DEFAULT:
            key_mapping = cls.default_key_mapping()
        return [cls.from_series(x,
                                key_mapping=key_mapping,
                                allow_dynamic_args=allow_dynamic_args, **kwargs) for x in df.iloc]

    @classmethod
    def from_dict(
            cls,
            _dict: Dict[str, Any],
            snake_case_conversion: bool = False,
            allow_dynamic_args: bool = True,
            key_mapping: Optional[Dict[str, str]] = DEFAULT,
            additional_data: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> Any:
        """Creates the current class from a dict.
        Will also convert keys to snake case and additional data can be inserted.

        Parameters
        ----------
        _dict : Dict[str, Any]
            The dict of entries to create the object from.
        snake_case_conversion : bool, optional
            If keys should be converted to snake case, by default False
        allow_dynamic_args : bool, optional
            If the cls should also accept dynamic arguments, by default True
        key_mapping : Optional[Dict[str, str]], optional
            Mapping of keys to convert will be applied before any other technique, by default None
        additional_data : Optional[Dict[str, Any]], optional
            Additional data to insert in the dict, by default None

        Returns
        -------
        Any
            The created object.
        """
        if additional_data is not None:
            _dict.update(additional_data)
        if key_mapping is DEFAULT:
            key_mapping = cls.default_key_mapping()
        if key_mapping is not None:
            converted = {}
            for key, value in _dict.items():
                if key in key_mapping:
                    converted[key_mapping[key]] = value
                else:
                    converted[key] = value
            _dict = converted
        if snake_case_conversion:
            converted = {}
            for key, value in _dict.items():
                converted[to_snake_case(key)] = value
            _dict = converted
        return ObjectFactory.create_from_kwargs(cls, allow_dynamic_args, ** _dict)

    @classmethod
    def from_series(
            cls, series: Series,  # type: ignore
            snake_case_conversion: bool = False,
            allow_dynamic_args: bool = True,
            key_mapping: Optional[Dict[str, str]] = DEFAULT,
            additional_data: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> Any:
        """Create the object from a series. "Inverts" the to_series operation.

        Parameters
        ----------
        series : Series
            The series to create the object from.
        snake_case_conversion : bool, optional
            If names or keys should be converted to snake case, by default False
        allow_dynamic_args : bool, optional
            If dynamic arguments should be allowed, by default True
        key_mapping : Optional[Dict[str, str]], optional
            Mapping of keys to convert will be applied before any other technique, by default None
        additional_data : Optional[Dict[str, Any]], optional
            Additional data to create the object, by default None

        Returns
        -------
        Any
            The created object.
        """
        d = series.to_dict()
        for k, v in d.items():
            is_na = pd.isna(v)
            if isinstance(is_na, bool) and is_na:
                d[k] = None
        return cls.from_dict(d,
                             snake_case_conversion=snake_case_conversion,
                             allow_dynamic_args=allow_dynamic_args,
                             key_mapping=key_mapping,
                             additional_data=additional_data,
                             **kwargs)
