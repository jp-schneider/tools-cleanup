from typing import List, Union, Dict
from tools.serialization.files.file_handle import FileHandle
import pandas as pd
import numpy as np
import re


def convert_string_to_array(value: str) -> np.ndarray:
    """Parses a string into a numpy array.

    Applicable for ND arrays.

    Example:
    ```python

    value = "[[1. 2. 3.]\n [4. 5. 6.]\n [7. 8. 9.]]"
    arr = match_arr(value)
    print(arr)
    ```

    Parameters
    ----------
    value : str
        A string representation of a numpy array.

    Returns
    -------
    np.ndarray
        Parsed numpy array.

    Raises
    ------
    ValueError
        If the value does not match the allowed pattern.
    """
    allowed_pattern = r"^( )*\[[\[\d\+\.\se\-\]\n\rnan]+\]( )*$"
    replace_with_comma = r"(?<=[\d\.n])( )(?=( )*[\d\-\+n])"
    line_feed_comma = r"(?<=\])(\r)?\n(?=(\s)*\[)"
    sub_line = ","
    sub_line_feed = r",\n"

    if not re.fullmatch(allowed_pattern, value):
        return value

    value = re.sub(replace_with_comma, sub_line, value)
    value = re.sub(line_feed_comma, sub_line_feed, value)
    # Replace nan with np.nan
    value = value.replace("nan", "np.nan")
    arr = np.array(eval(value))
    return arr


class DataFrameCSVFileHandle(FileHandle):
    """File handle for dataframes."""

    sep: str
    """The delimiter for the csv file."""

    header: bool
    """If the csv file has a header."""

    index_col: Union[str, List[str]]
    """The index column."""

    dtypes: Dict[str, np.dtype]

    def __init__(self, file_path: str,
                 sep: str = ";",
                 decoding: bool = False,
                 **kwargs):
        super().__init__(file_path, is_binary=False, need_to_open=False,
                         append=False, decoding=decoding, **kwargs)
        self.sep = sep
        self.header = True

    def from_file_conversion(self, file):
        df = pd.read_csv(self.file_path, sep=self.sep,
                         dtype=self.dtypes,
                         header=0 if self.header else None)
        df.set_index(self.index_col, inplace=True)
        self.check_for_parsable_columns(df)
        return df

    def to_file_conversion(self, obj: pd.DataFrame):
        self.index_col = list(obj.index.names)
        self.dtypes = obj.convert_dtypes().dtypes.to_dict()
        obj.to_csv(self.file_path, index=True,
                   sep=self.sep, header=self.header)

    def check_for_parsable_columns(self, data_frame: pd.DataFrame):
        """Checks if the columns of the values dataframe are parsable."""
        for col in data_frame.columns:
            if data_frame[col].dtype == np.dtype('O'):
                try:
                    data_frame[col] = data_frame[col].apply(
                        convert_string_to_array)
                except Exception as e:
                    pass
