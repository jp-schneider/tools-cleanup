from typing import List, Union, Dict
from tools.serialization.files.file_handle import FileHandle
import pandas as pd
import numpy as np


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
        return df

    def to_file_conversion(self, obj: pd.DataFrame):
        self.index_col = list(obj.index.names)
        self.dtypes = obj.convert_dtypes().dtypes.to_dict()
        obj.to_csv(self.file_path, index=True,
                   sep=self.sep, header=self.header)
