
from typing import Any, Optional, Union
from tools.serialization.json_convertible import JsonConvertible
import os
from io import BytesIO
from tools.util.path_tools import format_os_independent, numerated_file_name, relpath


class FileHandle(JsonConvertible):

    is_binary: bool
    """Wether the file is binary or not."""

    file_path: str
    """The path to the file."""

    need_to_open: bool
    """If the file needs to be opened."""

    append: bool
    """If the file needs to be appended."""

    override: bool
    """If the file needs to be overwritten."""

    make_dirs: bool
    """If the directories need to be created."""

    use_relative_path: bool
    """If the path is should be relative."""

    def __init__(self,
                 file_path: str,
                 is_binary: bool = False,
                 need_to_open: bool = True,
                 append: bool = False,
                 override: bool = True,
                 make_dirs: bool = True,
                 use_relative_path: bool = True,
                 decoding: bool = False,
                 **kwargs
                 ):
        super().__init__(decoding, **kwargs)
        if use_relative_path:
            p = os.path.normpath(os.path.abspath(file_path))
            rp = relpath(os.getcwd(), p, is_from_file=False, is_to_file=True)
            file_path = rp
        self.file_path = format_os_independent(file_path)
        self.is_binary = is_binary
        self.append = append
        self.need_to_open = need_to_open
        self.override = override
        self.make_dirs = make_dirs
        if decoding:
            return

    def from_file_conversion(self, file: Optional[Any]) -> Any:
        """Gets the file handle and returns a reconstructed object.

        Parameters
        ----------
        file : Any
            The File handle optained from opening the file.
            If Need to open is False, this will be None. and the function itself should open the file.

        Returns
        -------
        Any
            The reconstructed object.
        """
        return file

    def to_file_conversion(self, obj: Any) -> Optional[Union[str, bytes, BytesIO]]:
        """
        Convert the object to a string or bytes representation.

        Parameters
        ----------
        obj : Any
            Gets the object and returns a string or bytes representation of the object.
            Which will be written to the file.
            If the file is binary, it should return bytes.
            If none is returned, it assumes the object has be written already.

        Returns
        -------
        Optional[str | bytes | BytesIO]
            Anything which can be written.
        """

    def read(self) -> Any:
        """Read the file.

        Returns
        -------
        Any
            The reconstructed object.
        """
        if self.need_to_open:
            with open(self.file_path, "rb" if self.is_binary else "r") as f:
                return self.from_file_conversion(f)
        else:
            return self.from_file_conversion(None)

    def write(self, obj: Any) -> None:
        """Write the object to the file.

        Parameters
        ----------
        obj : Any
            The object to write.
        """
        # Check if directory exists
        if self.make_dirs:
            base_dir = os.path.dirname(self.file_path)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
        ser = self.to_file_conversion(obj)
        if ser is None:
            return
        else:
            mode = "a" if self.append else "w"
            # Check if basename exists
            if self.make_dirs:
                base_dir = os.path.dirname(self.file_path)
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir, exist_ok=True)
            with open(self.file_path, mode + ("b" if self.is_binary else "")) as f:
                f.write(ser)
                f.flush()

    @classmethod
    def for_object(cls,
                   obj: Any,
                   file_path: str,
                   **kwargs
                   ) -> "FileHandle":
        """Write the object to the file.

        Parameters
        ----------
        obj : Any
            The object to write.
        file_path : str
            The path to the file.
        **kwargs
            Additional keyword arguments which are passed to the constructor.

        Returns
        -------
        FileHandle
            The file handle.
        """
        fh = cls(file_path, **kwargs)
        fh.write(obj)
        return fh
