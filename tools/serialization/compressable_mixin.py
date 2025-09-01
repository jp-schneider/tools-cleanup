from abc import abstractmethod
from typing import Any, Dict
from enum import Enum
from tools.serialization.json_convertible import JsonConvertible
import base64
import io
import lzma


def _encode_buffer(buf: bytes) -> str:
    return base64.b64encode(buf).decode()


def _decode_buffer(buf: str) -> bytes:
    return base64.b64decode(buf.encode())


class CompressableMixin(JsonConvertible):
    """Mixin class to add compression capabilities to JSON serializable classes."""

    def __init__(self,
                 compression: bool = False,
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        self.compression = compression
        super().__init__(decoding=decoding, **kwargs)
        if decoding:
            return

    @classmethod
    def to_ascii(cls, value: Any, compression: bool, lzma_args: Dict[str, Any] = None) -> str:
        """Convert the object to a base64 encoded string with optional compression.

        Parameters
        ----------
        value : Any
            The object to convert.

        compression : bool
            Whether to apply compression.

        lzma_args : Dict[str, Any]
            Arguments for LZMA compression.

        Returns
        -------
        str
            The base64 encoded string of the (possibly compressed) object.
        """
        buf = cls.to_bytes(value)
        return cls.process_object_ascii(buf, compression, lzma_args)

    @classmethod
    def from_ascii(cls, value: str, compression: bool, lzma_args: Dict[str, Any] = None) -> Any:
        """Reconstruct the object from a base64 encoded string with optional decompression.

        Parameters
        ----------
        value : str
            The base64 encoded string to decode.

        compression : bool
            Whether the original data was compressed.

        lzma_args : Dict[str, Any]
            Arguments for LZMA decompression.

        Returns
        -------
        Any
            The reconstructed object.
        """
        buf = cls.recover_object_ascii(value, compression, lzma_args)
        return cls.from_bytes(buf)

    @classmethod
    @abstractmethod
    def to_bytes(cls, value: Any) -> bytes:
        """Convert the object to a byte representation.

        Parameters
        ----------
        value : Any
            The object to convert.

        Returns
        -------
        bytes
            The byte representation of the object.
        """
        pass

    @classmethod
    @abstractmethod
    def from_bytes(cls, buf: bytes) -> Any:
        """Reconstruct the object from its byte representation.

        Parameters
        ----------
        buf : bytes
            The byte representation of the object.

        Returns
        -------
        Any
            The reconstructed object.
        """
        pass

    @classmethod
    def process_object_ascii(cls, buf: bytes, compression: bool, lzma_args: Dict[str, Any] = None) -> str:
        """Process the byte buffer with optional compression and encode to base64 string.

        Parameters
        ----------
        buf : bytes
            The byte buffer to process.

        compression : bool
            Whether to apply compression.

        lzma_args : Dict[str, Any]
            Arguments for LZMA compression.

        Returns
        -------
        str
            The base64 encoded string of the (possibly compressed) byte buffer.
        """
        if lzma_args is None:
            lzma_args = {}
        if not compression:
            return _encode_buffer(buf)
        else:
            return _encode_buffer(lzma.compress(buf, **lzma_args))

    @classmethod
    def recover_object_ascii(cls, value: str, compression: bool, lzma_args: Dict[str, Any] = None) -> bytes:
        """Decode the base64 string and optionally decompress to recover the original byte buffer.

        Parameters
        ----------
        value : str
            The base64 encoded string to decode.

        compression : bool
            Whether the original data was compressed.

        lzma_args : Dict[str, Any]
            Arguments for LZMA decompression.

        Returns
        -------
        bytes
            The recovered byte buffer.
        """
        if lzma_args is None:
            lzma_args = {}
        decoded = _decode_buffer(value)
        if not compression:
            return decoded
        else:
            return lzma.decompress(decoded, **lzma_args)
