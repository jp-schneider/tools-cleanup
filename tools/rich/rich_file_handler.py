from rich.logging import RichHandler
from rich.console import Console
from logging import FileHandler, LogRecord, StreamHandler
from rich._null_file import NULL_FILE
from typing import Optional, Tuple, List
import os
import io
from enum import Enum


class TextType(str, Enum):
    """Enum for text types."""
    PLAIN = "plain"
    HTML = "html"


class RichFileHandler(RichHandler):
    """Rich File Handler that logs to a file using Rich formatting."""

    def __init__(self,
                 filename: str,
                 mode: str = "a",
                 **kwargs) -> None:
        self.buffer = io.StringIO()
        console = Console(width=200, file=self.buffer,
                          force_terminal=False, force_jupyter=False, record=True)
        if 'console' in kwargs:
            raise ValueError(
                "Cannot specify console in RichFileHandler, it is created automatically.")
        ext = os.path.splitext(filename)[1].lower().strip(".")
        self.file_format = TextType.HTML if ext in [
            "html", "htm"] else TextType.PLAIN
        super().__init__(console=console, **kwargs)
        self.file_handler = FileHandler(filename, mode=mode, encoding="utf-8")

    def _get_text(self) -> None:
        if self.file_format == TextType.HTML:
            return self.console.export_html(inline_styles=False)
        return self.console.export_text(styles=False)

    def emit(self, record: LogRecord) -> None:
        super().emit(record)
        if self.file_handler.stream is None:
            self.file_handler.stream = self.file_handler._open()
        try:
            text = self._get_text()
            self.file_handler.stream.write(text)
            self.file_handler.flush()
        except RecursionError:
            raise
        except Exception as e:
            self.handleError(record)

    def close(self):
        """
        Closes the file handler.
        """
        self.buffer.close()
        self.file_handler.close()
