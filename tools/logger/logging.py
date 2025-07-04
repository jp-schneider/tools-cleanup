import logging
import sys
from typing import Optional
from tools.util.reflection import check_package
from tools.util.package_tools import get_package_name, get_invoked_package_name, get_project_root_path
from rich.logging import RichHandler
from rich.console import Console

ONE_TIME_MESSAGES = dict()
"""Dictionary to store one time messages."""

_loggers = dict()
"""Dictionary to store loggers for other packages."""


def get_logger(package_name: Optional[str] = None) -> logging.Logger:
    """Gets the logger for the given package name.


    Parameters
    ----------
    package_name : str, optional
        The package name to get the logger for, by default None

    Returns
    -------
    logging.Logger
        The logger for the given package name.
    """
    if package_name is None:
        try:
            package_name = get_invoked_package_name()
        except Exception as error:
            print(f"Error while getting the package name: {error}")
            package_name = None
    if package_name is None:
        package_name = "tools"
    if package_name not in _loggers:
        _loggers[package_name] = logging.getLogger(package_name)
    return _loggers[package_name]


logger: logging.Logger = get_logger()
"""Default logger for the invoked package."""

tools_logger: logging.Logger = get_logger("tools")
"""Logger for the tools package."""


def basic_config(level: int = logging.INFO, filename: Optional[str] = None, reinit: bool = False) -> None:
    """Basic logging configuration with sysout logger.

    Parameters
    ----------
    level : logging._Level, optional
        The logging level to consider, by default logging.INFO
    """
    root = logging.getLogger()
    root.setLevel(level)
    _fmt = '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    _date_fmt = '%Y-%m-%d:%H:%M:%S'
    handlers = []
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(_fmt, _date_fmt))
        handlers.append(fh)
    logging.basicConfig(
        # format=_fmt,
        format="%(message)s",
        datefmt=_date_fmt,
        level=level,
        handlers=[RichHandler(
            rich_tracebacks=True, console=Console(width=255))] + handlers,
        force=reinit
    )
    # fmt = logging.Formatter(_fmt, _date_fmt)
    # root.handlers[0].setFormatter(fmt)
    # Set default for other loggers
    if check_package("matplotlib"):
        logger = logging.getLogger("matplotlib")
        logger.setLevel(logging.WARNING)


def get_messaged(key: str) -> bool:
    """Gets whether a message with the given key was already messaged.

    Parameters
    ----------
    key : str
        The key to check for.

    Returns
    -------
    bool
        Whether the message was already messaged.
    """
    global ONE_TIME_MESSAGES
    if key not in ONE_TIME_MESSAGES:
        ONE_TIME_MESSAGES[key] = 1
        return False
    else:
        ONE_TIME_MESSAGES[key] += 1
        return True
