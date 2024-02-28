import logging
import sys
from typing import Optional
from tools.util.package_tools import get_package_name, get_invoked_package_name, get_project_root_path

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
        package_name = get_invoked_package_name()
    if package_name not in _loggers:
        _loggers[package_name] = logging.getLogger(package_name)
    return _loggers[package_name]

logger: logging.Logger = get_logger()
"""Default logger for the invoked package."""

tools_logger: logging.Logger = get_logger(get_package_name(get_project_root_path()))
"""Logger for the tools package."""

def basic_config(level: int = logging.INFO):
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
    logging.basicConfig(format=_fmt,
                        datefmt=_date_fmt,
                        level=level)
    fmt = logging.Formatter(_fmt, _date_fmt)
    root.handlers[0].setFormatter(fmt)
    # Set default for other loggers
    if "matplotlib" in sys.modules:
        logger = logging.getLogger("matplotlib")
        logger.setLevel(logging.WARNING)

