import logging
import sys
from typing import Optional, Any
from tools.util.reflection import check_package
from tools.util.package_tools import get_package_name, get_invoked_package_name, get_project_root_path
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.text import Text
try:
    import pandas as pd  # noqa
except ImportError:
    pd = None

ONE_TIME_MESSAGES = dict()
"""Dictionary to store one time messages."""

_loggers = dict()
"""Dictionary to store loggers for other packages."""

_CONSOLE_WIDTH = 200


def set_console_width(width: int) -> None:
    """Sets the console width.

    Parameters
    ----------
    width : int
        The width to set the console to.
    """
    global _CONSOLE_WIDTH, CONSOLE
    _CONSOLE_WIDTH = width
    CONSOLE.width = width


def get_console_width() -> int:
    """Gets the console width.

    Returns
    -------
    int
        The console width.
    """
    return _CONSOLE_WIDTH


CONSOLE = Console(width=get_console_width())


def get_console() -> Console:
    """Gets the console instance."""
    return CONSOLE


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


def basic_config(level: int = logging.INFO, filename: Optional[str] = None, reinit: bool = False, include_pid: bool = False) -> None:
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

    rh = RichHandler(rich_tracebacks=True, console=get_console())
    if include_pid:
        from tools.logger.rich.log_custom_render import LogCustomRender
        rh._log_render = LogCustomRender(
            show_time=rh._log_render.show_time,
            show_level=rh._log_render.show_level,
            show_path=rh._log_render.show_path,
            time_format=rh._log_render.time_format,
            omit_repeated_times=rh._log_render.omit_repeated_times,
            level_width=None,
            show_pid=include_pid,
        )
    logging.basicConfig(
        # format=_fmt,
        format="%(message)s",
        datefmt=_date_fmt,
        level=level,
        handlers=[rh] + handlers,
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


def log_table(rich_table: Table, width: Optional[int] = None) -> Text:
    """Logs a rich table to the console.

    Parameters
    ----------
    rich_table : Table
        Rich table to log.
    width : Optional[int], optional
        Optional width of the table, by default the console width.

    Returns
    -------
    Text
        Rich Text object containing the table markup.
    """
    width = width if width is not None else get_console_width()
    console = Console(width=width)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get()).markup


def log_dataframe(df: "pd.DataFrame",
                  level: int = logging.INFO,
                  message: Optional[Any] = None,
                  row_limit: int = 20,
                  col_limit: int = 10,
                  first_rows: bool = True,
                  first_cols: bool = True,
                  console: Optional[Console] = None,
                  ) -> None:
    """Logs a dataframe using rich.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to log.
    row_limit : int, optional
        The number of rows to show, by default 20
    col_limit : int, optional
        The number of columns to show, by default 10
    first_rows : bool, optional
        Whether to show the first n rows or the last n rows, by default True
    first_cols : bool, optional
        Whether to show the first n columns or the last n columns, by default True
    console : Console, optional
        The console to use, by default None. If None, the default console is used.
    """
    from rich.highlighter import ReprHighlighter, NullHighlighter
    from tools.rich.dataframe import RichDataFrame

    if message is not None:
        if type(message) in [str, int, float, bool, complex]:
            message = str(message)
        else:
            message = repr(message)
        t = Text(message)
        ReprHighlighter().highlight(t)
        message = t + Text("\n")
    rdf = RichDataFrame(df, row_limit, col_limit,
                        first_rows, first_cols, console)
    tbls = log_table(rdf.table, width=get_console_width())
    logger.log(level, (message.markup if message is not None else Text()) + tbls,
               extra={"markup": True, "highlighter": NullHighlighter()}, stacklevel=2)


def log_only_to_handler(logger: logging.Logger, handler: logging.Handler, level: int, msg: str, *args, exc_info=None, extra=None, stack_info=False, stacklevel=1) -> None:
    """
    Logs a message only to a specific handler.

    Parameters
    ----------
    logger : logging.Logger
        The logger to use.
    handler : logging.Handler
        The handler to log to.
    level : int
        The logging level.
    msg : str
        The message to log.
    """
    record = _make_log_record(logger, level, msg, *args, exc_info=exc_info,
                              extra=extra, stack_info=stack_info, stacklevel=stacklevel)
    handler.handle(record)


def _make_log_record(logger: logging.Logger, level: int, msg: str, *args, exc_info=None, extra=None, stack_info=False, stacklevel=1) -> logging.LogRecord:
    """
    Low-level logging routine which creates a LogRecord.

    Parameters
    ----------
    logger : logging.Logger
        The logger to use.
    level : int
        The logging level.
    msg : str
        The message to log.
    Returns
    -------
    logging.LogRecord
        The created LogRecord.
    """
    sinfo = None
    if logging._srcfile:
        # IronPython doesn't track Python frames, so findCaller raises an
        # exception on some versions of IronPython. We trap it here so that
        # IronPython can use logging.
        try:
            fn, lno, func, sinfo = logger.findCaller(stack_info, stacklevel)
        except ValueError:  # pragma: no cover
            fn, lno, func = "(unknown file)", 0, "(unknown function)"
    else:  # pragma: no cover
        fn, lno, func = "(unknown file)", 0, "(unknown function)"
    if exc_info:
        if isinstance(exc_info, BaseException):
            exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
        elif not isinstance(exc_info, tuple):
            exc_info = sys.exc_info()
    record = logger.makeRecord(logger.name, level, fn, lno, msg, args,
                               exc_info, func, extra, sinfo)
    return record
