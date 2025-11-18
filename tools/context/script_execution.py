from logging import Handler
import logging
from traceback import FrameSummary
from tools.config.output_config import OutputConfig
from tools.config.experiment_output_config import ExperimentOutputConfig
from typing import Optional, Tuple
import os
from tools.logger.logging import logger, log_only_to_handler
from tools.util.format import get_frame_summary
import pandas as pd
from tools.util.path_tools import read_directory
from datetime import datetime
from dataclasses import dataclass, field
import re
from tools.util.format import strfdelta
import sys
import io
EXCEPTION_ERROR_CODES = {
    KeyboardInterrupt: 130,
    SyntaxError: 2,
    Exception: 1,
}

DETAILED_EXCEPTION_HANDLING = {
    RuntimeError
}

time_regex = r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])([\s|T](?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)(\.(?P<fraction>\d{1,6}))?(?P<timezone>Z|(?P<tz_offset_sign>[+-])(?P<tz_offset_hour>[01]\d|2[0-3])(?::(?P<tz_offset_minute>[0-5]\d))?)?)?$"
exit_code_message_regex = (
    r"^Exit code: (?P<exit_code>\d+)(\. Date: " +
    time_regex.lstrip('^').rstrip("$") + r")?(\n(?P<message>(.|\s)*))$"
)

running_file_pattern = r"^running_(?P<pid>\d+).txt$"
running_message_regex = (
    r"^Start-Date: (?P<start_date>" + time_regex.lstrip('^').rstrip("$") + r")" + os.linesep +
    r"(?P<message>.*)?$"
)


@dataclass
class ExitMessage:
    """Dataclass to hold the exit message and its path."""

    message: str
    """Exit message content."""

    exit_code: int
    """Exit code"""

    exit_time: Optional[datetime] = None
    """Datetime when the exit message was created."""

    @classmethod
    def from_file(cls, path: str) -> Optional["ExitMessage"]:
        """Load an exit message from a file.

        Parameters
        ----------
        path : str
            Path to the exit message file.

        Returns
        -------
        Optional[ExitMessage]
            An instance of ExitMessage if the file exists and is valid, otherwise None.
        """
        if not os.path.exists(path):
            return None
        content = None
        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            return None
        # Parse the exit code and datetime from the content
        m = re.match(exit_code_message_regex, content)
        code = int(m.group("exit_code"))
        # Check if the datetime is valid
        dt = None
        if "year" in m.groupdict():
            dt = datetime(
                year=int(m.group("year")),
                month=int(m.group("month")),
                day=int(m.group("day")),
                hour=int(m.group("hour")) if m.group("hour") else 0,
                minute=int(m.group("minute")) if m.group("minute") else 0,
                second=int(m.group("second")) if m.group("second") else 0,
            )
        message = m.group("message") if m.group("message") else ""
        return cls(message=message, exit_code=code, exit_time=dt)


def detailed_exception(err: Exception) -> int:
    if isinstance(err, RuntimeError):
        if "CUDA error: out of memory" in str(err):
            return 125


def get_exit_code(err: Optional[Exception]) -> int:
    if err is None:
        return 0
    for exception, code in EXCEPTION_ERROR_CODES.items():
        if isinstance(err, exception):
            if exception in DETAILED_EXCEPTION_HANDLING:
                return detailed_exception(err)
            return code
    return 1


def write_exit(config: OutputConfig, exit_code: int, err: Optional[Exception] = None, exit_time: Optional[datetime] = None, message: Optional[str] = None) -> Optional[str]:
    if exit_time is None:
        exit_time = datetime.now().astimezone()
    if config is None or not hasattr(config, "output_folder"):
        return None
    path = None
    try:
        # Write exit code within a text file called exit_{exit_code}.txt
        path = os.path.join(config.output_folder, f"exit_{exit_code:03d}.txt")
        # Content will be the exception message if exit code is not 0, on success Add a success message
        content = f"Exit code: {exit_code}. Date: {exit_time}" + os.linesep
        if exit_code != 0:
            import traceback
            if err is not None:
                content += f"{os.linesep}{type(err)}: {str(err)}"
                content += f"{os.linesep}Stacktrace:{os.linesep}{''.join(traceback.format_exception(err))}"
        if message is not None:
            content += f"Message:{os.linesep}{message}"
        with open(path, "w") as f:
            f.write(content)
    except Exception as err:
        return path
    return path


def write_running(config: OutputConfig, start_time: Optional[datetime] = None, message: str = "Running...") -> Optional[str]:
    """Writes a running message to a file in the output folder."""
    if start_time is None:
        start_time = datetime.now().astimezone()
    if config is None or not hasattr(config, "output_folder"):
        return None
    if config.output_folder is None:
        logger.warning("Output folder is None, can not write running message.")
        return None
    try:
        path = os.path.join(config.output_folder, f"running_{os.getpid()}.txt")
        # Create path if it does not exist
        if not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
        content = f"Start-Date: {start_time}{os.linesep}" + \
            f"Message: {message}{os.linesep}"
        with open(path, "w") as f:
            f.write(content)
    except Exception as err:
        logger.error(f"Failed to write running message: {err}")
        return None
    return path


def register_log_file(config: OutputConfig,
                      file_path: Optional[str] = "log/log_{year}_{month}_{day}__{hour}_{minute}_{second}__{pid}_{script}.log",
                      use_html: bool = False

                      ) -> Optional[Handler]:
    """Registers a log file in the output folder.

    Parameters
    ----------
    config : OutputConfig
        Config containing the output folder.
    file_path : Optional[str], optional
        Path to the log file, by default "log/log_{year}_{month}_{day}__{hour}_{minute}_{second}__{pid}_{sys.executable}.log"
    use_html : bool, optional
        Whether to use HTML formatting for the log file, by default False.
        If True, the log file will be saved as .html, otherwise as .log.
        If html is used, colors and styles will be preserved. If plain text is used, colors and styles will be removed, while basic formatting will be kept.
    Returns
    -------
    Optional[Handler]
        The log handler (if rich logging is used otherwise none).
    """
    from tools.util.format import parse_format_string
    handler = None
    full_path = None
    if config is None or not hasattr(config, "output_folder"):
        return None
    if config.output_folder is None:
        logger.warning("Output folder is None, can not register log file.")
        return handler
    try:
        now = datetime.now().astimezone()
        if file_path is None:
            file_path = "log/log_{year}_{month}_{day}__{hour}_{minute}_{second}__{pid}_{sys.executable}.log"
        items = dict(
            year=now.year,
            month=f"{now.month:02d}",
            day=f"{now.day:02d}",
            hour=f"{now.hour:02d}",
            minute=f"{now.minute:02d}",
            second=f"{now.second:02d}",
            pid=os.getpid(),
            script=os.path.basename(sys.argv[0]).replace(".", "_")
        )
        fp = parse_format_string(
            file_path, [config], additional_variables=items)[0]
        full_path = os.path.join(config.output_folder, fp)
        # Create directory if it does not exist
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # Add file handler to the logger
        from tools.logger.logging import get_console
        try:
            from tools.rich.rich_file_handler import RichFileHandler
        except ImportError:
            RichFileHandler = None
        if RichFileHandler is None:
            handler = logging.FileHandler(full_path)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        else:
            if use_html:
                full_path = full_path.replace(".log", ".html")
            handler = RichFileHandler(
                filename=full_path, rich_tracebacks=True, log_time_format="%Y-%m-%d %H:%M:%S")
        logger.addHandler(handler)
        return handler
    except Exception as err:
        logger.exception(f"Failed to register log file: {err}")
        return handler


def unregister_log_file(handler: Handler) -> None:
    """Unregisters a log file from the logger.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    """
    from tools.logger.logging import logger
    handlers = logger.handlers
    if handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def load_exit_codes(directory: str) -> pd.DataFrame:
    """Loads the path to all exit code files in a given directory.

    Parameters
    ----------
    directory : str
        Directory to search for exit code files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the exit codes and their corresponding file paths.
        Columns are 'code' and 'path'.
    """
    pattern = r"exit_(?P<code>[0-9]+).txt"
    files = read_directory(directory, pattern=pattern, parser=dict(code=int))
    return pd.DataFrame(files)


def load_exit_message(path: str) -> Optional[ExitMessage]:
    """Loads the exit message from a file.

    Parameters
    ----------
    path : str
        Path to the exit message file.

    Returns
    -------
    Optional[ExitMessage]
        An instance of ExitMessage if the file exists and is valid, otherwise None.
    """
    return ExitMessage.from_file(path)


class ScriptExecution:
    """Context manager which encapsulates the execution of a script and stores success or failure in a file."""

    config: OutputConfig
    """Config for the output."""

    scope: FrameSummary
    """Scope of the script execution."""

    def __init__(self,
                 config: OutputConfig,
                 log_on_keyboard_interrupt: bool = False,
                 create_log_file: bool = True,
                 use_html_log: bool = False
                 ):
        self.config = config
        self.scope = get_frame_summary(1)
        self.log_on_keyboard_interrupt = log_on_keyboard_interrupt
        self.start_date = None
        self.end_date = None
        self.running_file_path = None
        self.create_log_file = create_log_file
        self.log_handler = None
        self.use_html_log = use_html_log

    def __enter__(self):
        self.start_date = datetime.now().astimezone()
        # Write running message to the output folder
        self.running_file_path = write_running(
            self.config, start_time=self.start_date)
        if self.create_log_file:
            self.log_handler = register_log_file(
                self.config, use_html=self.use_html_log)
            # Create entry log message
            log_only_to_handler(logger, handler=self.log_handler, level=logging.INFO,
                                msg=f"Script Execution Context started for {self.scope.filename} at {self.start_date}")
            log_only_to_handler(logger, handler=self.log_handler, level=logging.INFO,
                                msg=f"Invocation:{os.linesep}{sys.executable} {' '.join(sys.argv)}")
            yaml = self.config.to_yaml(no_uuid=True, no_large_data=True)
            log_only_to_handler(logger, handler=self.log_handler,
                                level=logging.INFO, msg=f"Configuration:{os.linesep}{yaml}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_date = datetime.now().astimezone()
        self.end_date = end_date
        exit_code = get_exit_code(exc_val)
        if exc_val is not None:
            # Error occurred
            if not isinstance(exc_val, KeyboardInterrupt) or self.log_on_keyboard_interrupt:
                filename = os.path.basename(self.scope.filename).split(".")[0]
                logger.exception(
                    f"Raised {type(exc_val).__name__} in {filename}, exiting...")
                logger.info(
                    f"Exception occured, context terminated with exit code: {exit_code}")
        else:
            # No error occurred
            logger.info(
                f"Script Context executed successfully. Exit Code: {exit_code}")
        # Try to remove the running file
        if self.running_file_path is not None and os.path.exists(self.running_file_path):
            try:
                os.remove(self.running_file_path)
                self.running_file_path = None
            except Exception as e:
                logger.warning(f"Failed to remove running file: {e}")
        # Create a message containing start and end date, and the timedelta
        message = None
        if self.start_date is not None and self.end_date is not None:
            message = (f"Start-Date: {self.start_date}" + os.linesep +
                       f"End-Date: {self.end_date}" + os.linesep +
                       f"Duration: {strfdelta(self.end_date - self.start_date)}")
        write_exit(self.config, exit_code, exc_val,
                   exit_time=self.end_date, message=message)
        try:
            if isinstance(self.config, ExperimentOutputConfig):
                if self.config.experiment_logger == "wandb":
                    import wandb
                    wandb.finish(exit_code=exit_code)
        except Exception as e:
            logger.warning(f"Failed to finish wandb run: {e}")

        # Exit the file logging
        if self.create_log_file or (self.log_handler is not None):
            unregister_log_file(self.log_handler)
            self.log_handler = None
        return False
