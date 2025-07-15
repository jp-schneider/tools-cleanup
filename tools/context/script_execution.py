from traceback import FrameSummary
from tools.config.output_config import OutputConfig
from tools.config.experiment_output_config import ExperimentOutputConfig
from typing import Optional
import os
from tools.logger.logging import logger
from tools.util.format import get_frame_summary
import pandas as pd
from tools.util.path_tools import read_directory
from datetime import datetime
from dataclasses import dataclass, field
import re

EXCEPTION_ERROR_CODES = {
    KeyboardInterrupt: 130,
    SyntaxError: 2,
    Exception: 1,
}

DETAILED_EXCEPTION_HANDLING = {
    RuntimeError
}

time_regex = r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])(T(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)(?P<fraction>\.\d{1,6})?(?P<timezone>Z|(?P<tz_offset_sign>[+-])(?P<tz_offset_hour>[01]\d|2[0-3])(?::(?P<tz_offset_minute>[0-5]\d))?)?)?$"
exit_code_message_regex = (
    r"^Exit code: (?P<exit_code>\d+)(\. Date: " + time_regex + r")?(\n(?P<message>.*))?$"
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
        return cls(message=message, exit_code=code, datetime=dt)


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


def write_exit(config: OutputConfig, exit_code: int, err: Optional[Exception] = None, exit_time: Optional[datetime] = None):
    if exit_time is None:
        exit_time = datetime.now().astimezone()
    if config is None or not hasattr(config, "output_folder"):
        return
    try:
        # Write exit code within a text file called exit_{exit_code}.txt
        path = os.path.join(config.output_folder, f"exit_{exit_code:03d}.txt")
        # Content will be the exception message if exit code is not 0, on success Add a success message
        content = f"Exit code: {exit_code}. Date: {exit_time}\n"
        if exit_code != 0:
            import traceback
            if err is not None:
                content += f"\n{type(err)}: {str(err)}"
                content += f"\nStacktrace:\n{''.join(traceback.format_exception(err))}"
        with open(path, "w") as f:
            f.write(content)
    except Exception as err:
        return


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
                 ):
        self.config = config
        self.scope = get_frame_summary(1)
        self.log_on_keyboard_interrupt = log_on_keyboard_interrupt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_code = get_exit_code(exc_val)
        if exc_val is not None:
            # Error occurred
            if not isinstance(exc_val, KeyboardInterrupt) or self.log_on_keyboard_interrupt:
                filename = os.path.basename(self.scope.filename).split(".")[0]
                logger.exception(
                    f"Raised {type(exc_val).__name__} in {filename}, exiting...")
                logger.info(
                    f"Exception occured, Context Terminated with Exit Code: {exit_code}")
        else:
            # No error occurred
            logger.info(
                f"Script Context executed successfully. Exit Code: {exit_code}")
        write_exit(self.config, exit_code, exc_val)
        try:
            if isinstance(self.config, ExperimentOutputConfig):
                if self.config.experiment_logger == "wandb":
                    import wandb
                    wandb.finish(exit_code=exit_code)
        except Exception as e:
            logger.warning(f"Failed to finish wandb run: {e}")
        return False
