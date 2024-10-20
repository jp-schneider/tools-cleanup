from traceback import FrameSummary
from tools.config.output_config import OutputConfig
from typing import Optional
import os
from tools.logger.logging import logger
from tools.util.format import get_frame_summary
EXCEPTION_ERROR_CODES = {
    KeyboardInterrupt: 130,
    SyntaxError: 2,
    Exception: 1,
}

DETAILED_EXCEPTION_HANDLING = {
    RuntimeError
}


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


def write_exit(config: OutputConfig, exit_code: int, err: Optional[Exception] = None):
    try:
        # Write exit code within a text file called exit_{exit_code}.txt
        path = os.path.join(config.output_folder, f"exit_{exit_code:03d}.txt")
        # Content will be the exception message if exit code is not 0, on success Add a success message
        content = f"Exit code: {exit_code}"
        if exit_code != 0:
            import traceback
            if err is not None:
                content += f"\n{type(err)}: {str(err)}"
                content += f"\nStacktrace:\n{''.join(traceback.format_exception(err))}"
        with open(path, "w") as f:
            f.write(content)
    except Exception as err:
        return

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_code = get_exit_code(exc_val)
        if exc_val is not None:
            # Error occurred
            if not isinstance(exc_val, KeyboardInterrupt) or self.log_on_keyboard_interrupt:
                filename = os.path.basename(self.scope.filename).split(".")[0]
                logger.exception(f"Raised {type(exc_val).__name__} in {filename}, exiting...")
        write_exit(self.config, exit_code, exc_val)
        return False