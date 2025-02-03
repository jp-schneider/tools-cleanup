import os
from typing import Optional
from tools.error import ArgumentNoneError


class AlterWorkingDirectory():
    """Context manager for altering the working directory temporary."""

    def __init__(self, work_dir: str):
        """Creates the context manager with the new temporary working directory.

        Parameters
        ----------
        work_dir : str
            The new temporary working directory.

        Raises
        ------
        ArgumentNoneError
            If work_dir is None.
        ValueError
            If work_dir is not a directory.
        """
        if work_dir is None:
            raise ArgumentNoneError("work_dir")
        if not os.path.isdir(work_dir):
            raise ValueError(f"work_dir: {work_dir} is not a directory!")
        self.work_dir = work_dir
        self.old_dir: Optional[str] = None

    def __enter__(self):
        self.old_dir = os.getcwd()
        os.chdir(self.work_dir)

    def __exit__(self, type, value, traceback):
        if self.old_dir is not None:
            os.chdir(self.old_dir)
        return False
