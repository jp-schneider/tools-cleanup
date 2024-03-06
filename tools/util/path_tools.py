import os
import re
import subprocess


def relpath(from_: str, to: str, is_from_file: bool = True, is_to_file: bool = True) -> str:
    """Returns a relative path from from_ to to. Where _from and _to can be either a file or a directory.
    Simple wrapper around os.path.relpath which also handles files.

    Parameters
    ----------
    from_ : str
        From path which is a file or directory.
    to : str
        To path which is a file or directory.
    is_from_file : bool, optional
        Boolean argument specifying if from_ is a file (True) or directory (False), by default True
    is_to_file : bool, optional
        Boolean argument specifying if to is a file (True) or directory (False), by default True

    Returns
    -------
    str
        The relative path from from_ to to.
    """
    _to_dir = os.path.dirname(to) if is_to_file else to
    _from_dir = os.path.dirname(from_) if is_from_file else from_
    path = os.path.relpath(_to_dir, _from_dir)
    if is_to_file:
        return os.path.join(path, os.path.basename(to))
    else:
        return path

def format_os_independent(path: str) -> str:
    """Formats a path to be os independent.
    Replaces all backslashes with forward slashes.

    Parameters
    ----------
    path : str
        The resulting os invariant path.

    Returns
    -------
    str
        The os invariant path.
    """
    return path.replace("\\", "/")


def open_folder(path: str) -> None:
    """Opens the given path in the systems file explorer.

    Parameters
    ----------
    path : str
        Path f open in explorer.
    """
    from sys import platform
    path = os.path.abspath(os.path.normpath(path))
    if os.path.exists(path):
        if platform == "linux" or platform == "linux2":
            # linux
            raise NotImplementedError()
        elif platform == "darwin":
            # OS X
            raise NotImplementedError()
        elif platform == "win32":
            # Windows...
            subprocess.run(f"explorer {path}")


def open_in_default_program(path_to_file: str) -> None:
    """Opens the given file in the systems default program.

    Parameters
    ----------
    path_to_file : str
        Path to open in default program.
    """
    from sys import platform
    path = os.path.abspath(os.path.normpath(path_to_file))
    if os.path.exists(path):
        if platform == "linux" or platform == "linux2":
            # linux
            raise NotImplementedError()
        elif platform == "darwin":
            # OS X
            raise NotImplementedError()
        elif platform == "win32":
            # Windows...
            subprocess.run(f"powershell {path_to_file}")


def numerated_file_name(path: str, max_check: int = 1000) -> str:
    """Checks whether the given path exists, if so it will try 
    to evaluate a free path by appending a consecutive number.

    Parameters
    ----------
    path : str
        The path to check
    max_check : int, optional
        How much files should be checked until an error is raised, by default 1000

    Returns
    -------
    str
        A Path which is non existing.

    Raises
    ------
    ValueError
        If max_check is reached.
    """
    PATTERN = r" \((?P<number>[0-9]+)\)$"
    pattern = re.compile(PATTERN)
    i = 2
    for _ in range(max_check):
        if os.path.exists(path):
            directory = os.path.dirname(path)
            extension = os.path.splitext(path)[1]
            name = os.path.basename(path).replace(extension, '')
            match = pattern.match(name)
            if i == 2 and match is None:
                name += f' ({i})'
            else:
                if match:
                    number = match.group["number"]
                name = re.sub(pattern, repl=f' ({i})', string=name)
            path = os.path.join(directory, name + extension)
            i += 1
        else:
            return path
    raise ValueError(f"Could not find free path within max checks of: {max_check}!")



def numerated_folder_name(path: str, max_check: int = 1000) -> str:
    """Checks whether the given folder path exists, if so it will try 
    to evaluate a free path by appending a consecutive number.

    Parameters
    ----------
    path : str
        The path to check
    max_check : int, optional
        How much files should be checked until an error is raised, by default 1000

    Returns
    -------
    str
        A Path which is non existing.

    Raises
    ------
    ValueError
        If max_check is reached.
    """
    PATTERN = r" \((?P<number>[0-9]+)\)$"
    pattern = re.compile(PATTERN)
    i = 2
    for _ in range(max_check):
        if os.path.exists(path):
            directory = os.path.dirname(path)
            name = os.path.basename(path)
            match = pattern.match(name)
            if i == 2 and match is None:
                name += f' ({i})'
            else:
                if match:
                    number = match.group["number"]
                name = re.sub(pattern, repl=f' ({i})', string=name)
            path = os.path.join(directory, name)
            i += 1
        else:
            return path
    raise ValueError(f"Could not find free path within max checks of: {max_check}!")
