from functools import wraps
import os
from pathlib import Path

import re
import subprocess
from typing import Any, Callable, Dict, List, Optional, Set, Union
from tools.util.typing import _NOTSET, NOTSET, _DEFAULT, DEFAULT
from tools.logger.logging import logger

VSCODE_AVAILABLE = NOTSET


def has_vscode() -> bool:
    """
    Checks if vscode is available on the system an on the path.

    Returns
    -------
    bool
        If vscode is available.
    """
    global VSCODE_AVAILABLE
    if VSCODE_AVAILABLE is not NOTSET:
        return VSCODE_AVAILABLE
    try:
        subprocess.run("code --version", shell=True, check=True)
        VSCODE_AVAILABLE = True
    except Exception as err:
        VSCODE_AVAILABLE = False
    return VSCODE_AVAILABLE


def is_headless() -> bool:
    """Checks if the current environment is headless.

    Returns
    -------
    bool
        If the environment is headless.
    """
    from tools.util.format import str_to_bool
    is_headless = os.environ.get("HEADLESS", False)
    if isinstance(is_headless, bool):
        return is_headless
    else:
        return str_to_bool(str(is_headless))


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


def file_local(file_path: str, local_dir_path: str) -> str:
    """Returns the local path of a file.
    W.r.t. the local_dir_path in os independent format.

    Parameters
    ----------
    file_path : str
        The file path to get the local path.

    Returns
    -------
    str
        The local path of the file.
    """
    file_path = os.path.normpath(os.path.abspath(file_path))
    local_dir_path = os.path.normpath(os.path.abspath(local_dir_path))
    rel = relpath(local_dir_path, file_path,
                  is_from_file=False, is_to_file=True)
    return format_os_independent(rel)


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
    if os.path.isfile(path):
        path = os.path.dirname(path)
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


def read_directory(
    path: str,
    pattern: str,
    parser: Optional[Dict[str, Callable]] = None,
    path_key: str = "path"
) -> List[Dict[str, Any]]:
    """Reads a directory for files matching a regex pattern and returns a list of dictionaries with the readed groups and full filepath.

    Parameters
    ----------
    path : str
        The path to read the files from.

    pattern : str
        The regex pattern to match the files.
        Specify named groups to extract the values.

    parser : Optional[Dict[str, callable]], optional
        A parser dictionary which can contain keys which should correspond to named groups in the pattern,
        the value should be a callable which is invoked by the parsed value. The result is then written to the result dictionary, by default None

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with the readed groups and full filepath.

    """
    res = list()
    regex = re.compile(pattern)
    if not os.path.exists(path):
        from tools.logger.logging import logger
        logger.debug(f"Non existing path: {path}")
        return res
    for file in os.listdir(path):
        if pattern[0] == "^" and pattern[-1] == "$":
            match = regex.fullmatch(file)
        else:
            match = regex.search(file)
        if match:
            item = dict(match.groupdict())
            if parser is not None:
                for key, value in item.items():
                    if key in parser:
                        try:
                            item[key] = parser[key](value)
                        except Exception as err:
                            raise ValueError(
                                f"Could not parse value '{value}' for key '{key}' with parser '{parser[key]}' within File '{file}' using the pattern '{pattern}'."
                            ) from err
            _p = format_os_independent(os.path.join(path, file))
            item[path_key] = _p
            res.append(item)
    return res


def read_directory_recursive(
    path: str,
    parser: Optional[Dict[str, Callable]] = None,
    path_key: str = "path",
    recurse_in_matched_subdirs: bool = False,
    max_depth: int = 10,
    memo: Optional[Set[str]] = None,
    _regex_special_chars: Optional[set] = DEFAULT,
) -> List[Dict[str, Any]]:
    """Reads a directory for files matching a regex pattern and returns a
    list of dictionaries with the readed groups and full filepath.

    This is an adapted version of read_directory which allows to specify a subpath as a regex.
    And will recursively search for files matching the pattern.

    Parameters
    ----------
    path : str
        The path to read the files from.
        Can contain a regex pattern with named groups at any point. "/" is used as a separator.
        Must be a path in linux format e.g. call (format_os_independent).

    parser : Optional[Dict[str, Callable]], optional
        A parser dictionary which can contain keys which should correspond to named groups in the pattern,
        the value should be a callable which is invoked by the parsed value. The result is then written to the result dictionary, by default None

    path_key : str, optional
        The key to store the path in the result dictionary, by default "path"

    recurse_in_matched_subdirs : bool, optional
        If matched subdirectories should be recursively searched, by default False

    max_depth : int, optional
        The maximum depth to search for subdirectories, by default 10

    memo : Optional[Set[str]], optional
        A memo set to keep track of already visited directories, by default None

    _regex_special_chars : Optional[set], optional
        A set of special regex characters, by default DEFAULT
        Determines which characters are considered special, tests each subdirectory for these characters and evaluates as regex if contain at least one of the chars.
        If DEFAULT is used, the default set of special characters is used.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with the readed groups and full filepath.

    """
    if memo is None:
        memo = set()
    abspath = os.path.abspath(os.path.normpath(path))
    if abspath in memo:
        return []
    memo.add(abspath)

    directory = path.split("/")
    # Check to what extend the path is a regex
    p = []

    if _regex_special_chars is DEFAULT:
        special_chars = set(list(regex_special_chars()))
        special_chars.remove('.')
    else:
        special_chars = _regex_special_chars

    pattern_start = -1
    for i, d in enumerate(directory):
        if d == "." or d == "..":
            p.append(d)
            continue
        if set(d).intersection(special_chars):
            pattern_start = i
            break
        else:
            p.append(d)
    if len(p) == len(directory):
        # No regex in path, match all files
        return read_directory(path, ".*", parser, path_key)

    path = "/".join(p)
    next_patterns = directory[pattern_start:]
    next_pattern = next_patterns.pop(0)
    results = read_directory(path, next_pattern, parser, path_key)
    if len(next_patterns) == 0 and not recurse_in_matched_subdirs:
        return results
    else:
        if recurse_in_matched_subdirs and len(next_patterns) == 0:
            super_results = []
            if max_depth == 0 or max_depth < -1:
                return results
            else:
                if max_depth != -1:
                    max_depth -= 1
            for result in results:
                sub_path = result.get(path_key)
                if os.path.isfile(sub_path):
                    super_results.append(result)
                    continue
                result.pop(path_key)
                # Reusing last-pattern on subdirs
                new_path = sub_path + "/" + next_pattern
                rec_results = read_directory_recursive(
                    new_path, parser, path_key,
                    recurse_in_matched_subdirs=recurse_in_matched_subdirs,
                    max_depth=max_depth,
                    memo=memo,
                    _regex_special_chars=_regex_special_chars)
                for rec_result in rec_results:
                    rec_result.update(result)
                    super_results.append(rec_result)
            return super_results
        else:
            super_results = []
            for result in results:
                sub_path = result.pop(path_key)
                patterns = "/".join(next_patterns)
                # Check if subpath is a directory
                if not os.path.isdir(sub_path):
                    continue
                new_path = sub_path + "/" + patterns
                rec_results = read_directory_recursive(
                    new_path,
                    parser, path_key,
                    recurse_in_matched_subdirs=recurse_in_matched_subdirs,
                    memo=memo,
                    _regex_special_chars=_regex_special_chars)
                for rec_result in rec_results:
                    rec_result.update(result)
                    super_results.append(rec_result)
            return super_results

    return read_directory(path, pattern, parser, path_key)


def open_in_default_program(
        path_to_file: Union[str, Path],
        headless: Union[bool, _DEFAULT] = DEFAULT
) -> None:
    """Opens the given file in the systems default program.

    Parameters
    ----------
    path_to_file : str
        Path to open in default program.
    """
    from sys import platform
    if headless is DEFAULT:
        headless = is_headless()
    path_to_file: Path = process_path(
        path_to_file, need_exist=False, variable_name="path_to_file")
    path_to_file = path_to_file.resolve()
    if path_to_file.exists():
        if not headless:
            if platform == "linux" or platform == "linux2":
                # linux
                subprocess.run(f"xdg-open {path_to_file}")
            elif platform == "darwin":
                # OS X
                if has_vscode():
                    subprocess.run(f"code {path_to_file} -r", shell=True)
                else:
                    logger.warning(
                        "Default program opening is not supported on MacOS. Consider an implementation.")
            elif platform == "win32":
                # Windows...
                subprocess.run(f"powershell {path_to_file}")
        else:
            if has_vscode():
                subprocess.run(f"code {path_to_file} -r", shell=True)
            else:
                logger.warning(
                    "Could not open file in default program, vscode is not available and program is running in headless mode.")


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
    raise ValueError(
        f"Could not find free path within max checks of: {max_check}!")


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
    raise ValueError(
        f"Could not find free path within max checks of: {max_check}!")


def replace_unallowed_chars(path: str,
                            allow_dot: bool = True,
                            allow_space: bool = True,
                            replace_with: str = "_") -> str:
    """Replaces unallowed characters in a path with a given character.

    Parameters
    ----------
    path : str
        The path to replace the characters.

    allow_dot : bool, optional
        If dots should be allowed, by default True

    replace_with : str, optional
        The character to replace the unallowed characters, by default "_"

    Returns
    -------
    str
        The path with replaced characters.
    """
    unallowed = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]
    if not allow_dot:
        unallowed.append(".")
    if not allow_space:
        unallowed.append(" ")
    for char in unallowed:
        path = path.replace(char, replace_with)
    return path


def regex_special_chars() -> str:
    """Returns a string containing all special regex characters.

    Returns
    -------
    str
        String containing all special regex characters.
    """
    return r".^$*+?{}[]\|()"


def replace_file_unallowed_chars(file_name: str, replace_with: str = "_") -> str:
    """Replaces unallowed characters in a file name with a given character.

    Parameters
    ----------
    file_name : str
        File name to replace the characters.

    replace_with : str, optional
        The character to replace the unallowed characters, by default "_"

    Returns
    -------
    str
        The file name with replaced characters.
    """
    base_name, ext = file_name.split(".")[:-1], file_name.split(".")[-1]
    base_name = replace_unallowed_chars(
        replace_with.join(base_name), replace_with=replace_with)
    file_name = f"{base_name}.{ext}"
    return file_name


def process_path(
    path: Union[str, Path],
    need_exist: bool = False,
    make_exist: bool = False,
    allow_none: bool = False,
    interpolate: bool = False,
    interpolate_object: Optional[object] = None,
    variable_name: Optional[str] = None,
    reevaluate: bool = False,
) -> Optional[Path]:
    """Preprocesses a path string or Path object.

    Can interpolate certain values into the path string, based on the interpolate string
    and with the parse_format_string function.

    Parameters
    ----------
    path : str | Path
        Path to preprocess, may contain placeholders for interpolation.
    need_exist : bool, optional
        If the path need to exist beforehand, by default False
    make_exist : bool, optional
        If the path should be created, by default False
    allow_none : bool, optional
        If calling with an empty path is allowed, otherwise raise an error, by default False
    interpolate : bool, optional
        If path interpolation should be carried out, by default False
    interpolate_object : Optional[object], optional
        The object to lookup for when having interpolation active, by default None
    variable_name : Optional[str], optional
        The current variable name to display in errors, by default None
    reevaluate : bool, optional
        If the path is a ContextPath, reevaluate it with the given context, by default False.
    Returns
    -------
    Optional[Path]
        The processed path as Path object.
    """
    from tools.util.format import parse_format_string
    from tools.util.path import Path as ToolsPath
    from tools.serialization.files.context_path import ContextPath

    def _checks(p: Path):
        if need_exist and not p.exists():
            raise FileNotFoundError(
                f"Path {'for ' + variable_name + ' ' if variable_name is not None else ''}{p} does not exist.")
        if make_exist and not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        return p
    if path is None:
        if allow_none:
            return None
        else:
            raise ValueError(
                f"Path {'for ' + variable_name + ' ' if variable_name is not None else ''}must be set.")
    if isinstance(path, Path):
        return _checks(path)
    elif isinstance(path, ToolsPath):
        if reevaluate and isinstance(path, ContextPath):
            path = path.reevaluate(interpolate_object)
        pth = _checks(path)
        return pth
    elif not isinstance(path, str):
        raise ValueError(
            f"Path {'for ' + variable_name + ' ' if variable_name is not None else ''}must be a string or Path object.")
    if interpolate:
        p = ContextPath.from_format_path(path, context=interpolate_object)
    else:
        p = Path(path).resolve()
    return _checks(p)


def filer(
    default_ext: str,
    default_output_dir: Optional[str] = "./temp",
    path_param: Optional[str] = "path",
):
    """Decorator to handle file functions which return paths and optionally open them.

    Can also automatically create a temporary path if no path is given, and forward it to the function.

    Parameters
    ----------
    default_ext : str
        The default extension for the file.
        E.g. "mp4", "png", "jpg", etc.

    default_output_dir : Optional[str], optional
        The default output directory for the file, by default "./temp"

    path_param : Optional[str], optional
        The name of the parameter which is the path, by default "path"
        Must be supplied.
    """
    from uuid import uuid4

    # type: ignore
    def decorator(function: Callable[[Any], Union[str, Path]]) -> Callable[[Any], Union[str, Path]]:
        # Get which is the positional paramter corresponding to the path
        nonlocal default_ext
        import inspect

        sig = inspect.signature(function)
        path_param_index = None
        for i, param in enumerate(sig.parameters):
            if param == path_param:
                path_param_index = i
                break
        if path_param_index is None:
            raise ValueError(
                f"Could not find path parameter {path_param} in function signature. Please specify the correct parameter name as path_param.")

        # Check extension
        if default_ext.startswith("."):
            default_ext = default_ext[1:]
        # If the default extension contains a dot raise an error
        if "." in default_ext:
            raise ValueError(
                f"Default extension {default_ext} should not contain a dot.")

        def get_path(args, kwargs) -> Union[str, Path, _NOTSET]:
            if path_param_index < len(args):
                return args[path_param_index]
            if path_param in kwargs:
                return kwargs[path_param]
            return NOTSET

        @wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal default_ext
            open = kwargs.pop("open", False)
            ext = kwargs.pop("ext", default_ext)

            path = get_path(args, kwargs)
            if path is NOTSET:
                # Create a temporary path
                base = str(uuid4())
                path = os.path.join(default_output_dir, base + "." + ext)
                # Push the path in kwargs
                kwargs[path_param] = path

            try:
                out = function(*args, **kwargs)
            finally:
                pass

            is_path = isinstance(out, (str, Path))
            path = None
            if is_path:
                path = out if isinstance(out, Path) else Path(out)

            if is_path and open:
                try:
                    open_in_default_program(path)
                except Exception as err:
                    pass
            return out
        return wrapper
    return decorator


def is_path_only_basename(path: str) -> bool:
    """Checks if the given path is only a basename.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    bool
        If the path is only a basename.
    """
    return any([(x in path) for x in ["/", "\\", os.sep]])


def display_path(
        path: str,
        max_length: int = 30,
        filename_replace: str = "[...]",
        path_replace: str = "**"
) -> str:
    """Displays a path with a maximum length and adds
    "[... / ...]" for more directories than the first or [...] to shorten the basename.

    Parameters
    ----------
    path : str
        The path to display.
    max_length : int
        The maximum length of the path.
    filename_replace : str, optional
        The string to replace to long parts of the filename with, by default "[...]"
    path_replace : str, optional
        The string to replace to long parts of the path with, by default "**"

    Returns
    -------
    str
        The displayed path.
    """
    if len(path) <= max_length:
        return path
    parts = path.split(os.sep)
    filename = parts[-1]
    max_path_filename_length = max_length - len(filename_replace)
    if len(filename) >= max_length:
        # If already the filename is too long
        fn, ext = os.path.splitext(filename)
        mle = max_path_filename_length - len(ext)
        # Check if it has a path
        if len(parts) > 1:
            mle -= len(path_replace)
            filename = path_replace + fn[:mle] + filename_replace + ext
            return filename
        else:
            if len(filename) == max_length:
                return filename
            return fn[:mle] + filename_replace + ext
    else:
        # Try to preserve the filename and as much as starting and ending path components, shorten the middle
        max_path_length = max_length - len(filename)
        trimmed_path = None
        org_parts = parts
        parts = parts[:-1]

        for i in range(-1, -(len(parts)), -1):
            if i == -1:
                if len(os.path.join(*parts)) < max_path_length:
                    trimmed_path = os.path.join(parts)
                    break
            pts = parts[:i]
            assembled_path = None
            if len(pts) == 1:
                assembled_path = pts[0]
            else:
                assembled_path = os.path.join(*pts)

            if (len(assembled_path) + len(path_replace) + 2 * len(os.sep)) <= max_path_length:
                trimmed_path = os.path.join(assembled_path, path_replace)
                break

        if trimmed_path is None:
            trimmed_path = path_replace
        return os.path.join(trimmed_path, filename)
