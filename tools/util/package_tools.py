import os
from typing import Any, Callable, Dict, Optional
import inspect

EXECUTED_MODULE_PATH = None
"""Stores Module path of the currently executable package. If the package should not be the tools package, must be modified via the set_module_path function. Use get_module_path to read the value."""

try:
    import toml
except (ImportError, ModuleNotFoundError):
    toml = None


def get_module_path() -> str:
    """Gets the module path of the currently executable package.

    If the module path is not set, it will be set to the path of the package which contains this file.

    Returns
    -------
    str
        The module path of the currently executable package.
    """
    global EXECUTED_MODULE_PATH
    if EXECUTED_MODULE_PATH is None:
        EXECUTED_MODULE_PATH = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
    return EXECUTED_MODULE_PATH


def get_executed_module_root_path() -> str:
    """Gets the root path of the currently executed module.
    The root path is defined as the directory where the module is located.

    Returns
    -------
    str
        The root path of the currently executed module.
    """
    return os.path.abspath(os.path.join(get_module_path(), ".."))


def get_project_root_path() -> str:
    """Gets the root path of the current project.
    A project root path is defined as the directory where the
    pyproject.toml file is located.

    Returns
    -------
    str
        The absolute root path of the project.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_package_root_path() -> str:
    """Gets the package path of the project.
    A package root path is defined as the directory where the
    source code of the project is located.
    In this case it is the directory where the awesome package is located.

    Returns
    -------
    str
        The absolute package root path of the project.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_invoked_package_name() -> str:
    """Gets the package name of the package which invoked this function.
    The package name is defined as the name of the package.

    Returns
    -------
    str
        The package name of the package which invoked this function.
    """
    path = get_module_path()
    root_path = os.path.abspath(os.path.join(path, ".."))
    package_name = get_package_name(project_root_path=root_path)
    return package_name


def set_module_path(module_path: str) -> None:
    """Sets the module path of the currently executable package.

    Parameters
    ----------
    module_path : str
        The path of the primary module root, which contains all implementations for a certain package. Parent folder should contain the the pyproject.toml file is located.
    """
    global EXECUTED_MODULE_PATH
    EXECUTED_MODULE_PATH = os.path.normpath(module_path)


def set_exec_module(module: Any) -> None:
    """Sets the module path of the currently executable package.

    Parameters
    ----------
    module : Any
        The module which is currently executed.
    """
    set_module_path(os.path.join(module.__file__, ".."))


def is_project_root(path: str) -> bool:
    """Checks if the given path is a project root.
    A project root is defined as a directory which contains a pyproject.toml file.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    bool
        True if the path is a package root, False otherwise.
    """
    return os.path.isfile(os.path.join(path, "pyproject.toml"))


def get_package_info(project_root_path: Optional[str] = None) -> Dict[str, Any]:
    """Gets the package information of the project.
    The package information is defined as the package name, version and description.

    Parameters
    ----------
    project_root_path : str, optional
        The root path of the project, by default None

    Returns
    -------
    dict
        The package information of the project.
    """
    if toml is None:
        raise ImportError(
            "The package 'toml' is not installed. Please install it to use this function.")
    project_root_path = project_root_path if project_root_path is not None and is_project_root(
        project_root_path) else get_project_root_path()
    pyproject_path = os.path.join(project_root_path, "pyproject.toml")
    with open(pyproject_path, "r") as file:
        pyproject = toml.load(file)
    package_info = pyproject["tool"]["poetry"]
    return package_info


def get_package_name(project_root_path: Optional[str] = None) -> str:
    """Gets the package name of the project.
    The package name is defined as the name of the package.

    Returns
    -------
    str
        The package name of the project.
    """
    package_info = get_package_info(project_root_path=project_root_path)
    package_name = package_info["name"]
    return package_name
