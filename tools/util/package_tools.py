import os
from typing import Any, Callable, Dict, Optional
import inspect

EXECUTED_MODULE_PATH = None
"""Stores Module path of the currently executable package. If the package should not be the tools package, must be modified via the set_module_path function. Use get_module_path to read the value."""

try:
    import toml
except (ImportError, ModuleNotFoundError):
    toml = None


def current_module_path() -> str:
    """Gets the path of the current module.
    The current module is defined as the module which contains this file.

    Returns
    -------
    str
        The path of the current package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def is_installed_module(module_path: Optional[str] = None, is_root: bool = False) -> bool:
    """Checks if the current (or given) module is an installed package.
    An installed package is defined as a module which is installed in the site-packages directory.

    Parameters
    ----------
    module_path : str, optional
        The path of the package to check, by default None
        None will check the current module.

    is_root : bool, optional
        If True, the module_path is considered as the root path of the package, by default False

    Returns
    -------
    bool
        True if the module is installed as a package, False otherwise.
    """
    if module_path is None:
        module_path = current_module_path()
    if is_root:
        parent_path = module_path
    else:
        parent_path = os.path.abspath(
            os.path.normpath(os.path.join(module_path, "..")))
    if "site-packages" in os.path.basename(parent_path):
        return True
    return False


def is_dev_module(package_path: Optional[str] = None, is_root: bool = False) -> bool:
    """Checks if the current module is a development module.
    A development module is defined as a module which is not installed in the site-packages directory
    and has not a pyproject.toml file in its root directory.

    Parameters
    ----------
    package_path : str, optional
        The path of the package to check, by default None
        None will check the current package.

    is_root : bool, optional
        If True, the package_path is considered as the root path of the package, by default False

    Returns
    -------
    bool
        True if the module is a development module, False otherwise.
    """
    if package_path is None:
        package_path = current_module_path()
    if is_root:
        parent_path = package_path
    else:
        parent_path = os.path.abspath(
            os.path.normpath(os.path.join(package_path, "..")))
    if is_installed_module(module_path=package_path, is_root=is_root):
        return False
    # Check if a pyproject.toml file exists in the parent directory
    if not os.path.isfile(os.path.join(parent_path, "pyproject.toml")):
        return False
    return True


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
        EXECUTED_MODULE_PATH = current_module_path()
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
    return os.path.abspath(os.path.join(current_module_path(), ".."))


def get_package_root_path() -> str:
    """Gets the package path of the project.
    A package root path is defined as the directory where the
    source code of the project is located.
    In this case it is the directory where the tools package is located.

    Returns
    -------
    str
        The absolute package root path of the project.
    """
    return current_module_path()


def get_invoked_package_name() -> Optional[str]:
    """Gets the package name of the package which invoked this function.
    The package name is defined as the name of the package.

    Returns
    -------
    Optional[str]
        The package name of the package which invoked this function.
        Returns None if the package name could not be determined.
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
        The path of the primary module root, which contains all implementations for a certain package.
        Parent folder should contain the pyproject.toml file.
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


def get_package_info(project_root_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Gets the package information of the project.
    The package information is defined as the package name, version and description.
    Will return None if the project root path is not valid or the pyproject.toml file is not found.

    Parameters
    ----------
    project_root_path : str, optional
        The root path of the project, by default None

    Returns
    -------
    Optional[Dict[str, Any]]
        The package information of the project.
    """
    if toml is None:
        raise ImportError(
            "The package 'toml' is not installed. Please install it to use this function.")
    project_root_path = project_root_path if project_root_path is not None else get_project_root_path()
    if not is_project_root(project_root_path):
        return None
    # Check if the project root path is valid
    if not is_dev_module(project_root_path, is_root=True):
        return None
    pyproject_path = os.path.join(project_root_path, "pyproject.toml")
    with open(pyproject_path, "r") as file:
        pyproject = toml.load(file)
    package_info = pyproject.get("tool", dict()).get("poetry", dict())
    return package_info


def update_package_info(
    package_info: Dict[str, Any],
    project_root_path: Optional[str] = None,
):
    """
    Updates the package information of the project.
    The package information is defined as the package name, version and description.
    Will return None if the project root path is not valid or the pyproject.toml file is not found.

    Parameters
    ----------
    package_info : Dict[str, Any]
        The package information to update.
        Should contain at least the keys 'name', 'version' and 'description'.

    project_root_path : str, optional
        The root path of the project, by default None
        None will use the current project root path.

    """
    pyproject_path = os.path.join(project_root_path, "pyproject.toml")
    with open(pyproject_path, "r") as file:
        pyproject = toml.load(file)

    pyproject["tool"]["poetry"] = package_info
    with open(pyproject_path, "w") as file:
        toml.dump(pyproject, file)


def get_package_name(project_root_path: Optional[str] = None) -> Optional[str]:
    """Gets the package name of the project.
    The package name is defined as the name of the package.

    Returns
    -------
    Optional[str]
        The package name of the project, or None if the name could not be determined.
    """
    package_info = get_package_info(project_root_path=project_root_path)
    if package_info is None:
        return None
    package_name = package_info.get("name", None)
    return package_name
