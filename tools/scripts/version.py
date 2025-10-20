from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tools.mixin.argparser_mixin import ArgparserMixin
from enum import Enum
from tools.util.format import parse_enum
from tools.util.package_tools import get_package_info, set_module_path, update_package_info
import os
from tools.logger.logging import basic_config
import argparse
import re


class AutoBumpStrategy(Enum):
    """Enum for auto bump strategies."""
    NONE = "none"
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


VERSION_PATTERN = r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"


@dataclass
class VersionConfig(ArgparserMixin):

    commit_message_path: str = field()
    """The file path to the current commit message."""

    commit_message: Optional[str] = field(default=None)
    """The commit message. Will be filled based on the commit message file."""

    auto_bump: AutoBumpStrategy = field(default=AutoBumpStrategy.NONE)
    """Strategy to automatically bump the version. Will be filled based on the commit message."""

    dedicated_version: Optional[str] = field(default=None)
    """If set, the version will be set to this value. If not set, the version will be determined based on the auto_bump strategy."""

    updated_version: Optional[str] = field(default=None)
    """The updated version after the bump. Will be filled based on the auto_bump strategy or dedicated version. If a update in the commit msg is needed."""

    repository_path: str = field(default=None)
    """Path to the repository. If not set, will be determined from the commit message path"""

    package_info: Dict[str, Any] = field(default_factory=dict)
    """Package information of the project. Will be filled based on the repository path."""

    force: bool = field(default=False)
    """If set, the version will be changed even if it would indicate a downgrade."""

    git_tag: bool = field(default=True)
    """If set, a git tag will be created for the new version."""

    @classmethod
    def positional_args(cls):
        return ["commit_message_path"]

    @classmethod
    def argparser_ignore_fields(cls) -> List[str]:
        return ["commit_message", "package_info", "updated_version", "dedicated_version", "repository_path"]

    def prepare(self):
        if self.dedicated_version is not None:
            # Validate the dedicated version format
            if not re.match(VERSION_PATTERN, self.dedicated_version):
                raise ValueError(
                    f"Invalid version format: {self.dedicated_version}. Expected format: major.minor.patch")


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()


def get_config() -> VersionConfig:
    parser = argparse.ArgumentParser(
        description='Bumps the version of a package based on the commit message.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: VersionConfig = VersionConfig.parse_args(
        parser, add_config_path=False)
    config.prepare()
    return config


def parse_commit_message(cfg: VersionConfig):
    """Parses the commit message from the given file.

    Parameters
    ----------
    commit_message_file : str
        The path to the file containing the commit message.

    Returns
    -------
    str
        The parsed commit message.
    """
    import re
    commit_message_file = cfg.commit_message_path
    content = None
    if not os.path.exists(commit_message_file):
        raise FileNotFoundError(
            f"Commit message file not found: {commit_message_file}")
    with open(commit_message_file, "r") as file:
        content = file.read().strip()

    cfg.commit_message = content

    if cfg.repository_path is None:
        cfg.repository_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(commit_message_file)), ".."))

    cfg.package_info = get_package_info(cfg.repository_path)

    # Check the message for auto bump strategy
    auto_bump_pattern = r"--(?P<type>((major)|(minor)|(patch)))"
    found = [x.group("type") for x in re.finditer(auto_bump_pattern, content)]

    force_bump_pattern = r"--force"
    force_change = re.search(force_bump_pattern, content)
    if force_change:
        cfg.force = True

    if len(found) > 0:
        # Take the highest priority auto bump strategy
        if "major" in found:
            cfg.auto_bump = AutoBumpStrategy.MAJOR
        elif "minor" in found:
            cfg.auto_bump = AutoBumpStrategy.MINOR
        elif "patch" in found:
            cfg.auto_bump = AutoBumpStrategy.PATCH
    else:
        cfg.auto_bump = AutoBumpStrategy.NONE

    # Check for dedicated version
    version_pattern = r"--version(=|\s+)" + VERSION_PATTERN
    version_match = re.search(version_pattern, content)
    if version_match:
        major = version_match.group("major")
        minor = version_match.group("minor")
        patch = version_match.group("patch")
        cfg.dedicated_version = f"{major}.{minor}.{patch}"
        cfg.auto_bump = AutoBumpStrategy.NONE


def change_version(cfg: VersionConfig) -> bool:
    # Get the current version from the package info
    current_version = cfg.package_info.get("version", None)
    if current_version is None:
        raise ValueError(
            "Current version could not be determined from package info.")

    # Parse current version
    if not re.match(VERSION_PATTERN, current_version):
        raise ValueError(
            f"Current version format is invalid: {current_version}. Expected format: major.minor.patch")

    # Check if the current version matches the dedicated version
    if cfg.dedicated_version is not None and current_version == cfg.dedicated_version:
        return False  # No change needed

    if cfg.dedicated_version is not None:
        vers = cfg.dedicated_version
        # Parse the current version and check for downgrade
        current_major, current_minor, current_patch = map(
            int, current_version.split('.'))
        target_major, target_minor, target_patch = map(int, vers.split('.'))
        if (target_major < current_major or
            (target_major == current_major and target_minor < current_minor) or
                (target_major == current_major and target_minor == current_minor and target_patch < current_patch)):
            if not cfg.force:
                raise ValueError(
                    f"Downgrading version from {current_version} to {cfg.dedicated_version} is not allowed. If this is wanted, specify --force within message")
            else:
                print(
                    f"Forcefully downgrading version from {current_version} to {cfg.dedicated_version}.")
        cfg.updated_version = vers
    elif cfg.dedicated_version is None and cfg.auto_bump != AutoBumpStrategy.NONE:
        # Bump the version based on the auto_bump strategy
        major, minor, patch = map(int, current_version.split('.'))
        if cfg.auto_bump == AutoBumpStrategy.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif cfg.auto_bump == AutoBumpStrategy.MINOR:
            minor += 1
            patch = 0
        elif cfg.auto_bump == AutoBumpStrategy.PATCH:
            patch += 1
        cfg.updated_version = f"{major}.{minor}.{patch}"
    else:
        # No change needed
        return False

    # Update the package info with the new version
    cfg.package_info["version"] = cfg.updated_version
    update_package_info(
        cfg.package_info, project_root_path=cfg.repository_path)
    return True


def update_commit_message(cfg: VersionConfig):
    """Updates the commit message with the new version."""
    if cfg.updated_version is not None and cfg.dedicated_version is None:
        # Appending the new version to the commit message avoiding duplicates on failed commits.
        new_message = cfg.commit_message + " --version=" + cfg.updated_version

    # If we need to tag the commit, we append the tag to the message
    if cfg.git_tag:
        new_message += " --tag=v" + cfg.updated_version

    with open(cfg.commit_message_path, "w") as file:
        file.write(new_message)

    cfg.commit_message = new_message


def main(cfg: VersionConfig):
    """Main function to run the version bumping script."""
    parse_commit_message(cfg)
    need_change = change_version(cfg)

    if not need_change:
        exit(0)  # OK

    if need_change:
        # Auto-bumped version, inserting a new version tag in the commit message
        update_commit_message(cfg)


if __name__ == "__main__":
    print("Print sys args:\n", os.sys.argv)
    config()
    cfg = get_config()
    main(cfg)
