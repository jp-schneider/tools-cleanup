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
import subprocess

VERSION_PATTERN = r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)"
TAG_PATTERN = r"--tag(=|\s+)(?P<tag>[\d\w\_\-\.,]+)"


@dataclass
class TagConfig(ArgparserMixin):

    hash: str = field(default=None)
    """The commit hash to tag."""

    commit_message_path: str = field()
    """The file path to the current commit message."""

    commit_message: Optional[str] = field(default=None)
    """The commit message. Will be filled based on the commit message file."""

    tags: Optional[List[str]] = field(default_factory=list)
    """List of tags to be added to git for the recent commit."""

    @classmethod
    def positional_args(cls):
        return ["hash"]

    @classmethod
    def argparser_ignore_fields(cls) -> List[str]:
        return ["commit_message"]

    def prepare(self):
        pass


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()


def get_config() -> TagConfig:
    parser = argparse.ArgumentParser(
        description='Creates git tags based on the commit message.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: TagConfig = TagConfig.parse_args(parser)
    config.prepare()
    return config


def get_commit_message(cfg: TagConfig) -> str:
    # Use subprocess to get the commit message from the git command
    cmd = ["git", "log", "--format=%B", "-1", cfg.hash]
    val = None
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        val = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to get commit message for hash {cfg.hash}: {e}")
    cfg.commit_message = val

    # Check if tags are present in the commit message
    tag_matches = re.findall(TAG_PATTERN, cfg.commit_message)
    if tag_matches:
        cfg.tags = [match.group("tag") for match in tag_matches]


def create_tags(cfg: TagConfig):
    """Creates git tags for the current commit based on the tags found in the commit message."""
    if not cfg.tags:
        print("No tags found in the commit message.")
        return

    for tag in cfg.tags:
        # Check if the tag already exists
        cmd = ["git", "tag", "-l", tag]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print(f"Tag '{tag}' already exists. Skipping creation.")
                continue
        except subprocess.CalledProcessError as e:
            print(f"Failed to check for existing tag '{tag}': {e}")
            continue

        cmd = ["git", "tag", tag, cfg.hash]
        try:
            subprocess.run(cmd, check=True)
            print(f"Tag '{tag}' created for commit {cfg.hash}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create tag '{tag}': {e}")


def main(cfg: TagConfig):
    get_commit_message(cfg)
    if len(cfg.tags) == 0:
        print("No tags found in the commit message.")
        return
    print(f"Found tags: {cfg.tags}")
    create_tags(cfg)
    exit(0)  # OK


if __name__ == "__main__":
    config()
    cfg = get_config()
    main(cfg)
