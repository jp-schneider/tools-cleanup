from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from tools.util.typing import DEFAULT
import os

@dataclass
class EnvConfigMixin():
    """A mixin for configuration classes that require or been able to load an environment file."""

    env_file: Optional[str] = field(default=DEFAULT)
    """The environment file to load. If None, no environment file will be used. If DEFAULT it will try to load from .env."""

    def prepare(self):
        from tools.logger.logging import logger
        from dotenv import load_dotenv
        if self.env_file is not None:
            env_file = self.env_file
            is_default = env_file == DEFAULT
            if is_default:
                env_file = ".env"
            if not os.path.exists(env_file):
                if not is_default:
                    # Only warn if the user specified a file
                    logger.warning(
                        f"Environment file {env_file} does not exist.")
            else:
                loaded = load_dotenv(env_file)
                if loaded:
                    logger.info(f"Loaded environment file {env_file}.")
                else:
                    logger.warning(
                        f"Environment file {env_file} could not be loaded.")
        super().prepare()
        