
from dataclasses import dataclass, field

from tools.config.config import Config


@dataclass
class OutputConfig(Config):
    """Config which defines an actual output folder for experiments. This is heavily used when launch mechanisms need control over experiment outputs."""

    output_folder: str = field(default=None)
    """Folder where all outputs are stored overrides runs_path. Default is None."""