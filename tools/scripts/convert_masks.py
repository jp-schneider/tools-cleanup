#!/usr/bin/env python3
import numpy as np
from tools.io.image import load_image
from tools.util.typing import DEFAULT, _DEFAULT
from tools.util.path_tools import process_path, read_directory
from tools.util.format import parse_type, format_dataframe_string
from tools.metric.metric import Metric
from dataclasses import dataclass, field
import argparse
import logging  # noqa
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from tools.logger.logging import basic_config, logger
from tools.util.package_tools import set_module_path
from tools.context.script_execution import ScriptExecution
from tools.serialization.json_convertible import JsonConvertible
from tools.mixin.argparser_mixin import ArgparserMixin
from tools.segmentation.masking import load_channel_masks, save_mask
from tools.util.progress_factory import ProgressFactory
from tools.util.format import parse_format_string

@dataclass
class ConvertMasksConfig(JsonConvertible, ArgparserMixin):

    ov_masks_directory: str = field(default=None)
    """Directory containing the masks to convert."""

    ov_masks_filename_pattern: Optional[str] = field(default=None)
    """Pattern to match the masks to convert."""

    output_directory: str = field(default=None)
    """Directory to save the converted masks."""

    output_filename_pattern: Optional[str] = field(default="{object_id:05d}/{index:05d}.png")
    """Pattern to save the converted masks."""

    def prepare(self):
        self.ov_masks_directory = process_path(self.ov_masks_directory, need_exists=True, interpolate=True, interpolate_object=self, variable_name="ov_masks_directory")
        self.output_directory = process_path(self.output_directory, need_exists=False, interpolate=True, interpolate_object=self, variable_name="output_directory")


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    set_module_path(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    basic_config()


def get_config() -> ConvertMasksConfig:
    parser = argparse.ArgumentParser(
        description='Converts masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config: ConvertMasksConfig = ConvertMasksConfig.parse_args(parser)
    return config


def main(config: ConvertMasksConfig):
    logger.info("Converting masks")
    logger.info("Loading inputs....")

    pf = ProgressFactory()
    args = dict()
    if config.ov_masks_filename_pattern is not None:
        args["filename_pattern"] = config.ov_masks_filename_pattern
    masks_list = load_channel_masks(config.ov_masks_directory, **args, progress_bar=True, output_format="value")
    # Create output directory if it does not exist

    if not os.path.exists(config.output_directory):
        os.makedirs(config.output_directory)

    list_bar = pf.bar(total=len(masks_list), desc="Converting OV Groups")
    for mask_stack in masks_list:
        N, H, W = mask_stack.shape
        unique_values = np.unique(mask_stack) # Get unique values in the mask
        vals = [x for x in unique_values if x != 0]
        item_bar = pf.bar(total=len(vals) * N, desc="Saving individual masks", tag="CONVERT_ELEMENTS")

        for val in vals:
            formatted_paths = parse_format_string(config.output_filename_pattern, [dict(object_id=val) for _ in range(N)])
            for i, path in enumerate(formatted_paths):
                op = os.path.join(config.output_directory, path)  
                if not os.path.exists(os.path.dirname(op)):
                    os.makedirs(os.path.dirname(op))  
                mask = mask_stack[i]
                save_mask((mask == val)[..., None], op)
                item_bar.update(1)
        list_bar.update(1)

if __name__ == "__main__":
    config()
    cfg = get_config()
    with ScriptExecution(cfg):
        main(cfg)
