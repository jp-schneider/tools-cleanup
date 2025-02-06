#!/usr/bin/env python3
# Sample job file for an omni job. Can be used as a standalone file or within a job list.
import argparse
import asyncio
import os

from tools.logger.logging import basic_config
import matplotlib
import matplotlib.pyplot as plt
from tools.config.grid_search_config import GridSearchConfig
from tools.run.grid_search_runner import GridSearchRunner
plt.ioff()
matplotlib.use('agg')


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    from tools.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    basic_config()


def get_config() -> GridSearchConfig:
    parser = argparse.ArgumentParser(
        description='Can run multiple configs, one after another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config = GridSearchConfig.parse_args(parser, sep="-")
    return config


async def main(config: GridSearchConfig):
    from tools.logger.logging import logger
    runner = GridSearchRunner(config)
    runner.build(build_children=False)

    # Training
    if config.create_job_file:
        logger.info(f"Creating job file...")
        file = runner.create_job_file()
        logger.info(f"Created job file at: {file}")

    if not config.dry_run:
        logger.info(f"Start training of: {config.name_experiment}")
        runner.train()

if __name__ == "__main__":
    config()
    cfg = get_config()
    from tools.logger.logging import logger
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(cfg))
        loop.close()
    except Exception as err:
        logger.exception(
            f"Raised {type(err).__name__} in {current_filename()}, exiting...")
        exit(1)
    exit(0)
