from datetime import datetime, date
import os
import sys
from typing import Tuple
import re
import pytest
from tools.util.package_tools import set_exec_module
import tools
from tools.logger.logging import basic_config

OUTPUT_DIR = "tests/output/"


def config():
    basic_config()
    set_exec_module(tools)
    pass


@pytest.fixture(autouse=True, scope='session')
def session_encapsule():
    setup_session()
    yield
    teardown_session()


def setup_session():
    config()
    global OUTPUT_DIR
    _date, _time = today_time_now()
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, _date, _time)

    # Add tests to sys.path
    tests_dir = os.path.join(os.path.dirname(__file__), "..")
    if tests_dir not in sys.path:
        sys.path.append(tests_dir)
    pass


def teardown_session():
    pass


def today_time_now() -> Tuple[str, str]:
    today = date.today()
    date_today = today.strftime("%Y_%m_%d")
    now = datetime.now()
    date_now = now.strftime("%Y_%m_%d_%H_%M_%S")
    return date_today, date_now


def get_output_dir() -> str:
    """Returns the output dir for this test run."""
    dir = OUTPUT_DIR
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir


def get_test_name() -> str:
    return os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]


def get_test_name_for_path(allow_dot: bool = True) -> str:
    return re.sub(f'[^\w_{"." if allow_dot else ""} -]', '_', get_test_name())


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, int):
        return f'{argname}={val}'
    if isinstance(val, str):
        return f'{argname}=\'{val}\''
    if isinstance(val, tuple):
        vals = [str(x) for x in val]
        return f'{argname}=({",".join(vals) + ("," if len(vals) == 1 else "")})'
    if isinstance(val, complex):
        return f'{argname}={str(val)}'
    # return None to let pytest handle the formatting
    return None
