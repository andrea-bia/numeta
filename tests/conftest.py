import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="all",
        help="Backend to run tests on: 'fortran', 'c', or 'all' (default)",
    )


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        option = metafunc.config.getoption("backend")
        if option == "all":
            params = ["fortran", "c"]
        else:
            params = [option]
        metafunc.parametrize("backend", params)
