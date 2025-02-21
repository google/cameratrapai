# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom pytest configuration."""

import multiprocessing as mp

import pytest

from speciesnet import SUPPORTED_MODELS


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    mp.set_start_method("spawn")


def pytest_addoption(parser):
    """Adds extra pytest flags."""

    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Run tests on a given model only.",
    )
    parser.addoption(
        "--az",
        action="store_true",
        default=False,
        help="Run Azure tests.",
    )
    parser.addoption(
        "--gs",
        action="store_true",
        default=False,
        help="Run GCP tests.",
    )
    parser.addoption(
        "--s3",
        action="store_true",
        default=False,
        help="Run AWS tests.",
    )


def pytest_generate_tests(metafunc):
    """Generates extra pytest tests."""

    # Parametrize tests with a `model_name` fixture.
    if "model_name" in metafunc.fixturenames:
        model_name = metafunc.config.getoption("model")
        if model_name:
            metafunc.parametrize("model_name", [model_name], scope="module")
        else:
            metafunc.parametrize("model_name", SUPPORTED_MODELS, scope="module")


def pytest_configure(config):
    """Configures the pytest environment."""

    # Register markers for cloud tests.
    config.addinivalue_line("markers", "az: mark test for Azure only")
    config.addinivalue_line("markers", "gs: mark test for GCP only")
    config.addinivalue_line("markers", "s3: mark test for AWS only")


def pytest_collection_modifyitems(config, items):
    """Modifies collected pytest items."""

    # Add markers for cloud tests.
    if not config.getoption("--az"):
        skip_az = pytest.mark.skip(reason="needs --az option to run")
        for item in items:
            if "az" in item.keywords:
                item.add_marker(skip_az)
    if not config.getoption("--gs"):
        skip_gs = pytest.mark.skip(reason="needs --gs option to run")
        for item in items:
            if "gs" in item.keywords:
                item.add_marker(skip_gs)
    if not config.getoption("--s3"):
        skip_s3 = pytest.mark.skip(reason="needs --s3 option to run")
        for item in items:
            if "s3" in item.keywords:
                item.add_marker(skip_s3)
