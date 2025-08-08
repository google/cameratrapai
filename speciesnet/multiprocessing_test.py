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

# pylint: disable=missing-module-docstring

import json
import logging
from typing import Any, Optional

import pytest

from speciesnet.multiprocessing import SpeciesNet


def assert_approx_objs(
    obj1: Any,
    obj2: Any,
    rtol: Optional[float] = None,  # Relative tolerance.
    atol: Optional[float] = None,  # Absolute tolerance.
) -> None:
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        assert set(obj1.keys()) == set(obj2.keys())
        for key in obj1.keys():
            assert_approx_objs(obj1[key], obj2[key], rtol=rtol, atol=atol)
    elif isinstance(obj1, list) and isinstance(obj2, list):
        assert len(obj1) == len(obj2)
        for item1, item2 in zip(obj1, obj2):
            assert_approx_objs(item1, item2, rtol=rtol, atol=atol)
    else:
        assert obj1 == pytest.approx(obj2, rel=rtol, abs=atol)


@pytest.fixture(name="instances_dict")
def fx_instances_dict() -> dict:
    with open("test_data/instances_with_errors.json", mode="r", encoding="utf-8") as fp:
        return json.load(fp)


class TestSingleProcess:
    """Tests for single-process inference."""

    @pytest.fixture(scope="class")
    def model(self, model_name: str) -> SpeciesNet:
        return SpeciesNet(model_name)

    def test_predict(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.predict(
            instances_dict=instances_dict, run_mode="single_thread", progress_bars=True
        )
        predictions_dict2 = model.predict(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert_approx_objs(predictions_dict1, predictions_dict2, atol=1e-3)
        logging.info("Predictions (%s): %s", request.node.name, predictions_dict1)

    def test_classify(self, request, instances_dict, model) -> None:
        predictions_dict = model.classify(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        assert predictions_dict
        logging.info("Classifications (%s): %s", request.node.name, predictions_dict)

    def test_detect(self, request, instances_dict, model) -> None:
        predictions_dict = model.detect(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        assert predictions_dict
        logging.info("Detections (%s): %s", request.node.name, predictions_dict)

    def test_ensemble_from_past_runs(self, request, instances_dict, model) -> None:
        predictions_dict = model.ensemble_from_past_runs(
            instances_dict=instances_dict, progress_bars=True
        )
        assert predictions_dict
        logging.info("Ensemble results (%s): %s", request.node.name, predictions_dict)


class TestMultiProcess:
    """Tests for multi-process inference."""

    @pytest.fixture(scope="class")
    def model(self, model_name: str) -> SpeciesNet:
        return SpeciesNet(model_name, multiprocessing=True)

    def test_predict(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.predict(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        predictions_dict2 = model.predict(
            instances_dict=instances_dict, run_mode="multi_process", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert_approx_objs(predictions_dict1, predictions_dict2, atol=1e-3)
        logging.info("Predictions (%s): %s", request.node.name, predictions_dict1)

    def test_batch_predict(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.predict(
            instances_dict=instances_dict, batch_size=1, progress_bars=True
        )
        predictions_dict2 = model.predict(
            instances_dict=instances_dict, batch_size=4, progress_bars=True
        )
        predictions_dict3 = model.predict(
            instances_dict=instances_dict, batch_size=7, progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert predictions_dict3
        assert_approx_objs(predictions_dict1, predictions_dict2, atol=1e-3)
        assert_approx_objs(predictions_dict1, predictions_dict3, atol=1e-3)
        logging.info("Predictions (%s): %s", request.node.name, predictions_dict1)

    def test_classify(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.classify(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        predictions_dict2 = model.classify(
            instances_dict=instances_dict, run_mode="multi_process", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert_approx_objs(predictions_dict1, predictions_dict2, atol=1e-3)
        logging.info("Classifications (%s): %s", request.node.name, predictions_dict1)

    def test_batch_classify(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.classify(
            instances_dict=instances_dict, batch_size=1, progress_bars=True
        )
        predictions_dict2 = model.classify(
            instances_dict=instances_dict, batch_size=4, progress_bars=True
        )
        predictions_dict3 = model.classify(
            instances_dict=instances_dict, batch_size=7, progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert predictions_dict3
        assert_approx_objs(predictions_dict1, predictions_dict2, atol=1e-3)
        assert_approx_objs(predictions_dict1, predictions_dict3, atol=1e-3)
        logging.info("Classifications (%s): %s", request.node.name, predictions_dict1)

    def test_detect(self, request, instances_dict, model) -> None:
        predictions_dict1 = model.detect(
            instances_dict=instances_dict, run_mode="multi_thread", progress_bars=True
        )
        predictions_dict2 = model.detect(
            instances_dict=instances_dict, run_mode="multi_process", progress_bars=True
        )
        assert predictions_dict1
        assert predictions_dict2
        assert_approx_objs(predictions_dict1, predictions_dict2, atol=1e-3)
        logging.info("Detections (%s): %s", request.node.name, predictions_dict1)
