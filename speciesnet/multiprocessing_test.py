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

from speciesnet.constants import FORMAT_VERSION
from speciesnet.multiprocessing import SpeciesNet

# fmt: off
# pylint: disable=line-too-long

HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
DOMESTIC_DOG = "3d80f1d6-b1df-4966-9ff4-94053c7a902a;mammalia;carnivora;canidae;canis;familiaris;domestic dog"
PLAINS_ZEBRA = "dd39bbd5-077c-482e-9d33-bd176116c870;mammalia;perissodactyla;equidae;equus;quagga;plains zebra"
THOMSONS_GAZELLE = "dc5dbe17-a8ca-40a6-ac6a-3b6b1d63e6d6;mammalia;artiodactyla;bovidae;eudorcas;thomsonii;thomson's gazelle"

# pylint: enable=line-too-long
# fmt: on


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


def _predictions_by_filepath(predictions_dict: dict) -> dict[str, dict]:
    return {p["filepath"]: p for p in predictions_dict["predictions"]}


def _processed_classes(prediction: dict) -> list[Optional[str]]:
    return [
        (
            det["processed_classification"]["class"]
            if "processed_classification" in det
            else None
        )
        for det in prediction.get("detections", [])
    ]


class TestEndToEnd:
    """End-to-end tests checking the content of predictions."""

    @pytest.fixture(scope="class")
    def model(self, model_name: str) -> SpeciesNet:
        return SpeciesNet(model_name)

    @pytest.fixture(scope="class")
    def predictions(self, model) -> dict[str, dict]:
        with open("test_data/instances_with_errors.json", encoding="utf-8") as fp:
            instances_dict = json.load(fp)
        predictions_dict = model.predict(instances_dict=instances_dict)
        assert predictions_dict
        assert predictions_dict["info"] == {"format_version": FORMAT_VERSION}
        return _predictions_by_filepath(predictions_dict)

    def test_output_format(self, model, predictions) -> None:
        for prediction in predictions.values():
            assert not {
                "classifications",
                "classifications_list",
                "prediction",
                "prediction_score",
                "prediction_source",
                "processed_classifications",
            } & set(prediction)
            detections = prediction.get("detections", [])

            if model.whole_image_classifier and detections:
                # The whole-image pseudo-detection comes first, and is the only
                # classified detection.
                assert detections[0]["category"] == "0"
                assert detections[0]["label"] == "whole_image_classification"
                assert detections[0]["conf"] == 1.0
                assert detections[0]["bbox"] == [0.0, 0.0, 1.0, 1.0]
                assert all(det["category"] != "0" for det in detections[1:])
                classified = [0]
            else:
                classified = [
                    idx for idx, det in enumerate(detections) if det["conf"] >= 0.1
                ]

            # Detections are sorted by decreasing confidence.
            confs = [det["conf"] for det in detections]
            assert confs == sorted(confs, reverse=True)

            if "failures" in prediction:
                continue
            for idx, det in enumerate(detections):
                if idx in classified:
                    raw = det["raw_classifications"]
                    assert len(raw["classes"]) == 2
                    assert len(raw["scores"]) == 2
                    assert raw["scores"] == sorted(raw["scores"], reverse=True)
                    assert set(det["processed_classification"]) == {
                        "class",
                        "score",
                        "source",
                    }
                else:
                    assert "raw_classifications" not in det
                    assert "processed_classification" not in det

    def test_multi_object_images(self, model, predictions) -> None:
        if model.whole_image_classifier:
            pytest.skip("Whole-image models do not classify individual detections.")

        prediction = predictions["test_data/human_and_dog.jpg"]
        assert _processed_classes(prediction) == [HUMAN, DOMESTIC_DOG]
        assert [det["label"] for det in prediction["detections"]] == [
            "human",
            "animal",
        ]

        prediction = predictions["test_data/gazelles_and_zebra.jpg"]
        assert _processed_classes(prediction) == [
            THOMSONS_GAZELLE,
            THOMSONS_GAZELLE,
            PLAINS_ZEBRA,
            None,  # Below the detection threshold for classification.
            None,  # Below the detection threshold for classification.
        ]

    def test_no_classifiable_detections(self, model, predictions) -> None:
        if model.whole_image_classifier:
            pytest.skip("Whole-image models always classify the whole image.")

        # No detection at or above the threshold, so nothing gets classified.
        prediction = predictions["test_data/blank3.jpg"]
        assert prediction["detections"]
        assert _processed_classes(prediction) == [None] * len(prediction["detections"])

    def test_classification_options(self, model) -> None:
        filepath = "test_data/gazelles_and_zebra.jpg"
        instances_dict = {"instances": [{"filepath": filepath, "country": "TZA"}]}

        def _predict(**kwargs) -> dict:
            predictions_dict = model.predict(instances_dict=instances_dict, **kwargs)
            assert predictions_dict
            return predictions_dict["predictions"][0]

        def _num_classified(prediction: dict) -> int:
            return sum("raw_classifications" in det for det in prediction["detections"])

        default = _predict()
        num_detections = len(default["detections"])
        if model.whole_image_classifier:
            assert _num_classified(default) == 1
            assert _num_classified(_predict(max_classifications_per_image=0)) == 1
        else:
            assert _num_classified(default) == 3
            assert (
                _num_classified(_predict(detection_threshold_for_classification=0.0))
                == num_detections
            )
            capped = _predict(max_classifications_per_image=1)
            assert _num_classified(capped) == 1
            assert "raw_classifications" in capped["detections"][0]

        # The number of reported classes doesn't affect processed classifications.
        for max_classifications in [1, 5, 10, None]:
            prediction = _predict(max_classifications_per_detection=max_classifications)
            det = prediction["detections"][0]
            if max_classifications is None:
                assert len(det["raw_classifications"]["classes"]) > 10
            else:
                assert len(det["raw_classifications"]["classes"]) == (
                    max_classifications
                )
            assert _processed_classes(prediction) == _processed_classes(default)

    def test_classify_then_ensemble(self, model) -> None:
        with open("test_data/instances.json", encoding="utf-8") as fp:
            instances_dict = json.load(fp)

        detections_dict = model.detect(instances_dict=instances_dict)
        classifications_dict = model.classify(
            instances_dict=instances_dict,
            detections_dict=_predictions_by_filepath(detections_dict),
        )
        assert detections_dict and classifications_dict

        # Classifier-only output keeps enough classes to run the ensemble later.
        for prediction in classifications_dict["predictions"]:
            for det in prediction.get("detections", []):
                if "raw_classifications" in det:
                    assert len(det["raw_classifications"]["classes"]) == 5

        ensemble_dict = model.ensemble_from_past_runs(
            instances_dict=instances_dict,
            classifications_dict=_predictions_by_filepath(classifications_dict),
            detections_dict=_predictions_by_filepath(detections_dict),
        )
        predictions_dict = model.predict(instances_dict=instances_dict)
        assert_approx_objs(ensemble_dict, predictions_dict, atol=1e-3)
