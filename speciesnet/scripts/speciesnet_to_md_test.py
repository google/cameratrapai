# Copyright 2026 Google LLC
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

import pytest

from speciesnet.scripts import speciesnet_to_md


def _convert_predictions(tmp_path, predictions: list[dict]) -> dict:
    predictions_path = tmp_path / "predictions.json"
    results_path = tmp_path / "md_results.json"
    predictions_path.write_text(
        json.dumps({"predictions": predictions}), encoding="utf-8"
    )

    speciesnet_to_md.generate_md_results_from_predictions_json(
        predictions_path, results_path
    )

    return json.loads(results_path.read_text(encoding="utf-8"))


class TestGenerateMDResults:
    """Tests for conversion of SpeciesNet predictions to MegaDetector results."""

    @pytest.mark.parametrize(
        "classifier_only", [False, True], ids=["ensemble", "classifier"]
    )
    def test_non_blank_prediction_without_detections_creates_unknown_detection(
        self,
        tmp_path,
        classifier_only: bool,
    ) -> None:
        prediction = {
            "filepath": "animal.jpg",
            "classifications": {
                "classes": [speciesnet_to_md.animal_prediction_string],
                "scores": [0.75],
            },
        }
        if not classifier_only:
            prediction = prediction | {
                "detections": [],
                "prediction": speciesnet_to_md.animal_prediction_string,
                "prediction_score": 0.75,
            }

        results = _convert_predictions(tmp_path, [prediction])

        assert results["detection_categories"] == {"1": "unknown"}
        assert results["images"] == [
            {
                "file": "animal.jpg",
                "detections": [
                    {
                        "category": "1",
                        "conf": 0.75,
                        "bbox": [0, 0, 1, 1],
                        "classifications": [["0", 0.75]],
                    }
                ],
            }
        ]

    def test_unknown_detection_uses_id_after_existing_categories(
        self, tmp_path
    ) -> None:
        predictions = [
            {
                "filepath": "detected.jpg",
                "detections": [
                    {
                        "category": "3",
                        "label": "vehicle",
                        "conf": 0.9,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            },
            {
                "filepath": "classified.jpg",
                "detections": [],
                "classifications": {
                    "classes": [speciesnet_to_md.animal_prediction_string],
                    "scores": [0.8],
                },
                "prediction": speciesnet_to_md.animal_prediction_string,
                "prediction_score": 0.8,
            },
        ]

        results = _convert_predictions(tmp_path, predictions)

        assert results["detection_categories"] == {
            "3": "vehicle",
            "4": "unknown",
        }
        classified_image = next(
            image for image in results["images"] if image["file"] == "classified.jpg"
        )
        assert classified_image["detections"][0]["category"] == "4"
        detected_image = next(
            image for image in results["images"] if image["file"] == "detected.jpg"
        )
        assert detected_image["detections"] == [
            {
                "category": "3",
                "conf": 0.9,
                "bbox": [0.1, 0.2, 0.3, 0.4],
            }
        ]

    def test_blank_classification_without_detections_stays_empty(
        self, tmp_path
    ) -> None:
        prediction = {
            "filepath": "blank.jpg",
            "detections": [],
            "classifications": {
                "classes": [speciesnet_to_md.blank_prediction_string],
                "scores": [0.99],
            },
            "prediction": speciesnet_to_md.blank_prediction_string,
            "prediction_score": 0.99,
        }

        results = _convert_predictions(tmp_path, [prediction])

        assert results["detection_categories"] == {}
        assert results["images"] == [{"file": "blank.jpg", "detections": []}]
