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

"""Logic for combining predictions for the SpeciesNet ensemble."""

from typing import Callable, Optional

from speciesnet.constants import Classification
from speciesnet.constants import Detection

PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]


def combine_predictions(
    *,
    classifications_list: list[dict],
    detections: list[dict],
    country: Optional[str],
    admin1_region: Optional[str],
    taxonomy_map: dict,
    geofence_map: dict,
    enable_geofence: bool,
    geofence_fn: Callable,
    roll_up_fn: Callable,
) -> list[dict]:
    """Ensembles each classification with its corresponding detection.

    This operation leverages heuristics to make the most of the classifier and
    the detector predictions.

    Each element of `classifications_list` is ensembled with the element of
    `detections` at the same position, independently of all other elements.

    Args:
        classifications_list:
            List of dicts containing classification results, one per detection.
            "classes" and "scores" are expected to be provided among the dict keys.
        detections:
            List of detection results, one per classification. Each detection is
            expected to be a dict providing "label" and "conf" among its keys.
        country:
            Country (in ISO 3166-1 alpha-3 format) associated with predictions.
            Optional.
        admin1_region:
            First-level administrative division (in ISO 3166-2 format) associated with
            predictions. Optional.
        taxonomy_map:
            Dictionary mapping taxa to labels.
        geofence_map:
            Dictionary mapping full class strings to geofence rules.
        enable_geofence:
            Whether geofencing is enabled.
        geofence_fn:
            Callable to geofence animal classifications.
        roll_up_fn:
            Callable to roll up labels to the first matching level.

    Returns:
        A list of dicts describing the ensemble results, one per classification and in
        the same order. Each dict includes "class", "score" and "source" keys.

    Raises:
        ValueError:
            If `classifications_list` and `detections` have different lengths.
    """
    if len(classifications_list) != len(detections):
        raise ValueError(
            f"Expected one detection per classification, got "
            f"{len(classifications_list)} classifications and "
            f"{len(detections)} detections."
        )

    results = []

    for classifications, det in zip(classifications_list, detections):
        det_class = det["label"]
        det_score = det["conf"]
        prediction = None
        score = 0.0
        source = ""

        top_classification_class = classifications["classes"][0]
        top_classification_score = classifications["scores"][0]

        if det_class == Detection.HUMAN:
            # Threshold #1a: high-confidence HUMAN detections.
            if det_score > 0.7:
                prediction, score, source = Classification.HUMAN, det_score, "detector"
            elif (
                # Threshold #1b: mid-confidence HUMAN detections + high-confidence
                # HUMAN/VEHICLE classifications.
                det_score > 0.2
                and top_classification_class
                in {Classification.HUMAN, Classification.VEHICLE}
                and top_classification_score > 0.5
            ):
                prediction, score, source = (
                    Classification.HUMAN,
                    top_classification_score,
                    "classifier",
                )

        if prediction is None and det_class == Detection.VEHICLE:
            # Threshold #2a: mid-confidence VEHICLE detections + high-confidence HUMAN
            # classifications.
            if (
                det_score > 0.2
                and top_classification_class == Classification.HUMAN
                and top_classification_score > 0.5
            ):
                prediction, score, source = (
                    Classification.HUMAN,
                    top_classification_score,
                    "classifier",
                )
            # Threshold #2b: high-confidence VEHICLE detections.
            elif det_score > 0.7:
                prediction, score, source = (
                    Classification.VEHICLE,
                    det_score,
                    "detector",
                )
            # Threshold #2c: mid-confidence VEHICLE detections + high-confidence VEHICLE
            # classifications.
            elif (
                det_score > 0.2
                and top_classification_class == Classification.VEHICLE
                and top_classification_score > 0.4
            ):
                prediction, score, source = (
                    Classification.VEHICLE,
                    top_classification_score,
                    "classifier",
                )

        if prediction is None:
            # Threshold #3a: high-confidence BLANK "detections" + high-confidence BLANK
            # classifications.
            if (
                det_score < 0.2
                and top_classification_class == Classification.BLANK
                and top_classification_score > 0.5
            ):
                prediction, score, source = (
                    Classification.BLANK,
                    top_classification_score,
                    "classifier",
                )
            # Threshold #3b: extra-high-confidence BLANK classifications.
            elif (
                top_classification_class == Classification.BLANK
                and top_classification_score > 0.99
            ):
                prediction, score, source = (
                    Classification.BLANK,
                    top_classification_score,
                    "classifier",
                )
            elif top_classification_class not in {
                Classification.BLANK,
                Classification.HUMAN,
                Classification.VEHICLE,
            }:
                # Threshold #4a: extra-high-confidence ANIMAL classifications.
                if top_classification_score > 0.8:
                    prediction, score, source = geofence_fn(
                        labels=classifications["classes"],
                        scores=classifications["scores"],
                        country=country,
                        admin1_region=admin1_region,
                        taxonomy_map=taxonomy_map,
                        geofence_map=geofence_map,
                        enable_geofence=enable_geofence,
                    )
                # Threshold #4b: high-confidence ANIMAL classifications + mid-confidence
                # ANIMAL detections.
                elif (
                    top_classification_score > 0.65
                    and det_class == Detection.ANIMAL
                    and det_score > 0.2
                ):
                    prediction, score, source = geofence_fn(
                        labels=classifications["classes"],
                        scores=classifications["scores"],
                        country=country,
                        admin1_region=admin1_region,
                        taxonomy_map=taxonomy_map,
                        geofence_map=geofence_map,
                        enable_geofence=enable_geofence,
                    )

        if prediction is None:
            # Threshold #5a: high-confidence ANIMAL rollups.
            rollup = roll_up_fn(
                labels=classifications["classes"],
                scores=classifications["scores"],
                country=country,
                admin1_region=admin1_region,
                target_taxonomy_levels=[
                    "genus",
                    "family",
                    "order",
                    "class",
                    "kingdom",
                ],
                non_blank_threshold=0.65,
                taxonomy_map=taxonomy_map,
                geofence_map=geofence_map,
                enable_geofence=enable_geofence,
            )
            if rollup:
                prediction, score, source = rollup
            # Threshold #5b: mid-confidence ANIMAL detections.
            elif det_class == Detection.ANIMAL and det_score > 0.5:
                prediction, score, source = Classification.ANIMAL, det_score, "detector"
            else:
                prediction, score, source = (
                    Classification.UNKNOWN,
                    top_classification_score,
                    "classifier",
                )

        results.append(
            {
                "class": (
                    prediction.value
                    if isinstance(prediction, Classification)
                    else prediction
                ),
                "score": score,
                "source": source,
            }
        )

    return results
