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

"""Ensemble functionality of SpeciesNet."""

__all__ = [
    "SpeciesNetEnsemble",
]

import json
import time
from typing import Any, Callable

from absl import logging
from humanfriendly import format_timespan

from speciesnet.constants import Classification
from speciesnet.constants import Failure
from speciesnet.ensemble_prediction_combiner import combine_predictions_for_single_item
from speciesnet.geofence_utils import geofence_animal_classification
from speciesnet.geofence_utils import roll_up_labels_to_first_matching_level
from speciesnet.utils import ModelInfo

# Handy type aliases.
PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]


def _load_taxonomy_from_file(taxonomy_file):
    """Loads the taxonomy .txt file into a dict."""

    def _taxa_from_label(label: str) -> str:
        return ";".join(label.split(";")[1:6])

    # Create taxonomy map.
    with open(taxonomy_file, mode="r", encoding="utf-8") as fp:
        labels = [line.strip() for line in fp.readlines()]
        taxonomy_map = {_taxa_from_label(label): label for label in labels}

        for label in [
            Classification.BLANK,
            Classification.VEHICLE,
            Classification.UNKNOWN,
        ]:
            taxa = _taxa_from_label(label)
            if taxa in taxonomy_map:
                del taxonomy_map[taxa]

        for label in [Classification.HUMAN, Classification.ANIMAL]:
            taxa = _taxa_from_label(label)
            taxonomy_map[taxa] = label
    return taxonomy_map


class SpeciesNetEnsemble:
    """Ensemble component of SpeciesNet."""

    def __init__(
        self,
        model_name: str,
        geofence: bool = True,
        prediction_combiner: Callable = combine_predictions_for_single_item,
    ) -> None:
        """Loads the ensemble resources.

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
            geofence:
                Whether to enable geofencing. If `False` skip it entirely.
        """

        start_time = time.time()

        self.model_info = ModelInfo(model_name)
        self.enable_geofence = geofence
        self.taxonomy_map = self.load_taxonomy()
        self.geofence_map = self.load_geofence()
        self.prediction_combiner = prediction_combiner

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetEnsemble in %s.",
            format_timespan(end_time - start_time),
        )

    def load_taxonomy(self):
        """Loads the taxonomy from the model info."""

        return _load_taxonomy_from_file(self.model_info.taxonomy)

    def load_geofence(self):
        """Loads the geofence map from the model info."""

        with open(self.model_info.geofence, mode="r", encoding="utf-8") as fp:
            geofence_map = json.load(fp)
        return geofence_map

    def combine(  # pylint: disable=too-many-positional-arguments
        self,
        filepaths: list[str],
        classifier_results: dict[str, Any],
        detector_results: dict[str, Any],
        geolocation_results: dict[str, Any],
        partial_predictions: dict[str, dict],
    ) -> list[dict[str, Any]]:
        """Ensembles classifications and detections for a list of images.

        Args:
            filepaths:
                List of filepaths to ensemble predictions for.
            classifier_results:
                Dict of classifier results, with keys given by the filepaths to ensemble
                predictions for.
            detector_results:
                Dict of detector results, with keys given by the filepaths to ensemble
                predictions for.
            geolocation_results:
                Dict of geolocation results, with keys given by the filepaths to
                ensemble predictions for.
            partial_predictions:
                Dict of partial predictions from previous ensemblings, with keys given
                by the filepaths for which predictions where already ensembled. Used to
                skip re-ensembling for the matching filepaths.

        Returns:
            List of ensembled predictions.
        """

        results = []
        for filepath in filepaths:
            # Use the result from previously computed predictions when available.
            if filepath in partial_predictions:
                results.append(partial_predictions[filepath])
                continue

            # Check for failures.
            failure = Failure(0)
            if (
                filepath in classifier_results
                and "failures" not in classifier_results[filepath]
            ):
                classifications = classifier_results[filepath]["classifications"]
            else:
                classifications = None
                failure |= Failure.CLASSIFIER
            if (
                filepath in detector_results
                and "failures" not in detector_results[filepath]
            ):
                detections = detector_results[filepath]["detections"]
            else:
                detections = None
                failure |= Failure.DETECTOR
            if filepath in geolocation_results:
                geolocation = geolocation_results[filepath]
            else:
                geolocation = {}
                failure |= Failure.GEOLOCATION

            # Add as much raw information as possible to the prediction result.
            result = {
                "filepath": filepath,
                "failures": (
                    [f.name for f in Failure if f in failure] if failure else None
                ),
                "country": geolocation.get("country"),
                "admin1_region": geolocation.get("admin1_region"),
                "latitude": geolocation.get("latitude"),
                "longitude": geolocation.get("longitude"),
                "classifications": classifications,
                "detections": detections,
            }
            result = {key: value for key, value in result.items() if value is not None}

            # Most importantly, ensemble everything into a single prediction.
            if classifications is not None and detections is not None:
                prediction, score, source = self.prediction_combiner(
                    classifications=classifications,
                    detections=detections,
                    country=geolocation.get("country"),
                    admin1_region=geolocation.get("admin1_region"),
                    taxonomy_map=self.taxonomy_map,
                    geofence_map=self.geofence_map,
                    enable_geofence=self.enable_geofence,
                    geofence_fn=geofence_animal_classification,
                    roll_up_fn=roll_up_labels_to_first_matching_level,
                )
                result["prediction"] = (
                    prediction.value
                    if isinstance(prediction, Classification)
                    else prediction
                )
                result["prediction_score"] = score
                result["prediction_source"] = source

            # Finally, report the model version.
            result["model_version"] = self.model_info.version

            results.append(result)

        return results
