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
import math
import time
from typing import Any, Callable, Optional

from absl import logging
from humanfriendly import format_timespan

from speciesnet.constants import Classification
from speciesnet.constants import DEFAULT_MAX_CLASSIFICATIONS_PER_DETECTION
from speciesnet.constants import Detection
from speciesnet.constants import Failure
from speciesnet.constants import NUM_CLASSIFICATIONS_FOR_ENSEMBLE
from speciesnet.ensemble_prediction_combiner import combine_predictions
from speciesnet.geofence_utils import geofence_animal_classification
from speciesnet.geofence_utils import roll_up_labels_to_first_matching_level
from speciesnet.utils import ModelInfo
from speciesnet.utils import whole_image_detection

# Handy type aliases.
PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]

_WHOLE_IMAGE_CATEGORY = whole_image_detection()["category"]


def _top_classes(classifications: dict, num_classes: Optional[int]) -> dict:
    """Returns a copy of a classification dict, limited to its top classes.

    Args:
        classifications:
            Classification dict that includes "classes" and "scores" among its keys, in
            decreasing order of scores. Other keys (e.g. target classes and logits) are
            copied unchanged.
        num_classes:
            Number of top classes to keep. If `None`, all classes are kept.

    Returns:
        The limited classification dict.
    """

    result = dict(classifications)
    if num_classes is not None:
        result["classes"] = classifications["classes"][:num_classes]
        result["scores"] = classifications["scores"][:num_classes]
    return result


def _check_matching_detections(
    filepath: str, classified_detections: list[dict], detections: list[dict]
) -> None:
    """Checks that classifier results refer to the same detections as the detector.

    Args:
        filepath:
            Image filepath, used for reporting purposes only.
        classified_detections:
            Detections attached to the classifier results for the image.
        detections:
            Detections from the detector results for the image.

    Raises:
        ValueError:
            If the two lists of detections don't match.
    """

    matching = len(classified_detections) == len(detections) and all(
        det1["category"] == det2["category"]
        and len(det1["bbox"]) == len(det2["bbox"])
        and all(
            math.isclose(x1, x2, abs_tol=1e-4)
            for x1, x2 in zip(det1["bbox"], det2["bbox"])
        )
        for det1, det2 in zip(classified_detections, detections)
    )
    if not matching:
        raise ValueError(
            f"Classifications for `{filepath}` were computed on different detections "
            f"than the ones provided. Make sure classifications and detections come "
            f"from the same detector output."
        )


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
        prediction_combiner: Callable = combine_predictions,
        force_model_download: bool = False,
    ) -> None:
        """Loads the ensemble resources.

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
            geofence:
                Whether to enable geofencing. If `False` skip it entirely.
            force_model_download:
                Whether to download model files even if they are already present
                locally. Typically used to replace model files that were only partially
                downloaded, and are therefore corrupted. Has no effect when `model_name`
                refers to a local folder.
        """

        start_time = time.time()

        self.model_info = ModelInfo(
            model_name, force_model_download=force_model_download
        )
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
        classifier_results: dict[str, dict],
        detector_results: dict[str, dict],
        geolocation_results: dict[str, dict],
        partial_predictions: Optional[dict[str, dict]] = None,
        max_classifications_per_detection: Optional[
            int
        ] = DEFAULT_MAX_CLASSIFICATIONS_PER_DETECTION,
    ) -> list[dict[str, Any]]:
        """Ensembles classifications and detections for a list of images.

        Each classified detection is ensembled independently of all other detections,
        producing one processed classification per classified detection.

        Args:
            filepaths:
                List of filepaths to ensemble predictions for.
            classifier_results:
                Dict of classifier results, with keys given by the filepaths to ensemble
                predictions for. Each result provides either "failures" or a list of
                "detections", where classified detections have a "raw_classifications"
                field. For whole-image classifiers, that list contains a single
                whole-image pseudo-detection.
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
            max_classifications_per_detection:
                Number of top classes to report in the raw classifications of each
                detection. If `None`, all available classes are reported. Processed
                classifications are always computed from the top
                `NUM_CLASSIFICATIONS_FOR_ENSEMBLE` classes, regardless of this value.

        Returns:
            List of ensembled predictions.

        Raises:
            ValueError:
                If the detections attached to the classifier results of an image don't
                match the detections from the detector results for that image.
        """

        results = []
        for filepath in filepaths:
            # Use the result from previously computed predictions when available.
            if partial_predictions is not None and filepath in partial_predictions:
                results.append(partial_predictions[filepath])
                continue

            # Check for failures.
            failure = Failure(0)
            if (
                filepath in classifier_results
                and "failures" not in classifier_results[filepath]
            ):
                classified_detections = classifier_results[filepath]["detections"]
            else:
                classified_detections = None
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

            # Assemble the output detections, and pair each classified detection with
            # the detection to ensemble its classifications with.
            output_detections: Optional[list[dict]] = None
            pairs: list[tuple[int, dict]] = []

            # Handle the whole-image classifier case.
            if classified_detections and (
                classified_detections[0]["category"] == _WHOLE_IMAGE_CATEGORY
            ):
                # Whole-image classifier: the pseudo-detection comes first, followed by
                # the (unclassified) detector output. The pseudo-detection is ensembled
                # with the top detection, like the full image always was.
                output_detections = [dict(classified_detections[0])]
                if detections is not None:
                    output_detections += [dict(det) for det in detections]
                    pairs.append(
                        (
                            0,
                            (
                                detections[0]
                                if detections
                                else {"label": Detection.ANIMAL, "conf": 0.0}
                            ),
                        )
                    )
            elif detections is not None:
                if classified_detections is not None:
                    _check_matching_detections(
                        filepath, classified_detections, detections
                    )
                    output_detections = [dict(det) for det in classified_detections]
                    pairs = [
                        (idx, det)
                        for idx, det in enumerate(output_detections)
                        if "raw_classifications" in det
                    ]
                else:
                    output_detections = [dict(det) for det in detections]

            # Ensemble each classified detection.
            if pairs and output_detections is not None:
                processed_classifications = self.prediction_combiner(
                    classifications_list=[
                        _top_classes(
                            output_detections[idx]["raw_classifications"],
                            NUM_CLASSIFICATIONS_FOR_ENSEMBLE,
                        )
                        for idx, _ in pairs
                    ],
                    detections=[det for _, det in pairs],
                    country=geolocation.get("country"),
                    admin1_region=geolocation.get("admin1_region"),
                    taxonomy_map=self.taxonomy_map,
                    geofence_map=self.geofence_map,
                    enable_geofence=self.enable_geofence,
                    geofence_fn=geofence_animal_classification,
                    roll_up_fn=roll_up_labels_to_first_matching_level,
                )
                for (idx, _), processed in zip(pairs, processed_classifications):
                    output_detections[idx]["processed_classification"] = processed

            # Only report the requested number of top classes.
            for det in output_detections or []:
                if "raw_classifications" in det:
                    det["raw_classifications"] = _top_classes(
                        det["raw_classifications"], max_classifications_per_detection
                    )

            # Assemble the prediction result.
            result = {
                "filepath": filepath,
                "failures": (
                    [f.name for f in Failure if f in failure] if failure else None
                ),
                "country": geolocation.get("country"),
                "admin1_region": geolocation.get("admin1_region"),
                "latitude": geolocation.get("latitude"),
                "longitude": geolocation.get("longitude"),
                "detections": output_detections,
            }
            result = {key: value for key, value in result.items() if value is not None}

            result["model_version"] = self.model_info.version

            results.append(result)

        return results
