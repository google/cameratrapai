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

"""Script to run the SpeciesNet model.

Provides a command-line interface to execute the SpeciesNet model on various inputs. It
uses flags for specifying input, output, and run options, allowing the user to run the
model in different modes.
"""

import json
import multiprocessing as mp
from typing import Callable, Literal, Optional

from absl import app
from absl import flags
from absl import logging

from speciesnet import DEFAULT_MODEL
from speciesnet import only_one_true
from speciesnet import SpeciesNet
from speciesnet.ensemble_prediction_combiner import PredictionType
from speciesnet.utils import load_partial_predictions
from speciesnet.utils import prepare_instances_dict

_MODEL = flags.DEFINE_string(
    "model",
    DEFAULT_MODEL,
    "SpeciesNet model to load.",
)
_CLASSIFIER_ONLY = flags.DEFINE_bool(
    "classifier_only",
    False,
    "Whether to run only the classifier component.",
)
_DETECTOR_ONLY = flags.DEFINE_bool(
    "detector_only",
    False,
    "Whether to run only the detector component.",
)
_ENSEMBLE_ONLY = flags.DEFINE_bool(
    "ensemble_only",
    False,
    "Whether to run only the ensemble component.",
)
_GEOFENCE = flags.DEFINE_bool(
    "geofence",
    True,
    "Whether to enable geofencing during ensemble prediction.",
)
_INSTANCES_JSON = flags.DEFINE_string(
    "instances_json",
    None,
    "Input JSON file with instances to get predictions for.",
)
_FILEPATHS = flags.DEFINE_list(
    "filepaths",
    None,
    "List of image filepaths to get predictions for.",
)
_FILEPATHS_TXT = flags.DEFINE_string(
    "filepaths_txt",
    None,
    "Input TXT file with image filepaths to get predictions for.",
)
_FOLDERS = flags.DEFINE_list(
    "folders",
    None,
    "List of image folders to get predictions for.",
)
_FOLDERS_TXT = flags.DEFINE_string(
    "folders_txt",
    None,
    "Input TXT file with image folders to get predictions for.",
)
_CLASSIFICATIONS_JSON = flags.DEFINE_string(
    "classifications_json",
    None,
    "Input JSON file with classifications from previous runs.",
)
_DETECTIONS_JSON = flags.DEFINE_string(
    "detections_json",
    None,
    "Input JSON file with detections from previous runs.",
)
_PREDICTIONS_JSON = flags.DEFINE_string(
    "predictions_json",
    None,
    "Output JSON file for storing computed predictions.",
)
_RUN_MODE = flags.DEFINE_enum(
    "run_mode",
    "multi_thread",
    ["single_thread", "multi_thread", "multi_process"],
    "Running mode, determining the parallelism strategy to use at prediction time.",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    8,
    "Batch size for inference.",
)
_PROGRESS_BARS = flags.DEFINE_bool(
    "progress_bars",
    True,
    "Whether to show progress bars for the various inference components.",
)


def guess_predictions_source(
    predictions: dict[str, dict],
) -> Literal["classifier", "detector", "ensemble", "unknown", "invalid"]:
    """Guesses which model component generated given predictions.

    Args:
        predictions: Dict of predictions, keyed by filepaths.

    Returns:
        Returns "classifier", "detector" or "ensemble" when the corresponding component
        was identified as the source of predictions. Returns "invalid" when predictions
        contain both classifications and detections, but couldn't identify results from
        the ensemble. Returns "unknown" when no prediction is recognizable (e.g. when
        there are only failures).
    """

    found_classifications = False
    found_detections = False
    found_ensemble_results = False

    for prediction in predictions.values():
        if "classifications" in prediction:
            found_classifications = True
        if "detections" in prediction:
            found_detections = True
        if "prediction" in prediction:
            found_ensemble_results = True
        if found_classifications and found_detections and not found_ensemble_results:
            return "invalid"

    if found_ensemble_results:
        return "ensemble"
    if found_classifications:
        return "classifier"
    if found_detections:
        return "detector"
    return "unknown"


def custom_combine_predictions_fn(
    *,
    classifications: dict[str, list],
    detections: list[dict],
    country: Optional[str],
    admin1_region: Optional[str],
    taxonomy_map: dict,
    geofence_map: dict,
    enable_geofence: bool,
    geofence_fn: Callable,
    roll_up_fn: Callable,
) -> PredictionType:
    """Ensembles classifications and detections in a custom way.

    Args:
        classifications:
            Dict of classification results. "classes" and "scores" are expected to be
            provided among the dict keys.
        detections:
            List of detection results, sorted in decreasing order of their confidence
            score. Each detection is expected to be a dict providing "label" and "conf"
            among its keys.
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
        A tuple of <label, score, prediction_source> describing the ensemble result.
    """

    del detections  # Unused.
    del country  # Unused.
    del admin1_region  # Unused.
    del taxonomy_map  # Unused.
    del geofence_map  # Unused.
    del enable_geofence  # Unused.
    del geofence_fn  # Unused.
    del roll_up_fn  # Unused.

    # Always return the second classifier prediction.
    return (
        classifications["classes"][1],
        classifications["scores"][1],
        "custom_ensemble",
    )


def main(argv: list[str]) -> None:
    del argv  # Unused.

    # Check for a valid combination of components to run.
    components = [_CLASSIFIER_ONLY, _DETECTOR_ONLY, _ENSEMBLE_ONLY]
    components_names = [f"--{c.name}" for c in components]
    components_values = [c.value for c in components]
    components_strings = [
        f"{name}={value}" for name, value in zip(components_names, components_values)
    ]
    if any(components_values) and not only_one_true(*components_values):
        raise ValueError(
            f"Expected at most one of [{', '.join(components_names)}] to be provided. "
            f"Received: [{', '.join(components_strings)}]."
        )
    if _ENSEMBLE_ONLY.value and (
        not _CLASSIFICATIONS_JSON.value or not _DETECTIONS_JSON.value
    ):
        raise ValueError(
            f"Expected --{_CLASSIFICATIONS_JSON.name} and --{_DETECTIONS_JSON.name} to "
            f"be set when --{_ENSEMBLE_ONLY.name} is requested."
        )
    if _CLASSIFIER_ONLY.value:
        components = "classifier"
    elif _DETECTOR_ONLY.value:
        components = "detector"
    elif _ENSEMBLE_ONLY.value:
        components = "ensemble"
    else:
        components = "all"

    # Check for valid inputs.
    inputs = [_INSTANCES_JSON, _FILEPATHS, _FILEPATHS_TXT, _FOLDERS, _FOLDERS_TXT]
    inputs_names = [f"--{i.name}" for i in inputs]
    inputs_values = [i.value for i in inputs]
    inputs_strings = [
        f"{name}={value}" for name, value in zip(inputs_names, inputs_values)
    ]
    if not only_one_true(*inputs_values):
        raise ValueError(
            f"Expected exactly one of [{', '.join(inputs_names)}] to be provided. "
            f"Received: [{', '.join(inputs_strings)}]."
        )
    instances_dict = prepare_instances_dict(
        instances_json=_INSTANCES_JSON.value,
        filepaths=_FILEPATHS.value,
        filepaths_txt=_FILEPATHS_TXT.value,
        folders=_FOLDERS.value,
        folders_txt=_FOLDERS_TXT.value,
    )

    # Check the compatibility of output predictions with existing partial predictions.
    if _PREDICTIONS_JSON.value:
        partial_predictions, _ = load_partial_predictions(
            _PREDICTIONS_JSON.value, instances_dict["instances"]
        )
        predictions_source = guess_predictions_source(partial_predictions)

        if _CLASSIFIER_ONLY.value and predictions_source not in [
            "classifier",
            "unknown",
        ]:
            raise RuntimeError(
                f"The classifier risks overwriting previous predictions from "
                f"`{_PREDICTIONS_JSON.value}` that were produced by different "
                f"components. Make sure to provide a different output location to "
                f"--{_PREDICTIONS_JSON.name}."
            )

        if _DETECTOR_ONLY.value and predictions_source not in ["detector", "unknown"]:
            raise RuntimeError(
                f"The detector risks overwriting previous predictions from "
                f"`{_PREDICTIONS_JSON.value}` that were produced by different "
                f"components. Make sure to provide a different output location to "
                f"--{_PREDICTIONS_JSON.name}."
            )

        if _ENSEMBLE_ONLY.value and predictions_source not in ["ensemble", "unknown"]:
            raise RuntimeError(
                f"The ensemble risks overwriting previous predictions from "
                f"`{_PREDICTIONS_JSON.value}` that were produced by different "
                f"components. Make sure to provide a different output location to "
                f"--{_PREDICTIONS_JSON.name}."
            )

    else:
        user_input = input(
            "Continue without saving predictions to a JSON file? [y/N]: "
        )
        if user_input.lower() not in ["yes", "y"]:
            print(f"Please provide an output filepath via --{_PREDICTIONS_JSON.name}.")
            return

    # Load classifications and/or detections from previous runs.
    if _CLASSIFICATIONS_JSON.value:
        classifications_dict, _ = load_partial_predictions(
            _CLASSIFICATIONS_JSON.value, instances_dict["instances"]
        )
    else:
        classifications_dict = {}
    if _DETECTIONS_JSON.value:
        detections_dict, _ = load_partial_predictions(
            _DETECTIONS_JSON.value, instances_dict["instances"]
        )
    else:
        detections_dict = {}

    # Set running mode.
    run_mode = _RUN_MODE.value
    mp.set_start_method("spawn")

    # Make predictions.
    model = SpeciesNet(
        _MODEL.value,
        components=components,
        geofence=_GEOFENCE.value,
        # Uncomment the line below if you want to run your own custom ensembling
        # routine. And also, implement that routine! :-)
        # combine_predictions_fn=custom_combine_predictions_fn,
        multiprocessing=(run_mode == "multi_process"),
    )
    if _CLASSIFIER_ONLY.value:
        predictions_dict = model.classify(
            instances_json=_INSTANCES_JSON.value,
            filepaths=_FILEPATHS.value,
            filepaths_txt=_FILEPATHS_TXT.value,
            folders=_FOLDERS.value,
            folders_txt=_FOLDERS_TXT.value,
            detections_dict=detections_dict,
            run_mode=run_mode,
            batch_size=_BATCH_SIZE.value,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
        )
    elif _DETECTOR_ONLY.value:
        predictions_dict = model.detect(
            instances_json=_INSTANCES_JSON.value,
            filepaths=_FILEPATHS.value,
            filepaths_txt=_FILEPATHS_TXT.value,
            folders=_FOLDERS.value,
            folders_txt=_FOLDERS_TXT.value,
            run_mode=run_mode,
            batch_size=_BATCH_SIZE.value,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
        )
    elif _ENSEMBLE_ONLY.value:
        predictions_dict = model.ensemble_from_past_runs(
            instances_json=_INSTANCES_JSON.value,
            filepaths=_FILEPATHS.value,
            filepaths_txt=_FILEPATHS_TXT.value,
            folders=_FOLDERS.value,
            folders_txt=_FOLDERS_TXT.value,
            classifications_dict=classifications_dict,
            detections_dict=detections_dict,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
        )
    else:
        predictions_dict = model.predict(
            instances_json=_INSTANCES_JSON.value,
            filepaths=_FILEPATHS.value,
            filepaths_txt=_FILEPATHS_TXT.value,
            folders=_FOLDERS.value,
            folders_txt=_FOLDERS_TXT.value,
            run_mode=run_mode,
            batch_size=_BATCH_SIZE.value,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
        )
    if predictions_dict is not None:
        logging.info(
            "Predictions:\n%s",
            json.dumps(predictions_dict, ensure_ascii=False, indent=4),
        )


if __name__ == "__main__":
    app.run(main)
