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
from pathlib import Path
from typing import Callable, Literal, Optional

from absl import app
from absl import flags

from speciesnet import DEFAULT_MODEL
from speciesnet import only_one_true
from speciesnet import SpeciesNet
from speciesnet.constants import DEFAULT_DETECTION_THRESHOLD_FOR_CLASSIFICATION
from speciesnet.constants import DEFAULT_MAX_CLASSIFICATIONS_PER_DETECTION
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
    "Run only the classifier component. --classifier_only enables classifier-only mode, --noclassifier_only (default) disables it.",
)
_DETECTOR_ONLY = flags.DEFINE_bool(
    "detector_only",
    False,
    "Run only the detector component. --detector_only enables detector-only mode, --nodetector_only (default) disables it.",
)
_ENSEMBLE_ONLY = flags.DEFINE_bool(
    "ensemble_only",
    False,
    "Run only the ensemble component. --ensemble_only enables ensemble-only mode, --noensemble_only (default) disables it.",
)
_GEOFENCE = flags.DEFINE_bool(
    "geofence",
    True,
    "Enable geofencing during ensemble prediction. --geofence (default) enables geofencing, --nogeofence disables it.",
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
_COUNTRY = flags.DEFINE_string(
    "country",
    None,
    "Country (in ISO 3166-1 alpha-3 format, e.g. 'AUS') to enforce on all instances.",
)
_ADMIN1_REGION = flags.DEFINE_string(
    "admin1_region",
    None,
    "First-level administrative division (in ISO 3166-2 format, e.g. 'CA') to enforce on all "
    "instances.",
)
_TARGET_SPECIES_TXT = flags.DEFINE_string(
    "target_species_txt",
    None,
    "Input TXT file with species of interest to always compute classification scores for.",
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
    "Output JSON file for storing computed predictions. If this file exists, only instances "
    "that are not already present in the output will be processed.",
)
_RUN_MODE = flags.DEFINE_enum(
    "run_mode",
    "multi_thread",
    ["multi_thread", "multi_process"],
    "Parallelism strategy.",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    8,
    "Batch size for classifier inference.",
)
_PROGRESS_BARS = flags.DEFINE_bool(
    "progress_bars",
    True,
    "Whether to show progress bars for the various inference components. --progress_bars "
    "(default) enables progress bars, --noprogress_bars disables them.",
)
_BYPASS_PROMPTS = flags.DEFINE_bool(
    "bypass_prompts",
    False,
    "Whether to bypass confirmation prompts when expected files aren't supplied, or "
    "unexpected files are supplied. --bypass_prompts bypasses prompts, --nobypass_prompts "
    "(default) does not.",
)
_IGNORE_EXISTING_PREDICTIONS = flags.DEFINE_bool(
    "ignore_existing_predictions",
    False,
    "Whether to ignore existing predictions in the output JSON file and reprocess all "
    "instances. --ignore_existing_predictions bypasses loading partial results, "
    "--noignore_existing_predictions (default) resumes from existing predictions.",
)
_FORCE_MODEL_DOWNLOAD = flags.DEFINE_bool(
    "force_model_download",
    False,
    "Download model files even if they are already present locally, "
    "typically to replace model files that were only partially downloaded, and are "
    "therefore corrupted. --force_model_download forces a download, "
    "--noforce_model_download (default) does not. Has no effect when --model refers "
    "to a local folder.",
)
_DETECTION_THRESHOLD_FOR_CLASSIFICATION = flags.DEFINE_float(
    "detection_threshold_for_classification",
    DEFAULT_DETECTION_THRESHOLD_FOR_CLASSIFICATION,
    "Minimum detection confidence for a detection to be run through the classifier. "
    "Detections below this threshold are still included in the output, without "
    "classifications. Has no effect for whole-image classifiers.",
)
_MAX_CLASSIFICATIONS_PER_IMAGE = flags.DEFINE_integer(
    "max_classifications_per_image",
    None,
    "Maximum number of detections to classify per image (the highest-confidence "
    "detections above --detection_threshold_for_classification are classified). "
    "If not specified (or < 0), all detections above the threshold are classified. Has "
    "no effect for whole-image classifiers.",
)
_MAX_CLASSIFICATIONS_PER_DETECTION = flags.DEFINE_integer(
    "max_classifications_per_detection",
    DEFAULT_MAX_CLASSIFICATIONS_PER_DETECTION,
    "Number of top classes to report in the raw classifications of each classified "
    "detection. If < 0, all classes are reported. Processed classifications are always "
    "computed from the top 5 classes, regardless of this value. In classifier-only "
    "mode, at least the top 5 classes are reported, so that the ensemble can be run "
    "later.",
)


def guess_predictions_source(
    predictions: dict[str, dict],
) -> Literal["classifier", "detector", "ensemble", "unknown"]:
    """Guesses which model component generated given predictions.

    Ensemble results are recognized by their model version (which only the ensemble
    reports) or by processed classifications, classifier results by raw
    classifications, and detector results by detections without any classifications.
    Classifier results in which no detection was classified are indistinguishable from
    detector results.

    Args:
        predictions: Dict of predictions, keyed by filepaths.

    Returns:
        Returns "classifier", "detector" or "ensemble" when the corresponding component
        was identified as the source of predictions. Returns "unknown" when no
        prediction is recognizable (e.g. when there are only failures).
    """

    found_classifications = False
    found_detections = False
    found_ensemble_results = False

    for prediction in predictions.values():
        detections = prediction.get("detections") or []
        if "model_version" in prediction or any(
            "processed_classification" in det for det in detections
        ):
            found_ensemble_results = True
        if any("raw_classifications" in det for det in detections):
            found_classifications = True
        if "detections" in prediction:
            found_detections = True

    if found_ensemble_results:
        return "ensemble"
    if found_classifications:
        return "classifier"
    if found_detections:
        return "detector"
    return "unknown"


def custom_combine_predictions_fn(
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
    """Ensembles classifications and detections in a custom way.

    Args:
        classifications_list:
            List of classification results, one per detection. "classes" and "scores"
            are expected to be provided among the dict keys.
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
        the same order, each providing "class", "score" and "source" keys.
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
    return [
        {
            "class": classifications["classes"][1],
            "score": classifications["scores"][1],
            "source": "custom_ensemble",
        }
        for classifications in classifications_list
    ]


def say_yes_to_continue(question: str, stop_message: str) -> bool:
    if _BYPASS_PROMPTS.value:
        return True
    user_input = input(f"{question} [y/N]: ")
    if user_input.lower() in ["yes", "y"]:
        return True
    else:
        print(stop_message)
        return False


def local_file_exists(filepath: Optional[str]) -> bool:
    if not filepath:
        return False
    return Path(filepath).exists()


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

    # Validate region code formatting.
    if _COUNTRY.value:
        if not isinstance(_COUNTRY.value, str) or len(_COUNTRY.value) != 3:
            raise ValueError(f"Invalid three-letter country code {str(_COUNTRY.value)}")
    if _ADMIN1_REGION.value:
        if not isinstance(_ADMIN1_REGION.value, str) or len(_ADMIN1_REGION.value) != 2:
            raise ValueError(
                f"Invalid two-letter region code {str(_ADMIN1_REGION.value)}"
            )

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
        filepaths=list(_FILEPATHS.value) if _FILEPATHS.value else None,
        filepaths_txt=_FILEPATHS_TXT.value,
        folders=list(_FOLDERS.value) if _FOLDERS.value else None,
        folders_txt=_FOLDERS_TXT.value,
        country=_COUNTRY.value,
        admin1_region=_ADMIN1_REGION.value,
    )

    # Check the compatibility of output predictions with existing partial predictions.
    if _PREDICTIONS_JSON.value:
        if _IGNORE_EXISTING_PREDICTIONS.value:
            # When ignoring existing predictions, delete the file to ensure all instances
            # are reprocessed from scratch.
            if local_file_exists(_PREDICTIONS_JSON.value):
                print(f"Deleting existing predictions in `{_PREDICTIONS_JSON.value}`.")
                Path(_PREDICTIONS_JSON.value).unlink()
        else:
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

            if _DETECTOR_ONLY.value and predictions_source not in [
                "detector",
                "unknown",
            ]:
                raise RuntimeError(
                    f"The detector risks overwriting previous predictions from "
                    f"`{_PREDICTIONS_JSON.value}` that were produced by different "
                    f"components. Make sure to provide a different output location to "
                    f"--{_PREDICTIONS_JSON.name}."
                )

            if _ENSEMBLE_ONLY.value and predictions_source not in [
                "ensemble",
                "unknown",
            ]:
                raise RuntimeError(
                    f"The ensemble risks overwriting previous predictions from "
                    f"`{_PREDICTIONS_JSON.value}` that were produced by different "
                    f"components. Make sure to provide a different output location to "
                    f"--{_PREDICTIONS_JSON.name}."
                )

    else:
        if not say_yes_to_continue(
            question="Continue without saving predictions to a JSON file?",
            stop_message=(
                f"Please provide an output filepath via --{_PREDICTIONS_JSON.name}."
            ),
        ):
            return

    # Allow conversion of -1 to None for int-valued inputs
    max_classifications_per_image = _MAX_CLASSIFICATIONS_PER_IMAGE.value
    if max_classifications_per_image is not None and max_classifications_per_image < 0:
        max_classifications_per_image = None
    max_classifications_per_detection = _MAX_CLASSIFICATIONS_PER_DETECTION.value
    if (
        max_classifications_per_detection is not None
        and max_classifications_per_detection < 0
    ):
        max_classifications_per_detection = None

    # If a list of target species is given, check that it exists
    if _TARGET_SPECIES_TXT.value is not None and not local_file_exists(
        _TARGET_SPECIES_TXT.value
    ):
        raise RuntimeError(
            f"Target species file '{_TARGET_SPECIES_TXT.value}' specified via --{_PREDICTIONS_JSON.name} does not exist."
        )

    # Load classifications and/or detections from previous runs.
    classifications_dict, _ = load_partial_predictions(
        _CLASSIFICATIONS_JSON.value, instances_dict["instances"]
    )
    detections_dict, _ = load_partial_predictions(
        _DETECTIONS_JSON.value, instances_dict["instances"]
    )

    # Set running mode.
    run_mode: Literal["multi_thread", "multi_process"] = _RUN_MODE.value  # type: ignore
    mp.set_start_method("spawn")

    # Make predictions.
    model = SpeciesNet(
        _MODEL.value,
        components=components,
        geofence=_GEOFENCE.value,
        target_species_txt=_TARGET_SPECIES_TXT.value,
        # Uncomment the line below if you want to run your own custom ensembling
        # routine. And also, implement that routine! :-)
        # combine_predictions_fn=custom_combine_predictions_fn,
        multiprocessing=(run_mode == "multi_process"),
        force_model_download=_FORCE_MODEL_DOWNLOAD.value,
    )
    if hasattr(model, "classifier") and not hasattr(model, "detector"):
        if not model.whole_image_classifier and not local_file_exists(
            _DETECTIONS_JSON.value
        ):
            raise RuntimeError(
                "This classifier classifies detection crops, so it needs detections. "
                f"Please provide detections via --{_DETECTIONS_JSON.name} and make "
                "sure that file exists."
            )
        if model.whole_image_classifier and _DETECTIONS_JSON.value:
            if not say_yes_to_continue(
                question=(
                    "Classifier doesn't expect detections JSON, yet such file was "
                    f"provided via --{_DETECTIONS_JSON.name}. Continue even though "
                    "given detections JSON will have no effect?"
                ),
                stop_message=f"Please drop the --{_DETECTIONS_JSON.name} flag.",
            ):
                return
    if _CLASSIFIER_ONLY.value:
        predictions_dict = model.classify(
            instances_dict=instances_dict,
            detections_dict=detections_dict,
            run_mode=run_mode,
            batch_size=_BATCH_SIZE.value,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
            detection_threshold_for_classification=(
                _DETECTION_THRESHOLD_FOR_CLASSIFICATION.value
            ),
            max_classifications_per_image=max_classifications_per_image,
            max_classifications_per_detection=max_classifications_per_detection,
        )
    elif _DETECTOR_ONLY.value:
        predictions_dict = model.detect(
            instances_dict=instances_dict,
            run_mode=run_mode,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
        )
    elif _ENSEMBLE_ONLY.value:
        predictions_dict = model.ensemble_from_past_runs(
            instances_dict=instances_dict,
            classifications_dict=classifications_dict,
            detections_dict=detections_dict,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
            max_classifications_per_detection=max_classifications_per_detection,
        )
    else:
        predictions_dict = model.predict(
            instances_dict=instances_dict,
            run_mode=run_mode,
            batch_size=_BATCH_SIZE.value,
            progress_bars=_PROGRESS_BARS.value,
            predictions_json=_PREDICTIONS_JSON.value,
            detection_threshold_for_classification=(
                _DETECTION_THRESHOLD_FOR_CLASSIFICATION.value
            ),
            max_classifications_per_image=max_classifications_per_image,
            max_classifications_per_detection=max_classifications_per_detection,
        )
    if predictions_dict is not None:
        print(
            "Predictions:\n"
            + json.dumps(predictions_dict, ensure_ascii=False, indent=4)
        )


if __name__ == "__main__":
    app.run(main)
