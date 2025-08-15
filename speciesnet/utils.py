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

"""Common utilities."""

__all__ = [
    "BBox",
    "only_one_true",
    "file_exists",
    "load_rgb_image",
    "prepare_instances_dict",
    "load_json",
    "write_json",
]

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Optional, Union
import uuid

from absl import logging
from cloudpathlib import CloudPath
from huggingface_hub import snapshot_download
import kagglehub
import numpy as np
import PIL.Image
import PIL.ImageFile
import PIL.ImageOps
import requests

StrPath = Union[str, Path]

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported extensions for image discovery.
IMG_EXTENSIONS = {
    "jpg",
    "JPG",
    "jpeg",
    "JPEG",
    "png",
    "PNG",
    "tif",
    "TIF",
    "tiff",
    "TIFF",
    "webp",
    "WEBP",
}

# Custom agent for image requests over HTTP(S).
CUSTOM_HTTP_AGENT = {"User-Agent": "SpeciesNetBot/0.0 (github.com/google/cameratrapai)"}


@dataclass(frozen=True)
class ModelInfo:
    """Dataclass describing SpeciesNet model and its underlying resources to load."""

    version: str  # Model version.
    type_: str  # Model type.
    classifier: Path  # Path to classifier model.
    classifier_labels: Path  # Path to labels file used by clasifier.
    detector: Path  # Path to detector model.
    taxonomy: Path  # Path to taxonomy file used by ensemble.
    geofence: Path  # Path to geofence file used by ensemble.

    def __init__(self, model_name: str) -> None:
        """Creates dataclass to describe a given model.

        Args:
            model_name:
                String value identifying the model to be described by this dataclass.
                It can be a Kaggle identifier (starting with `kaggle:`), a HuggingFace
                identifier (starting with `hf:`) or a local folder to load the model
                from. If the model name is a remote identifier (Kaggle or HuggingFace),
                the model files are automatically downloaded on the first call.
        """

        # Download model files (if necessary) and set the base local directory.
        kaggle_prefix = "kaggle:"
        hf_prefix = "hf:"
        if model_name.startswith(kaggle_prefix):
            base_dir = kagglehub.model_download(model_name[len(kaggle_prefix) :])
        elif model_name.startswith(hf_prefix):
            base_dir = snapshot_download(model_name[len(hf_prefix) :])
        else:
            base_dir = model_name
        base_dir = Path(base_dir)

        # Load model info.
        info = load_json(base_dir / "info.json")

        # Download detector weights if not provided with the other model files.
        filepath_or_url = info["detector"]
        if filepath_or_url.startswith("http://") or filepath_or_url.startswith(
            "https://"
        ):
            filename = self._url_to_filename(filepath_or_url)
            info["detector"] = filename
            filepath = base_dir / filename
            if not filepath.exists():
                response = requests.get(filepath_or_url, stream=True, timeout=600)
                response.raise_for_status()
                with open(filepath, mode="wb") as fp:
                    for chunk in response.iter_content(chunk_size=8192):
                        fp.write(chunk)

        # Set dataclass fields using a workaround to bypass read-only constraints.
        object.__setattr__(self, "version", info["version"])
        object.__setattr__(self, "type_", info["type"])
        object.__setattr__(self, "classifier", base_dir / info["classifier"])
        object.__setattr__(
            self, "classifier_labels", base_dir / info["classifier_labels"]
        )
        object.__setattr__(self, "detector", base_dir / info["detector"])
        object.__setattr__(self, "taxonomy", base_dir / info["taxonomy"])
        object.__setattr__(self, "geofence", base_dir / info["geofence"])

    def _url_to_filename(self, url: str) -> str:
        """Sanitizes a URL to get a valid filename.

        Args:
            url: String value for the URL to sanitize.

        Returns:
            String value representing a valid filename obtained from the sanitized URL.
        """

        if "?" in url:
            filename = url.split("?")[0]
        else:
            filename = url
        filename = filename.replace(":", "_").replace("/", "_")
        return filename


@dataclass(frozen=True)
class PreprocessedImage:
    """Dataclass describing a preprocessed image."""

    arr: np.ndarray  # Multidimensional array of image pixels.
    orig_width: int  # Original image width.
    orig_height: int  # Original image height.


@dataclass(frozen=True)
class BBox:
    """Dataclass describing a bounding box."""

    xmin: float
    ymin: float
    width: float
    height: float


def load_json(filepath: StrPath) -> dict:
    """Loads a JSON file with UTF-8 encoding.

    Args:
        filepath: Path to the JSON file to load.

    Returns:
        The loaded JSON data as a dictionary.
    """
    with open(filepath, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def limit_float_precision(obj: Any, num_decimals: int) -> Any:
    """Recursively limits precision of floating-point numbers in nested data structures.

    Args:
        obj: The object to process (can be dict, list, float, or other types).
        num_decimals: Number of decimal places to which we should round floating-point
            numbers.

    Returns:
        The processed object with limited floating-point precision.
    """
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), num_decimals)
    elif isinstance(obj, dict):
        return {
            key: limit_float_precision(value, num_decimals)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [limit_float_precision(item, num_decimals) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(limit_float_precision(item, num_decimals) for item in obj)
    else:
        return obj


def write_json(
    data: Any, filepath: StrPath, num_decimals: Optional[int] = None
) -> None:
    """Writes JSON-serializable data to a file with UTF-8 encoding.

    Args:
        data: The JSON-serializable data to write.
        filepath: Path where to write the JSON file.
        num_decimals: Optional number of decimal places to which we should round
            floating-point numbers.  If None, no precision limiting is applied.
    """
    if num_decimals is not None:
        data = limit_float_precision(data, num_decimals)

    with open(filepath, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=1)


def only_one_true(*args) -> bool:
    """Checks that only one of the given arguments is `True`."""

    already_found = False
    for arg in args:
        if arg:
            if already_found:
                return False
            else:
                already_found = True
    return already_found


def file_exists(filepath_or_url: str) -> bool:
    """Checks whether a given file exists and is accessible.

    Args:
        filepath_or_url:
            String representing either a local file, an `http(s)://` URL or a cloud
            location (identified by one of these prefixes: `az://`, `gs://`, `s3://`).

    Returns:
        `True` if file exists and is accessible, or `False` otherwise.
    """

    try:
        if filepath_or_url.startswith("http://") or filepath_or_url.startswith(
            "https://"
        ):
            return requests.get(
                filepath_or_url, headers=CUSTOM_HTTP_AGENT, timeout=60
            ).ok
        elif (
            filepath_or_url.startswith("az://")
            or filepath_or_url.startswith("gs://")
            or filepath_or_url.startswith("s3://")
        ):
            return CloudPath(filepath_or_url).exists()  # type: ignore
        else:
            return Path(filepath_or_url).exists()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "`%s` while loading `%s` ==> %s", type(e).__name__, filepath_or_url, e
        )
        return False


def load_rgb_image(filepath_or_url: str) -> Optional[PIL.Image.Image]:
    """Loads a file as an RGB PIL image.

    Args:
        filepath_or_url:
            String representing either a local file, an `http(s)://` URL or a cloud
            location (identified by one of these prefixes: `az://`, `gs://`, `s3://`).

    Returns:
        An RGB PIL image if the file was loaded successfully, or `None` otherwise.
    """

    try:
        if filepath_or_url.startswith("http://") or filepath_or_url.startswith(
            "https://"
        ):
            file_contents = requests.get(
                filepath_or_url, headers=CUSTOM_HTTP_AGENT, timeout=60
            ).content
            img = PIL.Image.open(BytesIO(file_contents))
        else:
            if (
                filepath_or_url.startswith("az://")
                or filepath_or_url.startswith("gs://")
                or filepath_or_url.startswith("s3://")
            ):
                path = CloudPath(filepath_or_url)  # type: ignore
            else:
                path = Path(filepath_or_url)
            img = PIL.Image.open(path)

        img.load()
        img = img.convert("RGB")
        img = PIL.ImageOps.exif_transpose(img)
        return img

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "`%s` while loading `%s` ==> %s", type(e).__name__, filepath_or_url, e
        )
        return None


def prepare_instances_dict(  # pylint: disable=too-many-positional-arguments
    instances_dict: Optional[dict] = None,
    instances_json: Optional[StrPath] = None,
    filepaths: Optional[list[StrPath]] = None,
    filepaths_txt: Optional[StrPath] = None,
    folders: Optional[list[StrPath]] = None,
    folders_txt: Optional[StrPath] = None,
    country: Optional[str] = None,
    admin1_region: Optional[str] = None,
) -> dict:
    """Transforms various input formats into an instances dict.

    The instances dict is the most expressive input format of them all since, compared
    to others, it can also express country, latitude and longitude information.

    This function expects that only one input argument is provided.

    Args:
        instances_dict:
            Optional instances dict. If provided, this function is a no-op.
        instances_json:
            Optional path to load the instances dict from.
        filepaths:
            Optional list of filepaths to process.
        filepaths_txt:
            Optional path to load the list of filepaths to process from.
        folders:
            Optional list of folders to process.
        folders_txt:
            Optional path to load the list of folders to process from.
        country:
            Optional country to enforce on all instances.
        admin1_region:
            Optional first-level administrative division to enforce on all instances.

    Returns:
        An instances dict resulted from the input transformation.

    Raises:
        ValueError:
            If more than one input argument was provided.
    """

    def _enforce_location(
        instances_dict: dict, country: Optional[str], admin1_region: Optional[str]
    ) -> dict:
        if not country:
            return instances_dict
        location_dict = {"country": country}
        if admin1_region:
            location_dict["admin1_region"] = admin1_region
        return {
            "instances": [
                instance_dict | location_dict
                for instance_dict in instances_dict["instances"]
            ]
        }

    inputs_str = (
        "["
        "instances_dict, "
        "instances_json, "
        "filepaths, "
        "filepaths_txt, "
        "folders, "
        "folders_txt"
        "]"
    )
    inputs = eval(inputs_str)  # pylint: disable=eval-used
    if not only_one_true(*inputs):
        raise ValueError(
            f"Expected exactly one of {inputs_str} to be provided. "
            f"Received: {inputs}."
        )

    if instances_json is not None:
        instances_dict = load_json(instances_json)
    if instances_dict is not None:
        return _enforce_location(instances_dict, country, admin1_region)

    if folders_txt is not None:
        with open(folders_txt, mode="r", encoding="utf-8") as fp:
            folders = [line.strip() for line in fp.readlines()]
    if folders is not None:
        filepaths = []
        for folder in folders:
            base_dir = Path(folder)
            for ext in IMG_EXTENSIONS:
                filepaths.extend(base_dir.glob(f"**/*.{ext}"))
        filepaths = sorted(set(filepaths))

    if filepaths_txt is not None:
        with open(filepaths_txt, mode="r", encoding="utf-8") as fp:
            filepaths = [line.strip() for line in fp.readlines()]
    assert filepaths is not None
    return _enforce_location(
        {
            "instances": [
                {
                    "filepath": (
                        filepath if isinstance(filepath, str) else filepath.as_posix()
                    )
                }
                for filepath in filepaths
            ]
        },
        country,
        admin1_region,
    )


def load_partial_predictions(
    predictions_json: Optional[StrPath],
    instances: list[dict],
) -> tuple[dict[str, dict], list[dict]]:
    """Loads partial predictions and filters unprocessed instances from a given list.

    Args:
        predictions_json:
            Filepath to partial predictions to load. If missing, no previous predictions
            can be reused and all instances are considered unprocessed.
        instances:
            List of instances to check if they still need to be processed. Those found
            in the partial predictions are considered already processed and are filtered
            out.

    Returns:
        A tuple made of: (a) the partial predictions dict, and (b) the list of
        unprocessed instances.

    Raises:
        RuntimeError:
            If the partial predictions contain a filepath not found in the list of
            instances to check. To fix this, make sure that the partial predictions
            originated from the same list of instances to process.
    """

    if not predictions_json:
        return {}, instances
    predictions_json = Path(predictions_json)
    if not predictions_json.exists() or not predictions_json.is_file():
        return {}, instances

    logging.info("Loading partial predictions from `%s`.", predictions_json)

    partial_predictions = {}
    target_filepaths = {instance["filepath"] for instance in instances}
    predictions_dict = load_json(predictions_json)
    for prediction in predictions_dict["predictions"]:
        filepath = prediction["filepath"]
        if filepath not in target_filepaths:
            raise RuntimeError(
                f"Filepath from loaded predictions is missing from the set of "
                f"instances to process: `{filepath}`. Make sure you're resuming "
                f"the work using the same set of instances."
            )

        if "failures" in prediction:
            continue

        partial_predictions[prediction["filepath"]] = prediction

    instances_to_process = [
        instance
        for instance in instances
        if instance["filepath"] not in partial_predictions
    ]

    return partial_predictions, instances_to_process


def save_predictions(predictions_dict: dict, output_json: StrPath) -> None:
    """Saves a predictions dict to an output file.

    Args:
        predictions_dict:
            Predictions dict to save.
        output_json:
            Output filepath where to save the predictions dict in JSON format.
    """

    output_json = Path(output_json)
    stem = output_json.stem
    suffix = output_json.suffix
    output_json_tmp = output_json.parent / f"{stem}.tmp.{uuid.uuid4()}{suffix}"
    logging.info("Saving predictions to `%s`.", output_json_tmp)
    write_json(predictions_dict, output_json_tmp, num_decimals=4)
    logging.info("Moving `%s` to `%s`.", output_json_tmp, output_json)
    output_json_tmp.replace(output_json)  # Atomic operation.
