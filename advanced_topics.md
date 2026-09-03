# Advanced topics

## Table of Contents

- [Overview](#overview)
- [Running each component of run_model separately](#running-each-component-separately)
- [Alternative installation variants of the SpeciesNet Python package](#alternative-installation-variants)
- [SpeciesNet model variants](#speciesnet-model-variants)
- [Alternative input formats for run_model](#alternative-input-formats-for-run_model)
- [Alternative output formats for run_model](#alternative-output-formats-for-run_model)
- [Contributing code](#contributing-code)
- [Build status](#build-status)

## Overview

This document contains information about running SpeciesNet that is only relevant to a small subset of users.

## Running each component separately

Rather than running everything at once, you may want to run the detection, classification, and ensemble steps separately.  You can do that like this:

- Run the detector:

  > ```python -m speciesnet.scripts.run_model --detector_only --folders "c:\your\image\folder" --predictions_json "c:\your_detector_output_file.json"```

- Run the classifier, passing the file that you just created, which contains detection results:

  > ```python -m speciesnet.scripts.run_model --classifier_only --folders "c:\your\image\folder" --predictions_json "c:\your_clasifier_output_file.json" --detections_json "c:\your_detector_output_file.json"```

- Run the ensemble step, passing both the files that you just created, which contain the detection and classification results:

  > ```python -m speciesnet.scripts.run_model --ensemble_only --folders "c:\your\image\folder" --predictions_json "c:\your_ensemble_output_file.json" --detections_json "c:\your_detector_output_file.json" --classifications_json "c:\your_clasifier_output_file.json" --country CAN```

Note that in this example, we have specified the country code only for the ensemble step; the geofencing is part of the ensemble component, so the country code is only relevant for this step.

## Alternative installation variants

Depending on how you plan to run SpeciesNet, you may want to install additional dependencies:

- Minimal requirements:

  `pip install speciesnet`

- Minimal + notebook requirements:

  `pip install speciesnet[notebooks]`

- Minimal + server requirements:

  `pip install speciesnet[server]`

- Minimal + cloud requirements (`az` / `gs` / `s3`), e.g.:

  `pip install speciesnet[gs]`

- Any combination of the above requirements, e.g.:

  `pip install speciesnet[notebooks,server]`

## SpeciesNet model variants

There are two variants of the SpeciesNet classifier, which lend themselves to different ensemble strategies:

- [v4.0.3a](model_cards/v4.0.1a.md) (default): Always-crop model, i.e. we run the detector first and crop the image to the top detection bounding box before feeding it to the species classifier.
- [v4.0.3b](model_cards/v4.0.1b.md): Full-image model, i.e. we run both the detector and the species classifier on the full image, independently.

Both links point to the model cards for the 4.0.1 models; model cards were not updated for the 4.0.3 release, which only included changes to geofencing rules and minor taxonomy updates.

run_model.py defaults to v4.0.3a, but you can specify one model or the other using the --model option, for example:

- `--model kaggle:google/speciesnet/pyTorch/v4.0.3a/1`
- `--model kaggle:google/speciesnet/pyTorch/v4.0.3b/1`

If you are a DIY type and you plan to run the models outside of our ensemble, a couple of notes:

- The crop classifier (v4.0.3a) expects images to be cropped tightly to animals, then resized to 480x480px.
- The whole-image classifier (v4.0.3b) expects images to have been cropped vertically to remove some pixels from the top and bottom, then resized to 480x480px.

See [classifier.py](https://github.com/google/cameratrapai/blob/master/speciesnet/classifier.py) to see how preprocessing is implemented for both classifiers.

## Alternative input formats for run_model

In the examples in the [main README](README.md), we demonstrate calling `run_model.py` using the `--folders` option to point to your images, and optionally using the `--country` options to tell the ensemble what country your images came from.  `run_model.py` can also load a list of images from a .json file in the following format; this is particularly useful if you want to specify different countries/states for different subsets of your images.

When you call the model, you can either prepare your requests to match this format or, in some cases, other supported formats will be converted to this automatically.

```text
{
    "instances": [
        {
            "filepath": str  => Image filepath
            "country": str (optional)  => 3-letter country code (ISO 3166-1 Alpha-3) for the location where the image was taken
            "admin1_region": str (optional)  => First-level administrative division (in ISO 3166-2 format) within the country above
            "latitude": float (optional)  => Latitude where the image was taken
            "longitude": float (optional)  => Longitude where the image was taken
        },
        ...  => A request can contain multiple instances in the format above.
    ]
}
```

admin1_region is currently only supported in the US, where valid values for admin1_region are two-letter state codes.

Latitude and longitude are only used to determine admin1_region, so if you are specifying a state code, you don't need to specify latitude and longitude.

## Alternative output formats for run_model

The main README describes the typical output format for run_model, when you are running the full ensemble (detector and classifier).  This section describes the output format when you are running only the classifier or detector.

### Classifier-only inference

```text
{
    "predictions": [
        {
            "filepath": str  => Image filepath.
            "failures": list[str] (optional)  => List of internal components that failed during prediction (in this case, only "CLASSIFIER" can be in that list). If absent, the prediction was successful.
            "classifications": {  => dict (optional)  => Top-5 classifications. Included only if "CLASSIFIER" if not part of the "failures" field.
                "classes": list[str]  => List of top-5 classes predicted by the classifier, matching the decreasing order of their scores below.
                "scores": list[float]  => List of scores corresponding to top-5 classes predicted by the classifier, in decreasing order.
                "target_classes": list[str] (optional)  => List of target classes, only present if target classes are passed as arguments.
                "target_logits": list[float] (optional)  => Raw confidence scores (logits) of the target classes, only present if target classes are passed as arguments.
            }
        },
        ...  => A response will contain one prediction for each instance in the request.
    ]
}
```

### Detector-only inference

```text
{
    "predictions": [
        {
            "filepath": str  => Image filepath.
            "failures": list[str] (optional)  => List of internal components that failed during prediction (in this case, only "DETECTOR" can be in that list). If absent, the prediction was successful.
            "detections": [  => list (optional)  => List of detections with confidence scores > 0.01, in decreasing order of their scores. Included only if "DETECTOR" if not part of the "failures" field.
                {
                    "category": str  => Detection class "1" (= animal), "2" (= human) or "3" (= vehicle) from MegaDetector's raw output.
                    "label": str  => Detection class "animal", "human" or "vehicle", matching the "category" field above. Added for readability purposes.
                    "conf": float  => Confidence score of the current detection.
                    "bbox": list[float]  => Bounding box coordinates, in (xmin, ymin, width, height) format, of the current detection. Coordinates are normalized to the [0.0, 1.0] range, relative to the image dimensions.
                },
                ...  => A prediction can contain zero or multiple detections.
            ]
        },
        ...  => A response will contain one prediction for each instance in the request.
    ]
}
```

## Contributing code

If you're interested in contributing to our repo, rather than installing via pip, we recommend cloning the repo, then creating the Python virtual environment for development using the following commands:

```bash
python -m venv .env
source .env/bin/activate
pip install -e .[dev]
```

We use the following tools for testing and validating code:

- [`pytest`](https://github.com/pytest-dev/pytest/) for running tests:

    ```bash
    pytest -vv
    ```

- [`black`](https://github.com/psf/black) for formatting code:

    ```bash
    black .
    ```

- [`isort`](https://github.com/PyCQA/isort) for sorting Python imports consistently:

    ```bash
    isort .
    ```

- [`pylint`](https://github.com/pylint-dev/pylint) for linting Python code and flag various issues:

    ```bash
    pylint . --recursive=yes
    ```

- [`pyright`](https://github.com/microsoft/pyright) for static type checking:

    ```bash
    pyright
    ```

- [`pymarkdown`](https://github.com/jackdewinter/pymarkdown) for linting Markdown files:

    ```bash
    pymarkdown scan **/*.md
    ```

Handy one-liner to run all of the code formatting/checking steps from above:

```bash
black . && isort . && pylint . --recursive=yes && pyright
```

If you submit a PR to contribute your code back to this repo, you will be asked to sign a contributor license agreement; see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Build status

[![Python tests](https://github.com/google/cameratrapai/actions/workflows/python_tests.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/python_tests.yml)
[![Python style checks](https://github.com/google/cameratrapai/actions/workflows/python_style_checks.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/python_style_checks.yml)
[![Markdown style checks](https://github.com/google/cameratrapai/actions/workflows/markdown_style_checks.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/markdown_style_checks.yml)
