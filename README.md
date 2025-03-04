# SpeciesNet

An ensemble of AI models for classifying wildlife in camera trap images.

## Table of Contents

- [Overview](#overview)
- [Running SpeciesNet](#running-speciesnet)
  - [Setting up your Python environment](#setting-up-your-python-environment)
  - [Running the models](#running-the-models)
  - [Using GPUs](#using-gpus)
  - [Running each component separately](#running-each-component-separately)
- [Downloading SpeciesNet model weights directly](#downloading-speciesnet-model-weights-directly)
- [Contacting us](#contacting-us)
- [Citing SpeciesNet](#citing-speciesnet)
- [Supported models](#supported-models)
- [Input format](#input-format)
- [Output format](#output-format)
- [Ensemble decision-making](#ensemble-decision-making)
- [Alternative installation variants](#alternative-installation-variants)
- [Contributing code](#contributing-code)
- [Animal picture](#animal-picture)
- [Build status](#build-status)

## Overview

Effective wildlife monitoring relies heavily on motion-triggered wildlife cameras, or “camera traps”, which generate vast quantities of image data. Manual processing of these images is a significant bottleneck. AI can accelerate that processing, helping conservation practitioners spend more time on conservation, and less time reviewing images.

This repository hosts code for running an ensemble of two AI models: (1) an object detector that finds objects of interest in wildlife camera images, and (2) an image classifier that classifies those objects to the species level. This ensemble is used for species recognition in the [Wildlife Insights](https://www.wildlifeinsights.org/) platform.

The object detector used in this ensemble is [MegaDetector](https://github.com/agentmorris/MegaDetector), which finds animals, humans, and vehicles in camera trap images, but does not classify animals to species level.

The species classifier ([SpeciesNet](https://www.kaggle.com/models/google/speciesnet)) was trained at Google using a large dataset of camera trap images and an [EfficientNet V2 M](https://arxiv.org/abs/2104.00574) architecture. It is designed to classify images into one of more than 2000 labels, covering diverse animal species, higher-level taxa (like "mammalia" or "felidae"), and non-animal classes ("blank", "vehicle"). SpeciesNet has been trained on a geographically diverse dataset of over 65M images, including curated images from the Wildlife Insights user community, as well as images from publicly-available repositories.

The SpeciesNet ensemble combines these two models using a set of heuristics and, optionally, geographic information to assign each image to a single category.  See the "[ensemble decision-making](#ensemble-decision-making)" section for more information about how the ensemble combines information for each image to make a single prediction.

The full details of the models and the ensemble process are discussed in this research paper:

Gadot T, Istrate Ș, Kim H, Morris D, Beery S, Birch T, Ahumada J. [To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images](https://doi.org/10.1049/cvi2.12318). IET Computer Vision. 2024 Dec;18(8):1193-208.

## Running SpeciesNet

### Setting up your Python environment

The instructions on this page will assume that you have a Python virtual environment set up.  If you have not installed Python, or you are not familiar with Python virtual environments, start with our [installing Python](installing-python.md) page.  If you see a prompt that looks something like the following, you're all set to proceed to the next step:

![speciesnet conda prompt](https://github.com/google/cameratrapai/raw/main/images/conda-prompt-speciesnet.png)

### Installing the SpeciesNet Python package

You can install the SpeciesNet Python package via:

`pip install speciesnet`

To confirm that the package has been installed, you can run:

`python -m speciesnet.scripts.run_model --help`

You should see help text related to the main script you'll use to run SpeciesNet.

### Running the models

The easiest way to run the ensemble is via the "run_model" script, like this:

> ```python -m speciesnet.scripts.run_model.py --folders "c:\your\image\folder" --predictions_json "c:\your\output\file.json"```

Change `c:\your\image\folder` to the root folder where your images live, and change `c:\your\output\file.json` to the location where you want to put the output file containing the SpeciesNet results.

This will automatically download and run the detector and the classifier.  This command periodically logs output to the output file, and if this command doesn't finish (e.g. you have to cancel or reboot), you can just run the same command, and it will pick up where it left off.

These commands produce an output file in .json format; for details about this format, and information about converting it to other formats, see the "[output format](#output-format)" section below.

You can also run the three steps (detector, classifier, ensemble) separately; see the "[running each component separately](#running-each-component-separately)" section for more information.

In the above example, we didn't tell the ensemble what part of the world your images came from, so it may, for example, predict a kangaroo for an image from England.  If you want to let our ensemble filter predictions geographically, add, for example:

`--country GBR`

You can use any [ISO 3166-1 alpha-3 three-letter country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3).

If your images are from the USA, you can also specify a state name using the two-letter state abbreviation, by adding, for example:

`--admin1_region CA`

### Using GPUs

If you don't have an NVIDIA GPU, you can ignore this section.

If you have an NVIDIA GPU, you should be able to use it for both the detection and classification steps.  However, because our detector runs in PyTorch and our classifier runs in TensorFlow, this comes with two caveats...

#### 1. TensorFlow can only use GPUs in Windows inside WSL

Recent versions of TensorFlow do not support GPUs on "native Windows".  Everything will work fine on Windows, but our code won't use your GPU to run the classifier.  However, TensorFlow *does* support GPUs in [WSL](https://learn.microsoft.com/en-us/windows/wsl/) (the Windows Subsystem for Linux), which has been available as part of Windows since Windows 10, and is installed by default in Windows 11.  WSL is like a Linux prompt that runs inside your Windows OS.  If you're using Windows, and it's working great, but you want to use your GPU, try WSL, and feel free to [email us](mailto:cameratraps@google.com) if you get stuck setting things up in WSL.

#### 2. TensorFlow and PyTorch don't usually like using the GPU in the same Python environment

Most of the time, after installing the speciesnet Python package, the GPU will be available to *either* TensorFlow or PyTorch, but not both.  You can test which framework(s) can see your GPU by running:

`python -m speciesnet.scripts.gpu_test`

You might see "No GPUs reported by PyTorch" and/or "No GPUs reported by Tensorflow".  If both frameworks show that a GPU is available, congratulations, you've won the Python IT lottery.  More commonly, TensorFlow will not see the GPU.  If this is what you observe, don't worry, everything will still work, you'll just need to run each step in a separate Python environment.  We recommend creating an extra environment in this case called "speciesnet-tf", like this:

```bash
conda create -n speciesnet-tf python=3.11 pip -y
conda activate speciesnet-tf
pip install "numpy<2.0"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --force-reinstall
pip install "tensorflow[and-cuda]==2.15.1" --force-reinstall
```

This is forcing a CPU-only installation of PyTorch in that environment (which is OK, we won't be using PyTorch in this environment), then forcing a GPU installation of TensorFlow.  After this, you should be able to "[running each component separately](#running-each-component-separately)", just be sure to activate the "speciesnet" environment before running the detector, and the "speciesnet-tf" environment before running the classifier.

If this approach isn't working as advertised, [let us know](mailto:cameratraps@google.com).

### Running each component separately

Rather than running everything at once, you may want to run the detection, classification, and ensemble steps separately.  You can do that like this:

- Run the detector:

  > ```python -m speciesnet.scripts.run_model.py --detector_only --folders "c:\your\image\folder" --predictions_json "c:\your_detector_output_file.json"```
  
- Run the classifier, passing the file that you just created, which contains detection results:  

  > ```python -m speciesnet.scripts.run_model.py --classifier_only --folders "c:\your\image\folder" --predictions_json "c:\your_clasifier_output_file.json" --detections_json "c:\your_detector_output_file.json"```
  
- Run the ensemble step, passing both the files that you just created, which contain the detection and classification results:  

  > ```python -m speciesnet.scripts.run_model.py --ensemble_only --folders "c:\your\image\folder" --predictions_json "c:\your_ensemble_output_file.json" --detections_json "c:\your_detector_output_file.json" --classifications_json "c:\your_clasifier_output_file.json"```  

## Downloading SpeciesNet model weights directly

The `run_model.py` script recommended above will download model weights automatically.  If you want to use the SpeciesNet model weights outside of our script, or if you plan to be offline when you first run the script, you can download model weights directly from Kaggle.  Running our ensemble also requires [MegaDetector](https://github.com/agentmorris/MegaDetector), so in this list of links, we also include a direct link to the MegaDetector model weights.

* [SpeciesNet page on Kaggle](https://www.kaggle.com/models/google/speciesnet)
* [Direct link to version 4.0.0a weights](https://www.kaggle.com/api/v1/models/google/speciesnet/keras/v4.0.0a/3/download) (the crop classifier)
* [Direct link to version 4.0.0b weights](https://www.kaggle.com/api/v1/models/google/speciesnet/keras/v4.0.0b/3/download) (the whole-image classifier)
* [Direct link to MegaDetector weights](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)

## Contacting us

If you have issues or questions, either [file an issue](https://github.com/google/cameratrapai/issues) or email us at [cameratraps@google.com](mailto:cameratraps@google.com).

## Citing SpeciesNet

If you use this model, please cite:

```text
@article{gadot2024crop,
  title={To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images},
  author={Gadot, Tomer and Istrate, Ștefan and Kim, Hyungwon and Morris, Dan and Beery, Sara and Birch, Tanya and Ahumada, Jorge},
  journal={IET Computer Vision},
  year={2024},
  publisher={Wiley Online Library}
}
```

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

## Supported models

There are two variants of the SpeciesNet classifier, which lend themselves to different ensemble strategies:

- [v4.0.0a](model_cards/v4.0.0a) (default): Always-crop model, i.e. we run the detector first and crop the image to the top detection bounding box before feeding it to the species classifier.
- [v4.0.0b](model_cards/v4.0.0b): Full-image model, i.e. we run both the detector and the species classifier on the full image, independently.

run_model.py defaults to v4.0.0a, but you can specify one model or the other using the --model option, for example:

- `--model kaggle:google/speciesnet/keras/v4.0.0a`
- `--model kaggle:google/speciesnet/keras/v4.0.0b`

If you are a DIY type and you plan to run the models outside of our ensemble, a couple of notes:

- The crop classifier (v4.0.0a) expects images to be cropped tightly to animals, then resized to 480x480px.
- The whole-image classifier (v4.0.0b) expects images to have been cropped vertically to remove some pixels from the top and bottom, then resized to 480x480px.

See [classifier.py](https://github.com/google/cameratrapai/blob/master/speciesnet/classifier.py) to see how preprocessing is implemented for both classifiers.

## Input format

In the above examples, we demonstrate calling `run_model.py` using the `--folders` option to point to your images, and optionally using the `--country` options to tell the ensemble what country your images came from.  `run_model.py` can also load a list of images from a .json file in the following format; this is particularly useful if you want to specify different countries/states for different subsets of your images.

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

## Output format

`run_model.py` produces output in .json format, containing an array called "predictions", with one element per image.  We provide a script to convert this format to the format used by [MegaDetector](https://github.com/agentmorris/MegaDetector), which can be imported into [Timelapse](https://timelapse.ucalgary.ca/), see [speciesnet_to_md.py](speciesnet/scripts/speciesnet_to_md.py).

Each element always contains  field called "filepath"; the exact content of those elements will vary depending on which elements of the ensemble you ran.

### Full ensemble

```text
{
    "predictions": [
        {
            "filepath": str  => Image filepath.
            "failures": list[str] (optional)  => List of internal components that failed during prediction (e.g. "CLASSIFIER", "DETECTOR", "GEOLOCATION"). If absent, the prediction was successful.
            "country": str (optional)  => 3-letter country code (ISO 3166-1 Alpha-3) for the location where the image was taken. It can be overwritten if the country from the request doesn't match the country of (latitude, longitude).
            "admin1_region": str (optional)  => First-level administrative division (in ISO 3166-2 format) within the country above. If not provided in the request, it can be computed from (latitude, longitude) when those coordinates are specified. Included in the response only for some countries that are used in geofencing (e.g. "USA").
            "latitude": float (optional)  => Latitude where the image was taken, included only if (latitude, longitude) were present in the request.
            "longitude": float (optional)  => Longitude where the image was taken, included only if (latitude, longitude) were present in the request.
            "classifications": {  => dict (optional)  => Top-5 classifications. Included only if "CLASSIFIER" if not part of the "failures" field.
                "classes": list[str]  => List of top-5 classes predicted by the classifier, matching the decreasing order of their scores below.
                "scores": list[float]  => List of scores corresponding to top-5 classes predicted by the classifier, in decreasing order.
            },
            "detections": [  => list (optional)  => List of detections with confidence scores > 0.01, in decreasing order of their scores. Included only if "DETECTOR" if not part of the "failures" field.
                {
                    "category": str  => Detection class "1" (= animal), "2" (= human) or "3" (= vehicle) from MegaDetector's raw output.
                    "label": str  => Detection class "animal", "human" or "vehicle", matching the "category" field above. Added for readability purposes.
                    "conf": float  => Confidence score of the current detection.
                    "bbox": list[float]  => Bounding box coordinates, in (xmin, ymin, width, height) format, of the current detection. Coordinates are normalized to the [0.0, 1.0] range, relative to the image dimensions.
                },
                ...  => A prediction can contain zero or multiple detections.
            ],
            "prediction": str (optional)  => Final prediction of the SpeciesNet ensemble. Included only if "CLASSIFIER" and "DETECTOR" are not part of the "failures" field.
            "prediction_score": float (optional)  => Final prediction score of the SpeciesNet ensemble. Included only if the "prediction" field above is included.
            "prediction_source": str (optional)  => Internal component that produced the final prediction. Used to collect information about which parts of the SpeciesNet ensemble fired. Included only if the "prediction" field above is included.
            "model_version": str  => A string representing the version of the model that produced the current prediction.
        },
        ...  => A response will contain one prediction for each instance in the request.
    ]
}
```

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

## Ensemble decision-making

The SpeciesNet ensemble uses multiple steps to predict a single category for each image, combining the strengths of the detector and the classifier.

The ensembling strategy was primarily optimized for minimizing the human effort required to review collections of images. To do that, the guiding principles are:

- Help users to quickly filter out unwanted images (e.g., blanks): identify as many blank images as possible while minimizing missed animals, which can be more costly than misclassifying a non-blank image as one of the possible animal classes.
- Provide high-confidence predictions for frequent classes (e.g., deer).
- Make predictions on the lowest taxonomic level possible, while balancing precision: if the ensemble is not confident enough all the way to the species level, we would rather return a prediction we are confident about in a higher taxonomic level (e.g., family, or sometimes even "animal"), instead of risking an incorrect prediction on the species level.

Here is a breakdown of the different steps:

1. **Input processing:** Raw images are preprocessed and passed to both the object detector (MegaDetector) and the image classifier. The type of preprocessing will depend on the selected model. For "always crop" models, images are first processed by the object detector and then cropped based on the detection bounding box before being fed to the classifier. For "full image" models, images are preprocessed independently for both models.

2. **Object detection:** The detector identifies potential objects (animals, humans, or vehicles) in the image, providing their bounding box coordinates and confidence scores.

3. **Species classification:** The species classifier analyzes the (potentially cropped) image to identify the most likely species present. It provides a list of top-5 species classifications, each with a confidence score. The species classifier is a fully supervised model that classifies images into a fixed set of animal species, higher taxa, and non-animal labels.

4. **Detection-based human/vehicle decisions:** If the detector is highly confident about the presence of a human or vehicle, that label will be returned as the final prediction regardless of what the classifier predicts. If the detection is less confident and the classifier also returns human or vehicle as a top-5 prediction, with a reasonable score, that top prediction will be returned. This step prevents high-confidence detector predictions from being overridden by lower-confidence classifier predictions.

5. **Blank decisions:** If the classifier predicts "blank" with a high confidence score, and the detector has very low confidence about the presence of an animal (or is absent), that "blank" label is returned as a final prediction. Similarly, if a classification is "blank" with extra-high confidence (above 0.99), that label is returned as a final prediction regardless of the detector's output. This enables the model to filter out images with high confidence in being blank.

6. **Geofencing:** If the most likely species is an animal and a location (country and optional admin1 region) is provided for the image, a geofencing rule is applied. If that species is explicitly disallowed for that region based on the available geofencing rules, the prediction will be rolled up (as explained below) to a higher taxa level on that allow list.

7. **Label rollup:** If all of the previous steps do not yield a final prediction, a "rollup" is applied when there is a good classification score for an animal. "Rollup" is the process of propagating the classification predictions to the first matching ancestor in the taxonomy, provided there is a good score at that level. This means the model may assign classifications at the genus, family, order, class, or kingdom level, if those scores are higher than the score at the species level. This is a common strategy to handle long-tail distributions, common in wildlife datasets.

8. **Detection-based animal decisions:**  If the detector has a reasonable confidence `animal` prediction, `animal` will be returned along with the detector confidence.

9. **Unknown:** If no other rule applies, the `unknown` class is returned as the final prediction, to avoid making low-confidence predictions.

10. **Prediction source:** At each step of the prediction workflow, a `prediction_source` is stored. This will be included in the final results to help diagnose which parts of the overall SpeciesNet ensemble were actually used.

## Contributing code

If you're interested in contributing to our repo, rather than installing via pip, we recommend cloning the repo, then creating the Python virtual environment for development using the following commands:

```bash
python -m venv .env
source .env/bin/activate
pip install -e .[dev]
```

We use the following coding conventions:

- [`black`](https://github.com/psf/black) for code formatting:

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

- [`pytest`](https://github.com/pytest-dev/pytest/) for testing our code:

    ```bash
    pytest -vv
    ```

- [`pymarkdown`](https://github.com/jackdewinter/pymarkdown) for linting Markdown files:

    ```bash
    pymarkdown scan **/*.md
    ```

If you submit a PR to contribute your code back to this repo, you will be asked to sign a contributor license agreement; see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Animal picture

It would be unfortunate if this whole README about camera trap images didn't show you a single camera trap image, so...

![giant armadillo](https://github.com/google/cameratrapai/raw/main/images/sample_image_oct.jpg)

Image credit University of Minnesota, from the [Orinoquía Camera Traps](https://lila.science/datasets/orinoquia-camera-traps/) dataset.

## Build status

[![Python tests](https://github.com/google/cameratrapai/actions/workflows/python_tests.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/python_tests.yml)
[![Python style checks](https://github.com/google/cameratrapai/actions/workflows/python_style_checks.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/python_style_checks.yml)
[![Markdown style checks](https://github.com/google/cameratrapai/actions/workflows/markdown_style_checks.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/markdown_style_checks.yml)
