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
- [Visualizing SpeciesNet output](#visualizing-speciesnet-output)
- [Ensemble decision-making](#ensemble-decision-making)
- [Alternative installation variants](#alternative-installation-variants)
- [Contributing code](#contributing-code)
- [Animal picture](#animal-picture)
- [Build status](#build-status)

## Overview

Effective wildlife monitoring relies heavily on motion-triggered wildlife cameras, or “camera traps”, which generate vast quantities of image data. Manual processing of these images is a significant bottleneck. AI can accelerate that processing, helping conservation practitioners spend more time on conservation, and less time reviewing images.

This repository hosts code for running an ensemble of two AI models: (1) an object detector that finds objects of interest in wildlife camera images, and (2) an image classifier that classifies those objects to the species level. This ensemble is used for species recognition in the [Wildlife Insights](https://www.wildlifeinsights.org/) platform.

The object detector used in this ensemble is [MegaDetector](https://github.com/agentmorris/MegaDetector), which finds animals, humans, and vehicles in camera trap images, but does not classify animals to species level.

The species classifier ([SpeciesNet](https://www.kaggle.com/models/google/speciesnet)) was trained at Google using a large dataset of camera trap images and an [EfficientNet V2 M](https://arxiv.org/abs/2104.00298) architecture. It is designed to classify images into one of more than 2000 labels, covering diverse animal species, higher-level taxa (like "mammalia" or "felidae"), and non-animal classes ("blank", "vehicle"). SpeciesNet has been trained on a geographically diverse dataset of over 65M images, including curated images from the Wildlife Insights user community, as well as images from publicly-available repositories.

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

If you are on a Mac, and you receive an error during this step, add the "--use-pep517" option, like this:

`pip install speciesnet --use-pep517`

To confirm that the package has been installed, you can run:

`python -m speciesnet.scripts.run_model --help`

You should see help text related to the main script you'll use to run SpeciesNet.

### Running the models

The easiest way to run the ensemble is via the "run_model" script, like this:

> ```python -m speciesnet.scripts.run_model --folders "c:\your\image\folder" --predictions_json "c:\your\output\file.json"```

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

If you have an NVIDIA GPU, SpeciesNet should use it.  If SpeciesNet is using your GPU, when you start `run_model`, in the output, you will see something like this:

<pre>Loaded SpeciesNetClassifier in 0.96 seconds on <b>CUDA</b>.
Loaded SpeciesNetDetector in 0.7 seconds on <b>CUDA</b></pre>

"CUDA" is good news, that means "GPU".  

If SpeciesNet is <i>not</i> using your GPU, you will see something like this instead:

<pre>Loaded SpeciesNetClassifier in 9.45 seconds on <b>CPU</b>
Loaded SpeciesNetDetector in 0.57 seconds on <b>CPU</b></pre>

You can also directly check whether SpeciesNet can see your GPU by running:

`python -m speciesnet.scripts.gpu_test`

99% of the time, after you install SpeciesNet on Linux, it will correctly see your GPU right away.  On Windows, you will likely need to take at least one more step:

1. Install the GPU version of PyTorch, by activating your speciesnet Python environment (e.g. by running "conda activate speciesnet"), then running:

   > ```pip install torch torchvision --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu118```
   
2. If the GPU doesn't work immediately after that step, update your [GPU driver](https://www.nvidia.com/en-us/geforce/drivers/), then reboot.  Really, don't skip the reboot part, most problems related to GPU access can be fixed by upgrading your driver and rebooting.


### Running each component separately

Rather than running everything at once, you may want to run the detection, classification, and ensemble steps separately.  You can do that like this:

- Run the detector:

  > ```python -m speciesnet.scripts.run_model --detector_only --folders "c:\your\image\folder" --predictions_json "c:\your_detector_output_file.json"```
  
- Run the classifier, passing the file that you just created, which contains detection results:  

  > ```python -m speciesnet.scripts.run_model --classifier_only --folders "c:\your\image\folder" --predictions_json "c:\your_clasifier_output_file.json" --detections_json "c:\your_detector_output_file.json"```
  
- Run the ensemble step, passing both the files that you just created, which contain the detection and classification results:  

  > ```python -m speciesnet.scripts.run_model --ensemble_only --folders "c:\your\image\folder" --predictions_json "c:\your_ensemble_output_file.json" --detections_json "c:\your_detector_output_file.json" --classifications_json "c:\your_clasifier_output_file.json" --country CAN```  
  
Note that in this example, we have specified the country code only for the ensemble step; the geofencing is part of the ensemble component, so the country code is only relevant for this step.

## Downloading SpeciesNet model weights directly

The `run_model.py` script recommended above will download model weights automatically.  If you want to use the SpeciesNet model weights outside of our script, or if you plan to be offline when you first run the script, you can download model weights directly from Kaggle.  Running our ensemble also requires [MegaDetector](https://github.com/agentmorris/MegaDetector), so in this list of links, we also include a direct link to the MegaDetector model weights.

- [SpeciesNet page on Kaggle](https://www.kaggle.com/models/google/speciesnet)
- [Direct link to version 4.0.2a weights](https://www.kaggle.com/api/v1/models/google/speciesnet/pyTorch/v4.0.2a/1/download) (the crop classifier)
- [Direct link to version 4.0.2b weights](https://www.kaggle.com/api/v1/models/google/speciesnet/pyTorch/v4.0.2b/1/download) (the whole-image classifier)
- [Direct link to MegaDetector weights](https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt)

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

- [v4.0.2a](model_cards/v4.0.1a.md) (default): Always-crop model, i.e. we run the detector first and crop the image to the top detection bounding box before feeding it to the species classifier.
- [v4.0.2b](model_cards/v4.0.1b.md): Full-image model, i.e. we run both the detector and the species classifier on the full image, independently.

Both links point to the model cards for the 4.0.1 models; model cards were not updated for the 4.0.2 release, which only included changes to geofencing rules and minor taxonomy updates.

run_model.py defaults to v4.0.2a, but you can specify one model or the other using the --model option, for example:

- `--model kaggle:google/speciesnet/pyTorch/v4.0.2a/1`
- `--model kaggle:google/speciesnet/pyTorch/v4.0.2b/1`

If you are a DIY type and you plan to run the models outside of our ensemble, a couple of notes:

- The crop classifier (v4.0.2a) expects images to be cropped tightly to animals, then resized to 480x480px.
- The whole-image classifier (v4.0.2b) expects images to have been cropped vertically to remove some pixels from the top and bottom, then resized to 480x480px.

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

In the full ensemble output, the "classifications" field contains raw classifier output, before geofencing is applied.  So even if you specify a country code, you may see taxa in the "classifications" field that are not found in the country you specified.  The "prediction" field is the result of integrating the classification, detection, and geofencing information; if you specify a country code, the "prediction" field should only contain taxa that are found in the country you specified.

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
                "target_classes": list[str] (optional)  => List of target classes, only present if target classes are passed as arguments.
                "target_logits": list[float] (optional)  => Raw confidence scores (logits) of the target classes, only present if target classes are passed as arguments.
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

## Visualizing SpeciesNet output

As per above, many users will work with SpeciesNet results in open-source tools like [Timelapse](https://timelapse.ucalgary.ca/), which support the file format used by [MegaDetector](https://github.com/agentmorris/MegaDetector) (the format is described [here](https://lila.science/megadetector-output-format)).  Consequently, we provide a [speciesnet_to_md](speciesnet/scripts/speciesnet_to_md.py) script to convert from the SpeciesNet output format to this format.

However, if you want to use the command line or Python code to visualize SpeciesNet results, we recommend using the visualization tools provided in the [megadetector-utils Python package](https://pypi.org/project/megadetector-utils/).  For example, if you just ran SpeciesNet on some images like this:

```bash
IMAGE_DIR=/path/to/your/images
python -m speciesnet.scripts.run_model --folders ${IMAGE_DIR} --predictions_json ${IMAGE_DIR}/speciesnet-results.json
```

You can use the [visualize_detector_output](https://megadetector.readthedocs.io/en/latest/visualization.html#visualize_detector_output---CLI-interface) script from the megadetector-utils package, like this:

```bash
PREVIEW_DIR=/wherever/you/want/the/output
pip install megadetector-utils
python -m megadetector.visualization.visualize_detector_output ${IMAGE_DIR}/speciesnet-results.json ${PREVIEW_DIR}
```

That will produce a folder of images with SpeciesNet results visualized on each image.  A typical use of this script would also use the --sample argument (to render a random subset of images, if what you want is to quickly grok how SpeciesNet did on a large dataset), and often the --html_output_file argument, to wrap the results in an HTML page that makes it quick to scroll through them.  Putting those together will give you pages like these:

* [Fun preview page for Caltech Camera Traps](https://lila.science/public/speciesnet-previews/speciesnet-visualization-examples/caltech-camera-traps/)
* [Fun preview page for Idaho Camera Traps](https://lila.science/public/speciesnet-previews/speciesnet-visualization-examples/idaho-camera-traps/)
* [Fun preview page for Orinoquía Camera Traps](https://lila.science/public/speciesnet-previews/speciesnet-visualization-examples/orinoquia-camera-traps/)

To see all the options, run:

```bash
 python -m megadetector.visualization.visualize_detector_output --help
```

The other relevant script is [postprocess_batch_results](https://megadetector.readthedocs.io/en/latest/postprocessing.html#postprocess_batch_results---CLI-interface), which also renders sample images, but instead of just putting them in a flat folder, the purpose of this script is to allow you to quickly see samples of detections/non-detections, and to quickly see samples broken out by species.  So, for example, you can do:

```bash
python -m megadetector.postprocessing.postprocess_batch_results ${IMAGE_DIR}/speciesnet-results.json ${PREVIEW_DIR}
```

...to get pages like these:

* [Fancy postprocessing page for Caltech Camera Traps](https://lila.science/public/speciesnet-previews/speciesnet-postprocessing-examples/caltech-camera-traps/)
* [Fancy postprocessing page for Idaho Camera Traps](https://lila.science/public/speciesnet-previews/speciesnet-postprocessing-examples/idaho-camera-traps/)
* [Fancy postprocessing page for Orinoquía Camera Traps](https://lila.science/public/speciesnet-previews/speciesnet-postprocessing-examples/orinoquia-camera-traps/)

To see all the options, run:

```bash
python -m megadetector.postprocessing.postprocess_batch_results --help
```

Both of these modules can also be called from Python code instead of from the command line.

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

`black . && isort . && pylint . --recursive=yes && pyright`

If you submit a PR to contribute your code back to this repo, you will be asked to sign a contributor license agreement; see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Animal picture

It would be unfortunate if this whole README about camera trap images didn't show you a single camera trap image, so...

![giant armadillo](https://github.com/google/cameratrapai/raw/main/images/sample_image_oct.jpg)

Image credit University of Minnesota, from the [Orinoquía Camera Traps](https://lila.science/datasets/orinoquia-camera-traps/) dataset.

## Build status

[![Python tests](https://github.com/google/cameratrapai/actions/workflows/python_tests.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/python_tests.yml)
[![Python style checks](https://github.com/google/cameratrapai/actions/workflows/python_style_checks.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/python_style_checks.yml)
[![Markdown style checks](https://github.com/google/cameratrapai/actions/workflows/markdown_style_checks.yml/badge.svg)](https://github.com/google/cameratrapai/actions/workflows/markdown_style_checks.yml)
