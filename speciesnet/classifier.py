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

"""Classifier functionality of SpeciesNet.

Defines the SpeciesNetClassifier class, responsible for image classification for
SpeciesNet. It handles loading of classification models, preprocessing of input images,
and generating species predictions.
"""

__all__ = [
    "SpeciesNetClassifier",
]

import time
from typing import Any, Optional

from absl import logging
from humanfriendly import format_timespan
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F

from speciesnet.constants import Failure
from speciesnet.constants import mps_inference_lock
from speciesnet.utils import BBox
from speciesnet.utils import ModelInfo
from speciesnet.utils import PreprocessedImage


class SpeciesNetClassifier:
    """Classifier component of SpeciesNet."""

    IMG_SIZE = 480
    MAX_CROP_RATIO = 0.3
    MAX_CROP_SIZE = 400

    def __init__(
        self,
        model_name: str,
        target_species_txt: Optional[str] = None,
        device: Optional[str] = None,
        force_model_download: bool = False,
    ) -> None:
        """Loads the classifier resources.

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
            device:
                Specific device identifier, e.g. "cpu" or "cuda".  If None, "cuda"
                and "mps" will be used if available.
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

        # Select the best device available.
        if device is not None:
            logging.info("Using caller-supplied device %s.", device)
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Load the model.
        logging.info("Loading model from %s.", self.model_info.classifier)
        self.model = torch.load(
            self.model_info.classifier, map_location=self.device, weights_only=False
        )

        # Set the model in inference mode.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Load the labels.
        with open(self.model_info.classifier_labels, mode="r", encoding="utf-8") as fp:
            self.labels = {idx: line.strip() for idx, line in enumerate(fp.readlines())}

        # Load optional target labels.
        if target_species_txt is not None:
            with open(target_species_txt, mode="r", encoding="utf-8") as fp:
                self.target_labels = [
                    line.strip()
                    for line in fp.readlines()
                    if line.strip() in self.labels.values()
                ]
            labels_to_idx = {label: idx for idx, label in self.labels.items()}
            self.target_idx = [labels_to_idx[label] for label in self.target_labels]

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetClassifier in %s on %s.",
            format_timespan(end_time - start_time),
            self.device.upper(),
        )

    def preprocess(
        self,
        img: Optional[PIL.Image.Image],
        bboxes: Optional[list[BBox]] = None,
        resize: bool = True,
    ) -> Optional[list[PreprocessedImage]]:
        """Preprocesses an image according to this classifier's needs.

        This method prepares an input image for classification. It handles
        image loading, cropping, and resizing to the expected
        input size for the classifier model.

        In `always_crop` mode images are cropped according to the bounding boxes
        provided, producing one crop per bounding box (and no crops at all if no
        bounding boxes are provided). In `full_image` mode, bounding boxes are ignored,
        and the top and bottom of the image are cropped to prevent learning
        correlations between camera brand and species priors, producing a single crop.
        See the paper for more details.

        Args:
            img:
                PIL image to preprocess. If `None`, no preprocessing is performed.
            bboxes:
                Optional list of bounding boxes. Needed for some types of classifiers to
                crop the image to specific bounding boxes during preprocessing.
            resize:
                Whether to resize the image to some expected dimensions.

        Returns:
            A list of preprocessed images (one element per bounding box in `always_crop`
            mode, a single element in `full_image` mode), or `None` if no image was
            provided.
        """
        if img is None:
            return None

        img_tensor = F.pil_to_tensor(img)  # HWC to CHW.
        img_tensor = F.convert_image_dtype(img_tensor, torch.float32)

        results = []
        if self.model_info.type_ == "always_crop":
            for bbox in bboxes or []:
                crop = F.crop(
                    img_tensor,
                    int(bbox.ymin * img.height),
                    int(bbox.xmin * img.width),
                    int(bbox.height * img.height),
                    int(bbox.width * img.width),
                )
                if resize:
                    crop = F.resize(
                        crop,
                        [
                            SpeciesNetClassifier.IMG_SIZE,
                            SpeciesNetClassifier.IMG_SIZE,
                        ],
                        antialias=False,
                    )
                crop = F.convert_image_dtype(crop, torch.uint8)
                crop = crop.permute([1, 2, 0])  # CHW to HWC.
                results.append(PreprocessedImage(crop.numpy(), img.width, img.height))

        elif self.model_info.type_ == "full_image":
            # Crop top and bottom of image.
            target_height = max(
                int(img.height * (1.0 - SpeciesNetClassifier.MAX_CROP_RATIO)),
                img.height - SpeciesNetClassifier.MAX_CROP_SIZE,
            )
            crop = F.center_crop(img_tensor, [target_height, img.width])

            if resize:
                crop = F.resize(
                    crop,
                    [SpeciesNetClassifier.IMG_SIZE, SpeciesNetClassifier.IMG_SIZE],
                    antialias=False,
                )

            crop = F.convert_image_dtype(crop, torch.uint8)
            crop = crop.permute([1, 2, 0])  # CHW to HWC.
            results.append(PreprocessedImage(crop.numpy(), img.width, img.height))

        return results

    def predict(
        self,
        filepath: str,
        imgs: Optional[list[PreprocessedImage]],
        num_classes: Optional[int] = 5,
    ) -> dict[str, Any]:
        """Runs inference on the preprocessed crops of a given image.

        Args:
            filepath:
                Location of image to run inference on. Used for reporting purposes only,
                and not for loading the image.
            imgs:
                List of preprocessed image crops to run inference on. If `None`, a
                failure message is reported back. If empty, there is nothing to
                classify and an empty list of classifications is reported back.
            num_classes:
                Number of top classes to report for each crop. If `None`, all classes
                are reported.

        Returns:
            A dict containing either the top classifications for each of the given
            image crops (in decreasing order of confidence scores), or a failure message
            if no preprocessed image was provided.
        """

        return self.batch_predict([filepath], [imgs], num_classes=num_classes)[0]

    def batch_predict(
        self,
        filepaths: list[str],
        imgs_list: list[Optional[list[PreprocessedImage]]],
        num_classes: Optional[int] = 5,
    ) -> list[dict[str, Any]]:
        """Runs inference on a batch of preprocessed images.

        Args:
            filepaths:
                List of image locations to run inference on. Used for reporting purposes
                only, and not for loading the images.
            imgs_list:
                List of lists of preprocessed image crops to run inference on, one list
                per image. If a list is `None`, a corresponding failure message is
                reported back. If a list is empty, an empty list of classifications is
                reported back.
            num_classes:
                Number of top classes to report for each crop. If `None`, all classes
                are reported.

        Returns:
            A list of dict results. Each dict result contains `classifications_list`
            with the top classifications for each corresponding crop image, in the same
            order as the crops (and in decreasing order of confidence scores for each
            crop), or a failure message if no preprocessed image was provided.
        """

        predictions = {}

        inference_filepaths = []
        batch_arr = []

        for filepath, imgs in zip(filepaths, imgs_list):
            if imgs is None:
                predictions[filepath] = {
                    "filepath": filepath,
                    "failures": [Failure.CLASSIFIER.name],
                }
                continue

            predictions[filepath] = {"filepath": filepath, "classifications_list": []}

            for img in imgs:
                inference_filepaths.append(filepath)
                batch_arr.append(img.arr / 255.0)

        if not batch_arr:
            return [predictions[fp] for fp in filepaths if fp in predictions]

        batch_arr = np.stack(batch_arr, axis=0, dtype=np.float32)

        # Process crops in chunks of at most `max_crop_batch_size` so that a batch of
        # images with many bounding boxes does not cause excessive memory allocation
        # (especially on MPS unified memory) or hold the MPS lock for too long.
        max_crop_batch_size = 8
        logits_list = []
        for idx in range(0, len(batch_arr), max_crop_batch_size):
            chunk_arr = batch_arr[idx : idx + max_crop_batch_size]
            # On MPS devices, make sure all MPS operations happen serially.  This
            # lock is a no-op on non-MPS devices.
            with mps_inference_lock(self.device):
                chunk_tensor = torch.from_numpy(chunk_arr).to(self.device)
                logits_list.append(self.model(chunk_tensor).cpu())
                del chunk_tensor
        if self.device == "mps":
            with mps_inference_lock(self.device):
                torch.mps.empty_cache()
        logits = torch.cat(logits_list, dim=0)
        scores = torch.softmax(logits, dim=-1)
        num_labels = scores.shape[-1]
        k = num_labels if num_classes is None else min(num_classes, num_labels)
        scores, indices = torch.topk(scores, k=k, dim=-1)

        for crop_idx, (filepath, scores_arr, indices_arr) in enumerate(
            zip(
                inference_filepaths,
                scores.numpy(),
                indices.numpy(),
            )
        ):

            classification = {
                "classes": [self.labels[idx] for idx in indices_arr],
                "scores": scores_arr.tolist(),
            }

            if hasattr(self, "target_idx"):
                classification.update(
                    {
                        "target_classes": self.target_labels,
                        "target_logits": [
                            float(logits[crop_idx][idx]) for idx in self.target_idx
                        ],
                    }
                )

            predictions[filepath]["classifications_list"].append(classification)

        return [
            predictions[filepath] for filepath in filepaths if filepath in predictions
        ]
