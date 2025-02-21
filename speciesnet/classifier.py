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
and generating species predictions. The classifier uses TensorFlow and Keras for model
loading and inference.
"""

__all__ = [
    "SpeciesNetClassifier",
]

import time
from typing import Any, Optional

from absl import logging
from humanfriendly import format_timespan
import numpy as np
import PIL.ExifTags
import PIL.Image
import tensorflow as tf

from speciesnet.constants import Failure
from speciesnet.utils import BBox
from speciesnet.utils import ModelInfo
from speciesnet.utils import PreprocessedImage


class SpeciesNetClassifier:
    """Classifier component of SpeciesNet."""

    IMG_SIZE = 480
    MAX_CROP_RATIO = 0.3
    MAX_CROP_SIZE = 400

    def __init__(self, model_name: str) -> None:
        """Loads the classifier resources.

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
        """

        start_time = time.time()

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model_info = ModelInfo(model_name)
        self.model = tf.keras.models.load_model(
            self.model_info.classifier, compile=False
        )
        with open(self.model_info.classifier_labels, mode="r", encoding="utf-8") as fp:
            self.labels = {idx: line.strip() for idx, line in enumerate(fp.readlines())}

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetClassifier in %s.",
            format_timespan(end_time - start_time),
        )

    def preprocess(
        self,
        img: Optional[PIL.Image.Image],
        bboxes: Optional[list[BBox]] = None,
        resize: bool = True,
    ) -> Optional[PreprocessedImage]:
        """Preprocesses an image according to this classifier's needs.

        This method prepares an input image for classification. It handles
        image loading, cropping, and resizing to the expected
        input size for the classifier model.

        In `always_crop` mode images are cropped according to the bounding boxes
        provided. In `full_image` mode the top and bottom of the image are cropped
        to prevent learning correlations between camera brand and species priors.
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
            A preprocessed image, or `None` if no PIL image was provided initially.
        """

        if img is None:
            return None

        with tf.device("/cpu"):
            img_tensor = tf.convert_to_tensor(img)
            img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)

            if self.model_info.type_ == "always_crop":
                # Crop to top bbox if available, otherwise leave image uncropped.
                if bboxes:
                    img_tensor = tf.image.crop_to_bounding_box(
                        img_tensor,
                        int(bboxes[0].ymin * img.height),
                        int(bboxes[0].xmin * img.width),
                        int(bboxes[0].height * img.height),
                        int(bboxes[0].width * img.width),
                    )
            elif self.model_info.type_ == "full_image":
                # Crop top and bottom of image.
                target_height = tf.cast(
                    tf.math.floor(
                        tf.multiply(
                            tf.cast(tf.shape(img_tensor)[0], tf.float32),
                            tf.constant(1.0 - SpeciesNetClassifier.MAX_CROP_RATIO),
                        )
                    ),
                    tf.int32,
                )
                target_height = tf.math.maximum(
                    target_height,
                    tf.shape(img_tensor)[0] - SpeciesNetClassifier.MAX_CROP_SIZE,
                )
                img_tensor = tf.image.resize_with_crop_or_pad(
                    img_tensor, target_height, tf.shape(img_tensor)[1]
                )

            if resize:
                img_tensor = tf.image.resize(
                    img_tensor,
                    [SpeciesNetClassifier.IMG_SIZE, SpeciesNetClassifier.IMG_SIZE],
                )

            img_tensor = tf.image.convert_image_dtype(img_tensor, tf.uint8)
            return PreprocessedImage(img_tensor.numpy(), img.width, img.height)

    def predict(
        self, filepath: str, img: Optional[PreprocessedImage]
    ) -> dict[str, Any]:
        """Runs inference on a given preprocessed image.

        Args:
            filepath:
                Location of image to run inference on. Used for reporting purposes only,
                and not for loading the image.
            img:
                Preprocessed image to run inference on. If `None`, a failure message is
                reported back.

        Returns:
            A dict containing either the top-5 classifications for the given image (in
            decreasing order of confidence scores), or a failure message if no
            preprocessed image was provided.
        """

        return self.batch_predict([filepath], [img])[0]

    def batch_predict(
        self, filepaths: list[str], imgs: list[Optional[PreprocessedImage]]
    ) -> list[dict[str, Any]]:
        """Runs inference on a batch of preprocessed images.

        Args:
            filepaths:
                List of image locations to run inference on. Used for reporting purposes
                only, and not for loading the images.
            imgs:
                List of preprocessed images to run inference on. If an image is `None`,
                a corresponding failure message is reported back.

        Returns:
            A list of dict results. Each dict result contains either the top-5
            classifications for the corresponding image (in decreasing order of
            confidence scores), or a failure message if no preprocessed image was
            provided.
        """

        predictions = {}

        inference_filepaths = []
        batch_arr = []
        for filepath, img in zip(filepaths, imgs):
            if img is None:
                predictions[filepath] = {
                    "filepath": filepath,
                    "failures": [Failure.CLASSIFIER.name],
                }
            else:
                inference_filepaths.append(filepath)
                batch_arr.append(img.arr / 255)
        if not batch_arr:
            return list(predictions.values())
        batch_arr = np.stack(batch_arr, axis=0)

        img_tensor = tf.convert_to_tensor(batch_arr)
        logits = self.model(img_tensor, training=False)
        scores = tf.keras.activations.softmax(logits)
        scores, indices = tf.math.top_k(scores, k=5)

        for filepath, scores_arr, indices_arr in zip(
            inference_filepaths, scores.numpy(), indices.numpy()
        ):

            predictions[filepath] = {
                "filepath": filepath,
                "classifications": {
                    "classes": [self.labels[idx] for idx in indices_arr],
                    "scores": scores_arr.tolist(),
                },
            }

        return [predictions[filepath] for filepath in filepaths]
