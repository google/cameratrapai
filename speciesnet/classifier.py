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
        mc_dropout_passes: int = 1,
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
        """

        start_time = time.time()

        self.model_info = ModelInfo(model_name)

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

        self.mc_dropout_passes = mc_dropout_passes

        # Set the model in inference mode.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable Dropout layers if using MC Dropout.
        if self.mc_dropout_passes > 1:
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    
            # Hack for GraphModules (exported ONNX models) that have stripped Dropout
            if hasattr(self.model, 'graph'):
                import torch.nn.functional as F
                matmul_node = None
                for node in self.model.graph.nodes:
                    if node.target == "SpeciesNet/dense/MatMul":
                        matmul_node = node
                        break
                
                if matmul_node:
                    with self.model.graph.inserting_before(matmul_node):
                        dropout_node = self.model.graph.call_function(
                            F.dropout, 
                            args=(matmul_node.args[0],), 
                            kwargs={'p': 0.2, 'training': True}
                        )
                        new_args = list(matmul_node.args)
                        new_args[0] = dropout_node
                        matmul_node.args = tuple(new_args)
                    
                    self.model.graph.lint()
                    self.model.recompile()

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
            A list of preprocessed images, or `None` if no PIL image was provided initially.
            If `bboxes` are provided and the model is `always_crop`, it returns one image per bbox.
        """

        if img is None:
            return None

        img_tensor = F.pil_to_tensor(img)  # HWC to CHW.
        img_tensor = F.convert_image_dtype(img_tensor, torch.float32)
        
        results = []

        if self.model_info.type_ == "always_crop":
            # Crop to all bboxes if available, otherwise leave image uncropped.
            if bboxes:
                for bbox in bboxes:
                    cropped = F.crop(
                        img_tensor,
                        int(bbox.ymin * img.height),
                        int(bbox.xmin * img.width),
                        int(bbox.height * img.height),
                        int(bbox.width * img.width),
                    )
                    results.append(cropped)
            else:
                results.append(img_tensor)
        elif self.model_info.type_ == "full_image":
            # Crop top and bottom of image.
            target_height = max(
                int(img.height * (1.0 - SpeciesNetClassifier.MAX_CROP_RATIO)),
                img.height - SpeciesNetClassifier.MAX_CROP_SIZE,
            )
            cropped = F.center_crop(img_tensor, [target_height, img.width])
            results.append(cropped)

        final_results = []
        for t in results:
            if resize:
                t = F.resize(
                    t,
                    [SpeciesNetClassifier.IMG_SIZE, SpeciesNetClassifier.IMG_SIZE],
                    antialias=False,
                )

            t = F.convert_image_dtype(t, torch.uint8)
            t = t.permute([1, 2, 0])  # CHW to HWC.
            final_results.append(PreprocessedImage(t.numpy(), img.width, img.height))
            
        return final_results

    def predict(
        self, filepath: str, imgs: Optional[list[PreprocessedImage]]
    ) -> dict[str, Any]:
        """Runs inference on a given list of preprocessed images for a single filepath.

        Args:
            filepath:
                Location of image to run inference on.
            imgs:
                List of preprocessed images to run inference on. If `None` or empty, a failure message is
                reported back.

        Returns:
            A dict containing either the top-5 classifications for each bounding box
            or a failure message.
        """
        return self.batch_predict([filepath], [imgs])[0]

    def batch_predict(
        self, filepaths: list[str], imgs_list: list[Optional[list[PreprocessedImage]]]
    ) -> list[dict[str, Any]]:
        """Runs inference on a batch of preprocessed images.

        Args:
            filepaths:
                List of image locations to run inference on. Used for reporting purposes
                only, and not for loading the images.
            imgs_list:
                List of preprocessed images lists to run inference on. Each list corresponds to
                multiple bboxes of the respective filepath.

        Returns:
            A list of dict results. Each dict result contains a list of top-5 classifications
        """

        predictions = {}

        inference_filepaths = []
        inference_bbox_indices = []
        batch_arr = []
        
        for filepath, imgs in zip(filepaths, imgs_list):
            if imgs is None or len(imgs) == 0:
                predictions[filepath] = {
                    "filepath": filepath,
                    "failures": [Failure.CLASSIFIER.name],
                }
            else:
                for idx, img in enumerate(imgs):
                    inference_filepaths.append(filepath)
                    inference_bbox_indices.append(idx)
                    batch_arr.append(img.arr / 255)
                    
        if not batch_arr:
            return [predictions.get(fp, {"filepath": fp, "failures": [Failure.CLASSIFIER.name]}) for fp in filepaths]
        batch_arr = np.stack(batch_arr, axis=0, dtype=np.float32)

        batch_tensor = torch.from_numpy(batch_arr).to(self.device)

        if getattr(self, "mc_dropout_passes", 1) > 1:
            all_scores = []
            all_logits = []
            for _ in range(self.mc_dropout_passes):
                pass_logits = self.model(batch_tensor).cpu()
                all_logits.append(pass_logits)
                all_scores.append(torch.softmax(pass_logits, dim=-1))
            
            # Average the probabilities (after softmax), as per instructions.
            logits = torch.stack(all_logits).mean(dim=0)
            scores = torch.stack(all_scores).mean(dim=0)
        else:
            logits = self.model(batch_tensor).cpu()
            scores = torch.softmax(logits, dim=-1)

        scores, indices = torch.topk(scores, k=5, dim=-1)

        for file_idx, (filepath, bbox_idx, scores_arr, indices_arr) in enumerate(
            zip(inference_filepaths, inference_bbox_indices, scores.numpy(), indices.numpy())
        ):
            if filepath not in predictions:
                predictions[filepath] = {
                    "filepath": filepath,
                    "classifications": []
                }
            
            classification_dict = {
                "classes": [self.labels[idx] for idx in indices_arr],
                "scores": scores_arr.tolist(),
            }

            if hasattr(self, "target_idx"):
                classification_dict.update(
                    {
                        "target_classes": self.target_labels,
                        "target_logits": [
                            float(logits[file_idx][idx]) for idx in self.target_idx
                        ],
                    }
                )
                
            predictions[filepath]["classifications"].append(classification_dict)

        return [predictions.get(fp, {"filepath": fp, "failures": [Failure.CLASSIFIER.name]}) for fp in filepaths]
