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

"""Detector functionality of SpeciesNet."""

__all__ = [
    "SpeciesNetDetector",
]

import time
from typing import Any, Optional

from absl import logging
from humanfriendly import format_timespan
import numpy as np
import PIL.Image
import torch
import torch.backends
import torch.backends.mps
from yolov5.utils.augmentations import letterbox as yolov5_letterbox
from yolov5.utils.general import non_max_suppression as yolov5_non_max_suppression
from yolov5.utils.general import xyxy2xywhn as yolov5_xyxy2xywhn

try:
    from yolov5.utils.general import scale_boxes as yolov5_scale_boxes
except ImportError:
    from yolov5.utils.general import scale_coords as yolov5_scale_boxes

from speciesnet.constants import Detection
from speciesnet.constants import Failure
from speciesnet.utils import ModelInfo
from speciesnet.utils import PreprocessedImage


class SpeciesNetDetector:
    """Detector component of SpeciesNet."""

    IMG_SIZE = 1280
    STRIDE = 64
    DETECTION_THRESHOLD = 0.01

    def __init__(self, model_name: str) -> None:
        """Loads the detector resources.

        Code adapted from: https://github.com/agentmorris/MegaDetector
        which was released under the MIT License:
        https://github.com/agentmorris/MegaDetector/blob/main/LICENSE

        Args:
            model_name:
                String value identifying the model to be loaded. It can be a Kaggle
                identifier (starting with `kaggle:`), a HuggingFace identifier (starting
                with `hf:`) or a local folder to load the model from.
        """

        start_time = time.time()

        self.model_info = ModelInfo(model_name)

        # Select the best device available.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load the model.
        if self.device != "mps":
            checkpoint = torch.load(
                self.model_info.detector, map_location=self.device, weights_only=False
            )
            self.model = checkpoint["model"].float()
        else:
            checkpoint = torch.load(self.model_info.detector, weights_only=False)
            self.model = checkpoint["model"].float().to(self.device)

        # Set the model in inference mode.
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Fix compatibility issues to be able to load older YOLOv5 models with newer
        # versions of PyTorch.
        for m in self.model.modules():
            if isinstance(m, torch.nn.Upsample) and not hasattr(
                m, "recompute_scale_factor"
            ):
                m.recompute_scale_factor = None

        end_time = time.time()
        logging.info(
            "Loaded SpeciesNetDetector in %s on %s.",
            format_timespan(end_time - start_time),
            self.device.upper(),
        )

    def preprocess(self, img: Optional[PIL.Image.Image]) -> Optional[PreprocessedImage]:
        """Preprocesses an image according to this detector's needs.

        Args:
            img:
                PIL image to preprocess. If `None`, no preprocessing is performed.

        Returns:
            A preprocessed image, or `None` if no PIL image was provided initially.
        """

        if img is None:
            return None

        img_arr = yolov5_letterbox(
            np.asarray(img),
            new_shape=SpeciesNetDetector.IMG_SIZE,
            stride=SpeciesNetDetector.STRIDE,
            auto=True,
        )[0]
        return PreprocessedImage(img_arr, img.width, img.height)

    def _convert_yolo_xywhn_to_md_xywhn(self, yolo_xywhn: list[float]) -> list[float]:
        """Converts bbox XYWHN coordinates from YOLO's to MegaDetector's format.

        Args:
            yolo_xywhn:
                List of bbox coordinates in YOLO format, i.e.
                [x_center, y_center, width, height].

        Returns:
            List of bbox coordinates in MegaDetector format, i.e.
            [x_min, y_min, width, height].
        """

        x_center, y_center, width, height = yolo_xywhn
        x_min = x_center - width / 2.0
        y_min = y_center - height / 2.0
        return [x_min, y_min, width, height]

    def predict(
        self, filepath: str, img: Optional[PreprocessedImage]
    ) -> dict[str, Any]:
        """Runs inference on a given preprocessed image.

        Code adapted from: https://github.com/agentmorris/MegaDetector
        which was released under the MIT License:
        https://github.com/agentmorris/MegaDetector/blob/main/LICENSE

        Args:
            filepath:
                Location of image to run inference on. Used for reporting purposes only,
                and not for loading the image.
            img:
                Preprocessed image to run inference on. If `None`, a failure message is
                reported back.

        Returns:
            A dict containing either the detections above a fixed confidence threshold
            for the given image (in decreasing order of confidence scores), or a failure
            message if no preprocessed image was provided.
        """

        if img is None:
            return {
                "filepath": filepath,
                "failures": [Failure.DETECTOR.name],
            }

        # Prepare model input.
        img_tensor = torch.from_numpy(img.arr / 255)
        img_tensor = img_tensor.permute([2, 0, 1])  # HWC to CHW.
        batch_tensor = torch.unsqueeze(img_tensor, 0).float()  # CHW to NCHW.
        batch_tensor = batch_tensor.to(self.device)

        # Run inference.
        results = self.model(batch_tensor, augment=False)[0]
        if self.device == "mps":
            results = results.cpu()
        results = yolov5_non_max_suppression(
            prediction=results,
            conf_thres=SpeciesNetDetector.DETECTION_THRESHOLD,
        )
        results = results[0]  # Drop batch dimension.

        # Process detections.
        detections = []
        results[:, :4] = yolov5_scale_boxes(
            batch_tensor.shape[2:],
            results[:, :4],
            (img.orig_height, img.orig_width),
        ).round()
        for result in results:  # (x_min, y_min, x_max, y_max, conf, category)
            xyxy = result[:4]

            # We want to support multiple versions of xyxy2xywhn, some of which
            # are agnostic to input dimensionality (so can support 1D or 2D arrays),
            # some of which require 2D input.  To support both cases, we pass xyxy as
            # a 2D array, and convert back to 1D if necessary.
            ndims = xyxy.ndim
            if ndims == 1:
                xyxy = xyxy[None, :]
            xywhn = yolov5_xyxy2xywhn(xyxy, w=img.orig_width, h=img.orig_height)
            if ndims == 1:
                xywhn = xywhn[0]

            bbox = self._convert_yolo_xywhn_to_md_xywhn(xywhn.tolist())

            conf = result[4].item()

            category = str(int(result[5].item()) + 1)
            label = Detection.from_category(category)
            if label is None:
                logging.error("Invalid detection class: %s", category)
                continue

            detections.append(
                {
                    "category": category,
                    "label": label.value,
                    "conf": conf,
                    "bbox": bbox,
                }
            )

        # Sort detections by confidence score.
        detections = sorted(detections, key=lambda det: det["conf"], reverse=True)

        return {
            "filepath": filepath,
            "detections": detections,
        }
