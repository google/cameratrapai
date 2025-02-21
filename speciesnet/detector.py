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
import PIL.ExifTags
import PIL.Image
import torch
import torch.backends
import torch.backends.mps
from yolov5.utils.augmentations import letterbox as yolov5_letterbox
from yolov5.utils.general import non_max_suppression as yolov5_non_max_suppression
from yolov5.utils.general import scale_boxes as yolov5_scale_boxes
from yolov5.utils.general import xyxy2xywhn as yolov5_xyxy2xywhn

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
        self.model.eval()

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

    def preprocess(
        self,
        img: Optional[PIL.Image.Image],
        pad_to_square: bool = True,  # FIXME: True or False as default?
    ) -> Optional[PreprocessedImage]:
        """Preprocesses an image according to this detector's needs.

        Args:
            img:
                PIL image to preprocess. If `None`, no preprocessing is performed.
            pad_to_square:
                FIXME: add description

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
            color=(114, 114, 114),
        )[0]
        if pad_to_square:
            pad_width = (
                ((SpeciesNetDetector.IMG_SIZE - img_arr.shape[0]) // 2,) * 2,
                ((SpeciesNetDetector.IMG_SIZE - img_arr.shape[1]) // 2,) * 2,
                (0, 0),
            )
            img_arr = np.pad(img_arr, pad_width, "constant", constant_values=114)
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

    # FIXME: drop this
    def predict(
        self,
        filepath: str,
        img: Optional[PreprocessedImage],
        scale_bboxes: bool = True,  # FIXME: True or False as default?
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
            scale_bboxes:
                FIXME: add description

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
        img_arr = img.arr.transpose((2, 0, 1))  # HWC to CHW.
        img_arr = np.ascontiguousarray(img_arr)
        img_tensor = torch.from_numpy(img_arr).to(self.device)
        img_tensor = img_tensor.float() / 255
        img_tensor = torch.unsqueeze(img_tensor, 0)  # Add batch dimension.

        # Run inference.
        results = self.model(img_tensor, augment=False)[0]
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
            img_tensor.shape[2:],  # Skip batch + channels dimensions.
            results[:, :4],
            (
                (img.orig_height, img.orig_width)
                if scale_bboxes
                else (img.arr.shape[0], img.arr.shape[1])
            ),
        ).round()
        for result in results:  # (x_min, y_min, x_max, y_max, conf, category)
            xyxy = result[:4]
            xywhn = yolov5_xyxy2xywhn(
                xyxy,
                w=img.orig_width if scale_bboxes else img.arr.shape[1],
                h=img.orig_height if scale_bboxes else img.arr.shape[0],
            )
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

    def batch_predict(
        self,
        filepaths: list[str],
        imgs: list[Optional[PreprocessedImage]],
        scale_bboxes: bool = True,  # FIXME: True or False as default?
    ) -> list[dict[str, Any]]:
        """Runs inference on a batch of preprocessed images.

        Code adapted from: https://github.com/agentmorris/MegaDetector
        which was released under the MIT License:
        https://github.com/agentmorris/MegaDetector/blob/main/LICENSE

        FIXME
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

        batch_arr = []
        for img in imgs:
            if img is None:
                batch_arr.append(
                    np.zeros(
                        [
                            3,
                            SpeciesNetDetector.IMG_SIZE,
                            SpeciesNetDetector.IMG_SIZE,
                        ]
                    )
                )
            else:
                img_arr = img.arr.transpose((2, 0, 1))  # HWC to CHW.
                img_arr = np.ascontiguousarray(img_arr)
                batch_arr.append(img_arr / 255)
        batch_arr = np.stack(batch_arr, axis=0)
        batch_tensor = torch.from_numpy(batch_arr).float().to(self.device)

        # Run inference.
        batch_results = self.model(batch_tensor, augment=False)[0]
        if self.device == "mps":
            batch_results = batch_results.cpu()
        batch_results = yolov5_non_max_suppression(
            prediction=batch_results,
            conf_thres=SpeciesNetDetector.DETECTION_THRESHOLD,
        )

        # Process detections.
        predictions = []
        for filepath, img, img_tensor, results in zip(
            filepaths, imgs, batch_tensor, batch_results
        ):
            if img is None:
                predictions.append(
                    {
                        "filepath": filepath,
                        "failures": [Failure.DETECTOR.name],
                    }
                )
                continue

            detections = []
            results[:, :4] = yolov5_scale_boxes(
                img_tensor.shape[1:],  # Skip channels dimensions.
                results[:, :4],
                (
                    (img.orig_height, img.orig_width)
                    if scale_bboxes
                    else (img.arr.shape[0], img.arr.shape[1])
                ),
            ).round()

            for result in results:  # (x_min, y_min, x_max, y_max, conf, category)
                xyxy = result[:4]
                xywhn = yolov5_xyxy2xywhn(
                    xyxy,
                    w=img.orig_width if scale_bboxes else img.arr.shape[1],
                    h=img.orig_height if scale_bboxes else img.arr.shape[0],
                )
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

            predictions.append(
                {
                    "filepath": filepath,
                    "detections": detections,
                }
            )

        return predictions
