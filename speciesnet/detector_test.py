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

# pylint: disable=missing-module-docstring

import numpy as np
from numpy.testing import assert_array_equal
import PIL.Image
import pytest

from speciesnet.constants import Detection
from speciesnet.detector import SpeciesNetDetector
from speciesnet.utils import load_rgb_image
from speciesnet.utils import PreprocessedImage


class TestDetector:
    """Tests for the detector component."""

    @pytest.fixture(scope="class")
    def detector(self, model_name: str) -> SpeciesNetDetector:
        return SpeciesNetDetector(model_name)

    @pytest.fixture
    def img_green_w500_h700(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (500, 700), color=(0, 255, 0))

    @pytest.fixture
    def img_gray_green_gray_w23_w914_w23_h1280(self) -> PIL.Image.Image:
        img_gray = PIL.Image.new("RGB", (23, 1280), color=(114, 114, 114))
        img_green = PIL.Image.new("RGB", (914, 1280), color=(0, 255, 0))
        img = PIL.Image.new("RGB", (960, 1280))
        img.paste(img_gray, (0, 0))
        img.paste(img_green, (23, 0))
        img.paste(img_gray, (937, 0))
        return img

    @pytest.fixture
    def img_green_w2000_h1111(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (2000, 1111), color=(0, 255, 0))

    @pytest.fixture
    def img_gray_green_gray_w1280_h28_h711_h29(self) -> PIL.Image.Image:
        img_gray = PIL.Image.new("RGB", (1280, 29), color=(114, 114, 114))
        img_green = PIL.Image.new("RGB", (1280, 711), color=(0, 255, 0))
        img = PIL.Image.new("RGB", (1280, 768))
        img.paste(img_gray, (0, 0))
        img.paste(img_green, (0, 28))
        img.paste(img_gray, (0, 739))
        return img

    @pytest.fixture
    def img_green_w1280_h1280(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (1280, 1280), color=(0, 255, 0))

    def test_preprocess(  # pylint: disable=too-many-positional-arguments
        self,
        detector,
        img_green_w500_h700,  # input
        img_gray_green_gray_w23_w914_w23_h1280,  # output
        img_green_w2000_h1111,  # input
        img_gray_green_gray_w1280_h28_h711_h29,  # output
    ) -> None:

        assert detector.preprocess(None) is None

        preprocessed = detector.preprocess(img_green_w500_h700)
        assert preprocessed
        assert preprocessed.orig_width == 500
        assert preprocessed.orig_height == 700
        assert preprocessed.arr.shape == (1280, 960, 3)
        assert_array_equal(preprocessed.arr, img_gray_green_gray_w23_w914_w23_h1280)

        preprocessed = detector.preprocess(img_green_w2000_h1111)
        assert preprocessed
        assert preprocessed.orig_width == 2000
        assert preprocessed.orig_height == 1111
        assert preprocessed.arr.shape == (768, 1280, 3)
        assert_array_equal(preprocessed.arr, img_gray_green_gray_w1280_h28_h711_h29)

    def test_predict(self, detector, img_green_w1280_h1280) -> None:

        filepath = "missing.jpg"
        prediction = detector.predict(filepath, None)
        assert prediction["filepath"] == filepath
        assert "failures" in prediction

        filepath = "green.png"
        prediction = detector.predict(
            filepath, PreprocessedImage(np.asarray(img_green_w1280_h1280), 1280, 1280)
        )
        assert prediction["filepath"] == filepath
        assert "failures" not in prediction
        assert "detections" in prediction
        scores = [det["conf"] for det in prediction["detections"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.fixture(
        # Expected MDv5 detections.
        params=[
            ("test_data/blank.jpg", []),
            ("test_data/blank2.jpg", []),
            (
                "test_data/blank3.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0101,
                        "bbox": [0.0009191, 0, 0.996, 0.9571],
                    }
                ],
            ),
            (
                "test_data/african_elephants.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0105,
                        "bbox": [0.4863, 0.4466, 0.04394, 0.0996],
                    },
                    {
                        "category": "1",
                        "conf": 0.0126,
                        "bbox": [0.1416, 0.4082, 0.1049, 0.1217],
                    },
                    {
                        "category": "1",
                        "conf": 0.0135,
                        "bbox": [0.6386, 0.4231, 0.02685, 0.03385],
                    },
                    {
                        "category": "1",
                        "conf": 0.0175,
                        "bbox": [0.3198, 0.4127, 0.05126, 0.05598],
                    },
                    {
                        "category": "1",
                        "conf": 0.0419,
                        "bbox": [0.6406, 0.429, 0.05712, 0.05403],
                    },
                    {
                        "category": "1",
                        "conf": 0.0517,
                        "bbox": [0.145, 0.4088, 0.07177, 0.07617],
                    },
                    {
                        "category": "1",
                        "conf": 0.0605,
                        "bbox": [0.3769, 0.444, 0.04687, 0.07096],
                    },
                    {
                        "category": "1",
                        "conf": 0.0798,
                        "bbox": [0.3208, 0.414, 0.07226, 0.1158],
                    },
                    {
                        "category": "1",
                        "conf": 0.142,
                        "bbox": [0.102, 0.4055, 0.1118, 0.11],
                    },
                    {
                        "category": "1",
                        "conf": 0.636,
                        "bbox": [0.3403, 0.4433, 0.08349, 0.09114],
                    },
                    {
                        "category": "1",
                        "conf": 0.703,
                        "bbox": [0.854, 0.4785, 0.08593, 0.08268],
                    },
                    {
                        "category": "1",
                        "conf": 0.816,
                        "bbox": [0.9418, 0.4674, 0.0581, 0.1673],
                    },
                    {
                        "category": "1",
                        "conf": 0.903,
                        "bbox": [0.8291, 0.511, 0.1044, 0.1106],
                    },
                    {
                        "category": "1",
                        "conf": 0.912,
                        "bbox": [0.5644, 0.4218, 0.08496, 0.1673],
                    },
                    {
                        "category": "1",
                        "conf": 0.935,
                        "bbox": [0.7041, 0.4765, 0.1108, 0.125],
                    },
                ],
            ),
            (
                "test_data/african_elephants_bw.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0101,
                        "bbox": [0.8535, 0.4791, 0.03808, 0.03515],
                    },
                    {
                        "category": "1",
                        "conf": 0.0105,
                        "bbox": [0.3637, 0.4453, 0.08251, 0.1002],
                    },
                    {
                        "category": "1",
                        "conf": 0.0148,
                        "bbox": [0.4531, 0.4427, 0.07275, 0.1035],
                    },
                    {
                        "category": "1",
                        "conf": 0.0151,
                        "bbox": [0.4965, 0.4518, 0.03466, 0.08854],
                    },
                    {
                        "category": "1",
                        "conf": 0.0284,
                        "bbox": [0.03417, 0.3561, 0.1689, 0.1699],
                    },
                    {
                        "category": "1",
                        "conf": 0.0348,
                        "bbox": [0.3305, 0.4088, 0.05908, 0.1152],
                    },
                    {
                        "category": "1",
                        "conf": 0.0434,
                        "bbox": [0.499, 0.4505, 0.06787, 0.1087],
                    },
                    {
                        "category": "1",
                        "conf": 0.0841,
                        "bbox": [0.1123, 0.3541, 0.1, 0.1412],
                    },
                    {
                        "category": "1",
                        "conf": 0.139,
                        "bbox": [0.3666, 0.442, 0.05615, 0.06966],
                    },
                    {
                        "category": "1",
                        "conf": 0.484,
                        "bbox": [0.3378, 0.4316, 0.08447, 0.1035],
                    },
                    {
                        "category": "1",
                        "conf": 0.665,
                        "bbox": [0.8525, 0.4791, 0.08691, 0.09635],
                    },
                    {
                        "category": "1",
                        "conf": 0.818,
                        "bbox": [0.9414, 0.468, 0.05859, 0.1692],
                    },
                    {
                        "category": "1",
                        "conf": 0.874,
                        "bbox": [0.5644, 0.4173, 0.09423, 0.1705],
                    },
                    {
                        "category": "1",
                        "conf": 0.883,
                        "bbox": [0.8295, 0.511, 0.1044, 0.1113],
                    },
                    {
                        "category": "1",
                        "conf": 0.931,
                        "bbox": [0.7031, 0.4765, 0.1118, 0.1256],
                    },
                ],
            ),
            (
                "test_data/african_elephants_cmyk.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0101,
                        "bbox": [0.4858, 0.4466, 0.04443, 0.0996],
                    },
                    {
                        "category": "1",
                        "conf": 0.0104,
                        "bbox": [0.6386, 0.4231, 0.02783, 0.0345],
                    },
                    {
                        "category": "1",
                        "conf": 0.012,
                        "bbox": [0.142, 0.4082, 0.1035, 0.1217],
                    },
                    {
                        "category": "1",
                        "conf": 0.0221,
                        "bbox": [0.3198, 0.4127, 0.05126, 0.05598],
                    },
                    {
                        "category": "1",
                        "conf": 0.0415,
                        "bbox": [0.6411, 0.4296, 0.05664, 0.05338],
                    },
                    {
                        "category": "1",
                        "conf": 0.0544,
                        "bbox": [0.145, 0.4082, 0.07128, 0.07682],
                    },
                    {
                        "category": "1",
                        "conf": 0.0667,
                        "bbox": [0.3769, 0.444, 0.04687, 0.07161],
                    },
                    {
                        "category": "1",
                        "conf": 0.0841,
                        "bbox": [0.3208, 0.4147, 0.07177, 0.1152],
                    },
                    {
                        "category": "1",
                        "conf": 0.156,
                        "bbox": [0.101, 0.4055, 0.1123, 0.11],
                    },
                    {
                        "category": "1",
                        "conf": 0.637,
                        "bbox": [0.3403, 0.4433, 0.08349, 0.09114],
                    },
                    {
                        "category": "1",
                        "conf": 0.706,
                        "bbox": [0.854, 0.4785, 0.08593, 0.08268],
                    },
                    {
                        "category": "1",
                        "conf": 0.815,
                        "bbox": [0.9418, 0.4674, 0.0581, 0.1679],
                    },
                    {
                        "category": "1",
                        "conf": 0.903,
                        "bbox": [0.8295, 0.511, 0.104, 0.1106],
                    },
                    {
                        "category": "1",
                        "conf": 0.914,
                        "bbox": [0.5644, 0.4218, 0.08496, 0.1673],
                    },
                    {
                        "category": "1",
                        "conf": 0.934,
                        "bbox": [0.7041, 0.4765, 0.1108, 0.125],
                    },
                ],
            ),
            (
                "test_data/african_elephants_truncated_file.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0107,
                        "bbox": [0.4199, 0.4733, 0.01318, 0.02734],
                    },
                    {
                        "category": "1",
                        "conf": 0.0182,
                        "bbox": [0.3198, 0.4127, 0.05078, 0.05468],
                    },
                    {
                        "category": "1",
                        "conf": 0.0186,
                        "bbox": [0.6381, 0.4238, 0.0288, 0.0332],
                    },
                    {
                        "category": "1",
                        "conf": 0.0526,
                        "bbox": [0.6406, 0.429, 0.05712, 0.05273],
                    },
                    {
                        "category": "1",
                        "conf": 0.0565,
                        "bbox": [0.1494, 0.4088, 0.07031, 0.07552],
                    },
                    {
                        "category": "1",
                        "conf": 0.0597,
                        "bbox": [0.3764, 0.4433, 0.04492, 0.05533],
                    },
                    {
                        "category": "1",
                        "conf": 0.0759,
                        "bbox": [0.3247, 0.4212, 0.06542, 0.1067],
                    },
                    {
                        "category": "1",
                        "conf": 0.193,
                        "bbox": [0.1103, 0.4062, 0.1044, 0.1035],
                    },
                    {
                        "category": "1",
                        "conf": 0.602,
                        "bbox": [0.3403, 0.444, 0.08398, 0.08789],
                    },
                    {
                        "category": "1",
                        "conf": 0.691,
                        "bbox": [0.854, 0.4785, 0.08593, 0.08333],
                    },
                    {
                        "category": "1",
                        "conf": 0.822,
                        "bbox": [0.9414, 0.4674, 0.05859, 0.1679],
                    },
                    {
                        "category": "1",
                        "conf": 0.9,
                        "bbox": [0.83, 0.5117, 0.1035, 0.11],
                    },
                    {
                        "category": "1",
                        "conf": 0.902,
                        "bbox": [0.5644, 0.4218, 0.08496, 0.1666],
                    },
                    {
                        "category": "1",
                        "conf": 0.931,
                        "bbox": [0.7045, 0.4759, 0.1103, 0.1256],
                    },
                ],
            ),
            (
                "test_data/african_elephants_with_exif_orientation.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0127,
                        "bbox": [0.1435, 0.4075, 0.1147, 0.1256],
                    },
                    {
                        "category": "1",
                        "conf": 0.0147,
                        "bbox": [0.6386, 0.4231, 0.02685, 0.03385],
                    },
                    {
                        "category": "1",
                        "conf": 0.02,
                        "bbox": [0.3198, 0.4127, 0.05126, 0.05533],
                    },
                    {
                        "category": "1",
                        "conf": 0.0514,
                        "bbox": [0.6411, 0.4296, 0.05664, 0.05273],
                    },
                    {
                        "category": "1",
                        "conf": 0.0556,
                        "bbox": [0.3769, 0.444, 0.04687, 0.07096],
                    },
                    {
                        "category": "1",
                        "conf": 0.0558,
                        "bbox": [0.145, 0.4082, 0.07226, 0.07812],
                    },
                    {
                        "category": "1",
                        "conf": 0.0714,
                        "bbox": [0.3208, 0.4147, 0.07275, 0.1152],
                    },
                    {
                        "category": "1",
                        "conf": 0.146,
                        "bbox": [0.103, 0.4062, 0.1113, 0.108],
                    },
                    {
                        "category": "1",
                        "conf": 0.65,
                        "bbox": [0.3398, 0.4433, 0.08398, 0.09179],
                    },
                    {
                        "category": "1",
                        "conf": 0.698,
                        "bbox": [0.854, 0.4785, 0.08593, 0.08268],
                    },
                    {
                        "category": "1",
                        "conf": 0.815,
                        "bbox": [0.9418, 0.4674, 0.0581, 0.1673],
                    },
                    {
                        "category": "1",
                        "conf": 0.903,
                        "bbox": [0.8295, 0.511, 0.104, 0.1106],
                    },
                    {
                        "category": "1",
                        "conf": 0.912,
                        "bbox": [0.5644, 0.4218, 0.08496, 0.1673],
                    },
                    {
                        "category": "1",
                        "conf": 0.934,
                        "bbox": [0.7041, 0.4765, 0.1108, 0.125],
                    },
                ],
            ),
            (
                "test_data/american_black_bear.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.959,
                        "bbox": [0.4331, 0.4283, 0.3232, 0.3222],
                    }
                ],
            ),
            (
                "test_data/domestic_cattle.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.942,
                        "bbox": [0.8886, 0.483, 0.1113, 0.1093],
                    },
                    {
                        "category": "1",
                        "conf": 0.952,
                        "bbox": [0.5791, 0.3723, 0.07617, 0.1295],
                    },
                    {
                        "category": "1",
                        "conf": 0.954,
                        "bbox": [0.3803, 0.4348, 0.1176, 0.1575],
                    },
                ],
            ),
            (
                "test_data/domestic_dog.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0137,
                        "bbox": [0.2402, 0.3092, 0.2548, 0.2109],
                    },
                    {
                        "category": "1",
                        "conf": 0.929,
                        "bbox": [0.2377, 0.08398, 0.5161, 0.6497],
                    },
                ],
            ),
            (
                "test_data/human.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.0118,
                        "bbox": [0.07877, 0.64, 0.107, 0.173],
                    },
                    {
                        "category": "1",
                        "conf": 0.0738,
                        "bbox": [0.4104, 0.6707, 0.09472, 0.1238],
                    },
                    {
                        "category": "2",
                        "conf": 0.933,
                        "bbox": [0.7115, 0.4976, 0.0664, 0.2424],
                    },
                ],
            ),
            (
                "test_data/ocelot.jpg",
                [
                    {
                        "category": "1",
                        "conf": 0.971,
                        "bbox": [0.0478, 0.5548, 0.521, 0.2743],
                    }
                ],
            ),
            (
                "test_data/vehicle.jpg",
                [
                    {
                        "category": "3",
                        "conf": 0.977,
                        "bbox": [0, 0.02083, 1.0, 0.6816],
                    }
                ],
            ),
        ]
    )
    def predicted_vs_expected(self, detector, request) -> tuple[list, list]:
        filepath, label = request.param
        img = detector.preprocess(load_rgb_image(filepath))
        assert img is not None
        return detector.predict(filepath, img)["detections"], label

    def test_detections(self, predicted_vs_expected) -> None:
        predicted, expected = predicted_vs_expected
        assert predicted == sorted(predicted, key=lambda det: det["conf"], reverse=True)
        expected = sorted(expected, key=lambda det: det["conf"], reverse=True)
        for pred_det, exp_det in zip(predicted, expected):
            assert pred_det["category"] == exp_det["category"]
            assert pred_det["label"] == Detection.from_category(exp_det["category"])
            assert pred_det["conf"] == pytest.approx(pred_det["conf"], abs=1.5e-3)
            assert pred_det["bbox"] == pytest.approx(exp_det["bbox"], abs=1.5e-3)
