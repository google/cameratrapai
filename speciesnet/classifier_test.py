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

from speciesnet.classifier import SpeciesNetClassifier
from speciesnet.utils import BBox
from speciesnet.utils import load_rgb_image
from speciesnet.utils import PreprocessedImage

# fmt: off
# pylint: disable=line-too-long

BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"

AFRICAN_ELEPHANT = "55631055-3e0e-4b7a-9612-dedebe9f78b0;mammalia;proboscidea;elephantidae;loxodonta;africana;african elephant"
AMERICAN_BLACK_BEAR = "436ddfdd-bc43-44c3-a25d-34671d3430a0;mammalia;carnivora;ursidae;ursus;americanus;american black bear"
DOMESTIC_CATTLE = "aca65aaa-8c6d-4b69-94de-842b08b13bd6;mammalia;artiodactyla;bovidae;bos;taurus;domestic cattle"
DOMESTIC_DOG = "3d80f1d6-b1df-4966-9ff4-94053c7a902a;mammalia;carnivora;canidae;canis;familiaris;domestic dog"
OCELOT = "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot"

# pylint: enable=line-too-long
# fmt: on


class TestClassifier:
    """Tests for the classifier component."""

    @pytest.fixture(scope="class")
    def classifier(self, model_name: str) -> SpeciesNetClassifier:
        return SpeciesNetClassifier(model_name)

    @pytest.fixture
    def img_green_w40_h100(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (40, 100), color=(0, 255, 0))

    @pytest.fixture
    def img_green_w100_h700(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (100, 700), color=(0, 255, 0))

    @pytest.fixture
    def img_red_green_blue_w100_h150_h700_h150(self) -> PIL.Image.Image:
        img_red = PIL.Image.new("RGB", (100, 150), color=(255, 0, 0))
        img_green = PIL.Image.new("RGB", (100, 700), color=(0, 255, 0))
        img_blue = PIL.Image.new("RGB", (100, 150), color=(0, 0, 255))
        img = PIL.Image.new("RGB", (100, 1000))
        img.paste(img_red, (0, 0))
        img.paste(img_green, (0, 150))
        img.paste(img_blue, (0, 850))
        return img

    @pytest.fixture
    def img_green_w100_h2000(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (100, 2000), color=(0, 255, 0))

    @pytest.fixture
    def img_red_green_blue_w100_h200_h2000_h200(self) -> PIL.Image.Image:
        img_red = PIL.Image.new("RGB", (100, 200), color=(255, 0, 0))
        img_green = PIL.Image.new("RGB", (100, 2000), color=(0, 255, 0))
        img_blue = PIL.Image.new("RGB", (100, 200), color=(0, 0, 255))
        img = PIL.Image.new("RGB", (100, 2400))
        img.paste(img_red, (0, 0))
        img.paste(img_green, (0, 200))
        img.paste(img_blue, (0, 2200))
        return img

    @pytest.fixture
    def img_green_w480_h480(self) -> PIL.Image.Image:
        return PIL.Image.new("RGB", (480, 480), color=(0, 255, 0))

    def test_preprocess(  # pylint: disable=too-many-positional-arguments
        self,
        classifier,
        img_red_green_blue_w100_h150_h700_h150,  # input
        img_green_w40_h100,  # output
        img_green_w100_h700,  # output
        img_red_green_blue_w100_h200_h2000_h200,  # input
        img_green_w100_h2000,  # output
        img_green_w480_h480,  # output
    ) -> None:

        assert classifier.preprocess(None) is None

        if classifier.model_info.type_ == "always_crop":

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (1000, 100, 3)
            assert_array_equal(preprocessed.arr, img_red_green_blue_w100_h150_h700_h150)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                bboxes=[],
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (1000, 100, 3)
            assert_array_equal(preprocessed.arr, img_red_green_blue_w100_h150_h700_h150)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                bboxes=[BBox(0 / 100, 150 / 1000, 100 / 100, 700 / 1000)],
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (700, 100, 3)
            assert_array_equal(preprocessed.arr, img_green_w100_h700)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                bboxes=[BBox(0 / 100, 150 / 1000, 100 / 100, 700 / 1000)],
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (480, 480, 3)
            assert_array_equal(preprocessed.arr, img_green_w480_h480)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                bboxes=[BBox(10 / 100, 150 / 1000, 40 / 100, 100 / 1000)],
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (100, 40, 3)
            assert_array_equal(preprocessed.arr, img_green_w40_h100)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                bboxes=[BBox(10 / 100, 150 / 1000, 40 / 100, 100 / 1000)],
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (480, 480, 3)
            assert_array_equal(preprocessed.arr, img_green_w480_h480)

        elif classifier.model_info.type_ == "full_image":

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                bboxes=[BBox(10 / 100, 150 / 1000, 40 / 100, 100 / 1000)],
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (700, 100, 3)
            assert_array_equal(preprocessed.arr, img_green_w100_h700)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (700, 100, 3)
            assert_array_equal(preprocessed.arr, img_green_w100_h700)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h150_h700_h150,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 1000
            assert preprocessed.arr.shape == (480, 480, 3)
            assert_array_equal(preprocessed.arr, img_green_w480_h480)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h200_h2000_h200,
                resize=False,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 2400
            assert preprocessed.arr.shape == (2000, 100, 3)
            assert_array_equal(preprocessed.arr, img_green_w100_h2000)

            preprocessed = classifier.preprocess(
                img_red_green_blue_w100_h200_h2000_h200,
            )
            assert preprocessed
            assert preprocessed.orig_width == 100
            assert preprocessed.orig_height == 2400
            assert preprocessed.arr.shape == (480, 480, 3)
            assert_array_equal(preprocessed.arr, img_green_w480_h480)

    def test_predict(self, classifier, img_green_w480_h480) -> None:

        filepath = "missing.jpg"
        prediction = classifier.predict(filepath, None)
        assert prediction["filepath"] == filepath
        assert "failures" in prediction

        filepath = "green.png"
        prediction = classifier.predict(
            filepath, PreprocessedImage(np.asarray(img_green_w480_h480), 480, 480)
        )
        assert prediction["filepath"] == filepath
        assert "failures" not in prediction
        assert "classifications" in prediction
        assert "classes" in prediction["classifications"]
        assert "scores" in prediction["classifications"]
        assert prediction["classifications"]["scores"] == sorted(
            prediction["classifications"]["scores"], reverse=True
        )

    @pytest.fixture(
        # Expected classifications.
        params=[
            (
                "test_data/blank.jpg",
                [],
                BLANK,
            ),
            (
                "test_data/blank2.jpg",
                [],
                BLANK,
            ),
            (
                "test_data/blank3.jpg",
                [BBox(0.0009191, 0, 0.996, 0.9571)],
                BLANK,
            ),
            (
                "test_data/african_elephants.jpg",
                [BBox(0.7041, 0.4765, 0.1108, 0.125)],
                AFRICAN_ELEPHANT,
            ),
            (
                "test_data/african_elephants_bw.jpg",
                [BBox(0.7031, 0.4765, 0.1118, 0.1256)],
                AFRICAN_ELEPHANT,
            ),
            (
                "test_data/african_elephants_cmyk.jpg",
                [BBox(0.7041, 0.4765, 0.1108, 0.125)],
                AFRICAN_ELEPHANT,
            ),
            (
                "test_data/african_elephants_truncated_file.jpg",
                [BBox(0.7045, 0.4759, 0.1103, 0.1256)],
                AFRICAN_ELEPHANT,
            ),
            (
                "test_data/african_elephants_with_exif_orientation.jpg",
                [BBox(0.7041, 0.4765, 0.1108, 0.125)],
                AFRICAN_ELEPHANT,
            ),
            (
                "test_data/american_black_bear.jpg",
                [BBox(0.4331, 0.4283, 0.3232, 0.3222)],
                AMERICAN_BLACK_BEAR,
            ),
            (
                "test_data/domestic_cattle.jpg",
                [BBox(0.3803, 0.4348, 0.1176, 0.1575)],
                DOMESTIC_CATTLE,
            ),
            (
                "test_data/domestic_dog.jpg",
                [BBox(0.2377, 0.08398, 0.5161, 0.6497)],
                DOMESTIC_DOG,
            ),
            (
                "test_data/human.jpg",
                [BBox(0.7115, 0.4976, 0.0664, 0.2424)],
                HUMAN,
            ),
            (
                "test_data/ocelot.jpg",
                [BBox(0.0478, 0.5548, 0.521, 0.2743)],
                OCELOT,
            ),
            (
                "test_data/vehicle.jpg",
                [BBox(0, 0.02083, 1.0, 0.6816)],
                VEHICLE,
            ),
        ]
    )
    def predicted_vs_expected(self, classifier, request) -> tuple[dict, str]:
        filepath, bboxes, label = request.param
        img = classifier.preprocess(load_rgb_image(filepath), bboxes=bboxes)
        assert img is not None
        return classifier.predict(filepath, img)["classifications"], label

    def test_classifications(self, predicted_vs_expected) -> None:
        classifications, label = predicted_vs_expected
        assert classifications["classes"][0] == label
        assert classifications["scores"] == sorted(
            classifications["scores"], reverse=True
        )

    def test_target_species_batched_vs_non_batched(
        self, model_name: str, tmp_path
    ) -> None:
        """Test that target_species_txt works consistently
        with batch and non-batch predict."""

        # Create a temporary target species file with a subset of species
        target_species_file = tmp_path / "target_species.txt"
        target_species = [
            AFRICAN_ELEPHANT,
            DOMESTIC_DOG,
            HUMAN,
            BLANK,
        ]
        target_species_file.write_text("\n".join(target_species) + "\n")

        # Create a classifier with target_species_txt
        classifier_with_targets = SpeciesNetClassifier(
            model_name,
            target_species_txt=str(target_species_file),
        )

        # Test images with various species
        test_cases = [
            ("test_data/african_elephants.jpg", [BBox(0.7041, 0.4765, 0.1108, 0.125)]),
            ("test_data/domestic_dog.jpg", [BBox(0.2377, 0.08398, 0.5161, 0.6497)]),
            ("test_data/human.jpg", [BBox(0.7115, 0.4976, 0.0664, 0.2424)]),
            ("test_data/blank.jpg", []),
        ]

        # Preprocess all images
        filepaths = []
        preprocessed_imgs = []
        for filepath, bboxes in test_cases:
            img = classifier_with_targets.preprocess(
                load_rgb_image(filepath), bboxes=bboxes
            )
            filepaths.append(filepath)
            preprocessed_imgs.append(img)

        # Test 1: Non-batched prediction (batch_size=1)
        non_batched_predictions = []
        for filepath, img in zip(filepaths, preprocessed_imgs):
            prediction = classifier_with_targets.predict(filepath, img)
            non_batched_predictions.append(prediction)

        # Test 2: Batched prediction (batch_size>1)
        batched_predictions = classifier_with_targets.batch_predict(
            filepaths, preprocessed_imgs
        )

        # Verify that both approaches produce identical results
        assert len(non_batched_predictions) == len(batched_predictions)

        for i, (non_batched, batched) in enumerate(
            zip(non_batched_predictions, batched_predictions)
        ):
            # Check that both have target_logits
            assert "target_logits" in non_batched["classifications"]
            assert "target_logits" in batched["classifications"]

            # Check that target_classes are present and identical
            assert "target_classes" in non_batched["classifications"]
            assert "target_classes" in batched["classifications"]
            assert (
                non_batched["classifications"]["target_classes"]
                == batched["classifications"]["target_classes"]
            )

            # Check that target_logits are identical
            non_batched_logits = non_batched["classifications"]["target_logits"]
            batched_logits = batched["classifications"]["target_logits"]

            assert len(non_batched_logits) == len(batched_logits)
            assert len(non_batched_logits) == len(target_species)

            # Use np.allclose for floating point comparison
            # Note: Using relaxed tolerances to account for minor numerical differences
            # in batched vs non-batched processing (e.g., from fp32 operations).
            # If this test fails with larger differences, it indicates a bug where
            # batched and non-batched predictions produce different results.
            np.testing.assert_allclose(
                non_batched_logits,
                batched_logits,
                rtol=1e-3,  # 0.1% relative tolerance
                atol=1e-3,  # 0.001 absolute tolerance
                err_msg=f"Target logits mismatch for image {i} ({filepaths[i]})",
            )

            # Also verify that regular classifications match
            assert (
                non_batched["classifications"]["classes"]
                == batched["classifications"]["classes"]
            )
            np.testing.assert_allclose(
                non_batched["classifications"]["scores"],
                batched["classifications"]["scores"],
                rtol=1e-3,  # 0.1% relative tolerance
                atol=1e-5,  # 0.00001 absolute tolerance
                err_msg=f"Scores mismatch for image {i} ({filepaths[i]})",
            )
