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

import json
from pathlib import Path
from typing import Generator

import pytest

from speciesnet.utils import file_exists
from speciesnet.utils import load_partial_predictions
from speciesnet.utils import load_rgb_image
from speciesnet.utils import ModelInfo
from speciesnet.utils import prepare_instances_dict
from speciesnet.utils import save_predictions

# fmt: off
# pylint: disable=line-too-long

AZ_VIA_HTTPS_TEST_DIR = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped"
AZ_VIA_HTTPS_TEST_IMG = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/59420eea-23d2-11e8-a6a3-ec086b02610b.jpg"
GS_TEST_DIR = "gs://public-datasets-lila/caltech-unzipped"
GS_TEST_IMG = "gs://public-datasets-lila/caltech-unzipped/cct_images/59420eea-23d2-11e8-a6a3-ec086b02610b.jpg"
S3_TEST_DIR = "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped"
S3_TEST_IMG = "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/caltech-unzipped/cct_images/59420eea-23d2-11e8-a6a3-ec086b02610b.jpg"

# pylint: enable=line-too-long
# fmt: on


class TestModelInfo:
    """Tests for the ModelInfo dataclass."""

    def test_model_type(self, model_name: str) -> None:
        model_info = ModelInfo(model_name)
        assert model_info.type_ in {"always_crop", "full_image"}


class TestPrepareInstancesDict:
    """Tests for the standardization of input formats."""

    @pytest.fixture
    def filepaths_as_strings(self) -> list[str]:
        return [
            "test_data/african_elephants.jpg",
            "test_data/african_elephants_bw.jpg",
            "test_data/african_elephants_cmyk.jpg",
            "test_data/african_elephants_truncated_file.jpg",
            "test_data/african_elephants_with_exif_orientation.jpg",
            "test_data/american_black_bear.jpg",
            "test_data/blank.jpg",
            "test_data/blank2.jpg",
            "test_data/blank3.jpg",
            "test_data/domestic_cattle.jpg",
            "test_data/domestic_dog.jpg",
            "test_data/human.jpg",
            "test_data/ocelot.jpg",
            "test_data/vehicle.jpg",
        ]

    @pytest.fixture
    def filepaths_as_paths(self, filepaths_as_strings) -> list[Path]:
        return [Path(f) for f in filepaths_as_strings]

    @pytest.fixture
    def folders_as_strings(self) -> list[str]:
        return [
            "test_data",
        ]

    @pytest.fixture
    def folders_as_paths(self, folders_as_strings) -> list[Path]:
        return [Path(f) for f in folders_as_strings]

    @pytest.fixture
    def instances_dict(self, filepaths_as_strings) -> dict:
        return {"instances": [{"filepath": f} for f in filepaths_as_strings]}

    @pytest.fixture
    def instances_dict_portugal(self, filepaths_as_strings) -> dict:
        return {
            "instances": [
                {
                    "filepath": f,
                    "country": "PRT",
                }
                for f in filepaths_as_strings
            ]
        }

    @pytest.fixture
    def instances_dict_portugal_porto(self, filepaths_as_strings) -> dict:
        return {
            "instances": [
                {
                    "filepath": f,
                    "country": "PRT",
                    "admin1_region": "13",
                }
                for f in filepaths_as_strings
            ]
        }

    def test_invalid_inputs(
        self, instances_dict, filepaths_as_strings, folders_as_paths
    ) -> None:
        with pytest.raises(ValueError):
            prepare_instances_dict()
        with pytest.raises(ValueError):
            prepare_instances_dict(
                instances_dict=instances_dict, filepaths=filepaths_as_strings
            )
        with pytest.raises(ValueError):
            prepare_instances_dict(
                instances_dict=instances_dict, folders_txt="test_data/local_folders.txt"
            )
        with pytest.raises(ValueError):
            prepare_instances_dict(
                instances_dict=instances_dict,
                filepaths_txt="test_data/local_filepaths.txt",
                folders=folders_as_paths,
            )

    def test_instances_conversion(self, instances_dict) -> None:
        assert prepare_instances_dict(instances_dict=instances_dict) == instances_dict
        assert (
            prepare_instances_dict(instances_json="test_data/local_instances.json")
            == instances_dict
        )
        assert (
            prepare_instances_dict(
                instances_json=Path("test_data/local_instances.json")
            )
            == instances_dict
        )

    def test_filepaths_conversion(
        self, instances_dict, filepaths_as_strings, filepaths_as_paths
    ) -> None:
        assert prepare_instances_dict(filepaths=filepaths_as_strings) == instances_dict
        assert prepare_instances_dict(filepaths=filepaths_as_paths) == instances_dict
        assert (
            prepare_instances_dict(filepaths_txt="test_data/local_filepaths.txt")
            == instances_dict
        )
        assert (
            prepare_instances_dict(filepaths_txt=Path("test_data/local_filepaths.txt"))
            == instances_dict
        )

    def test_folders_conversion(
        self, instances_dict, folders_as_strings, folders_as_paths
    ) -> None:
        assert prepare_instances_dict(folders=folders_as_strings) == instances_dict
        assert prepare_instances_dict(folders=folders_as_paths) == instances_dict
        assert (
            prepare_instances_dict(folders_txt="test_data/local_folders.txt")
            == instances_dict
        )
        assert (
            prepare_instances_dict(folders_txt=Path("test_data/local_folders.txt"))
            == instances_dict
        )

    def test_country_overwrites(self, instances_dict, instances_dict_portugal) -> None:
        assert (
            prepare_instances_dict(instances_dict=instances_dict, country="PRT")
            == instances_dict_portugal
        )

    def test_admin1_region_overwrites(
        self, instances_dict, instances_dict_portugal_porto
    ) -> None:
        assert (
            prepare_instances_dict(instances_dict=instances_dict, admin1_region="13")
            == instances_dict
        )
        assert (
            prepare_instances_dict(
                instances_dict=instances_dict, country="PRT", admin1_region="13"
            )
            == instances_dict_portugal_porto
        )


class TestFileExists:
    """Tests for the existence of files."""

    def test_local_file(self) -> None:
        assert file_exists("test_data/blank.jpg")
        assert not file_exists("test_data/missing.jpg")

    def test_http_file(self) -> None:
        assert file_exists("http://picsum.photos/200/300")
        assert not file_exists("http://picsum.photos/missing.jpg")

    def test_https_file(self) -> None:
        assert file_exists("https://picsum.photos/300/400")
        assert not file_exists("https://picsum.photos/missing.jpg")

    @pytest.mark.az
    def test_az_via_https_file(self) -> None:
        # We can't test the `az://` scheme directly because it's a pseudo-URL scheme,
        # not officially supported by Azure. However, we still support it via
        # `cloudpathlib` and `az://` URLs should work without issues when the user is
        # properly authenticated:
        # https://github.com/drivendataorg/cloudpathlib/issues/157
        assert file_exists(AZ_VIA_HTTPS_TEST_IMG)
        assert not file_exists(AZ_VIA_HTTPS_TEST_DIR + "/missing.jpg")

    @pytest.mark.gs
    def test_gs_file(self) -> None:
        assert file_exists(GS_TEST_IMG)
        assert not file_exists("gs://missing.jpg")
        assert not file_exists(GS_TEST_DIR + "/missing.jpg")

    @pytest.mark.s3
    def test_s3_file(self) -> None:
        assert file_exists(S3_TEST_IMG)
        assert not file_exists("s3://missing.jpg")
        assert not file_exists(S3_TEST_DIR + "/missing.jpg")


class TestLoadRGBImage:
    """Tests for the image loading."""

    def test_valid_image(self) -> None:
        img = load_rgb_image("test_data/african_elephants.jpg")
        assert img
        assert img.size == (2048, 1536)
        assert img.mode == "RGB"

    def test_missing_image(self) -> None:
        img = load_rgb_image("test_data/missing.jpg")
        assert img is None

    def test_truncated_image(self) -> None:
        img = load_rgb_image("test_data/african_elephants_truncated_file.jpg")
        assert img
        assert img.size == (2048, 1536)
        assert img.mode == "RGB"

    def test_exif_transpose(self) -> None:
        img = load_rgb_image("test_data/african_elephants_with_exif_orientation.jpg")
        assert img
        assert img.size == (2048, 1536)
        assert img.mode == "RGB"

    def test_cmyk_image(self) -> None:
        img = load_rgb_image("test_data/african_elephants_cmyk.jpg")
        assert img
        assert img.size == (2048, 1536)
        assert img.mode == "RGB"

    def test_bw_image(self) -> None:
        img = load_rgb_image("test_data/african_elephants_bw.jpg")
        assert img
        assert img.size == (2048, 1536)
        assert img.mode == "RGB"

    def test_http_image(self) -> None:
        img = load_rgb_image("http://picsum.photos/200/300")
        assert img
        assert img.size == (200, 300)
        assert img.mode == "RGB"

    def test_https_image(self) -> None:
        img = load_rgb_image("https://picsum.photos/300/400")
        assert img
        assert img.size == (300, 400)
        assert img.mode == "RGB"

    @pytest.mark.az
    def test_az_via_https_image(self) -> None:
        # We can't test the `az://` scheme directly because it's a pseudo-URL scheme,
        # not officially supported by Azure. However, we still support it via
        # `cloudpathlib` and `az://` URLs should work without issues when the user is
        # properly authenticated:
        # https://github.com/drivendataorg/cloudpathlib/issues/157
        img = load_rgb_image(AZ_VIA_HTTPS_TEST_IMG)
        assert img
        assert img.size == (2048, 1494)
        assert img.mode == "RGB"

    @pytest.mark.gs
    def test_gs_image(self) -> None:
        img = load_rgb_image(GS_TEST_IMG)
        assert img
        assert img.size == (2048, 1494)
        assert img.mode == "RGB"

    @pytest.mark.s3
    def test_s3_image(self) -> None:
        img = load_rgb_image(S3_TEST_IMG)
        assert img
        assert img.size == (2048, 1494)
        assert img.mode == "RGB"


class TestLoadPartialPredictions:
    """Tests for the loading of partial predictions."""

    def test_no_loading(self, tmp_path) -> None:
        valid_instances = [
            {"filepath": "a.jpg"},
            {"filepath": "b.jpg"},
            {"filepath": "c.jpg"},
        ]
        assert load_partial_predictions(None, valid_instances) == ({}, valid_instances)
        assert load_partial_predictions(tmp_path, valid_instances) == (
            {},
            valid_instances,
        )

    @pytest.fixture
    def valid_predictions_json(self, tmp_path) -> Generator[Path, None, None]:
        json_content = {
            "predictions": [
                {
                    "filepath": "a.jpg",
                    "failures": ["CLASSIFIER", "DETECTOR"],
                    "country": "COUNTRY_A",
                    "model_version": "0.0.0",
                },
                {
                    "filepath": "b.jpg",
                    "failures": ["CLASSIFIER"],
                    "country": "COUNTRY_B",
                    "detections": [
                        {
                            "category": "1",
                            "label": "animal",
                            "conf": 0.5,
                            "bbox": [0.0, 0.1, 0.2, 0.3],
                        }
                    ],
                    "model_version": "0.0.0",
                },
                {
                    "filepath": "c.jpg",
                    "failures": ["DETECTOR"],
                    "country": "COUNTRY_C",
                    "classifications": {
                        "classes": ["X", "Y", "Z"],
                        "scores": [0.5, 0.3, 0.2],
                    },
                    "model_version": "0.0.0",
                },
                {
                    "filepath": "d.jpg",
                    "failures": ["GEOLOCATION"],
                    "classifications": {
                        "classes": ["R", "S", "T"],
                        "scores": [0.7, 0.2, 0.1],
                    },
                    "detections": [
                        {
                            "category": "2",
                            "label": "human",
                            "conf": 0.7,
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                        }
                    ],
                    "prediction": "R",
                    "prediction_score": 0.7,
                    "prediction_source": "test",
                    "model_version": "0.0.0",
                },
                {
                    "filepath": "e.jpg",
                    "country": "COUNTRY_E",
                    "classifications": {
                        "classes": ["K", "L", "M"],
                        "scores": [0.9, 0.1, 0.0],
                    },
                    "detections": [
                        {
                            "category": "2",
                            "label": "human",
                            "conf": 0.7,
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                        }
                    ],
                    "prediction": "K",
                    "prediction_score": 0.9,
                    "prediction_source": "test",
                    "model_version": "0.0.0",
                },
                {
                    "filepath": "f.jpg",
                    "classifications": {
                        "classes": ["XYZ"],
                        "scores": [0.8],
                    },
                    "detections": [],
                    "prediction": "XYZ",
                    "prediction_score": 0.4,
                    "prediction_source": "test",
                    "model_version": "0.0.0",
                },
            ]
        }

        filepath = tmp_path / "valid_predictions.json"
        with open(filepath, mode="w", encoding="utf-8") as fp:
            json.dump(json_content, fp)
        yield filepath
        filepath.unlink()

    def test_successful_loading(self, valid_predictions_json) -> None:
        valid_instances = [
            {"filepath": "g.jpg"},
            {"filepath": "f.jpg"},
            {"filepath": "e.jpg"},
            {"filepath": "d.jpg"},
            {"filepath": "c.jpg"},
            {"filepath": "b.jpg"},
            {"filepath": "a.jpg"},
        ]
        valid_predictions, filtered_instances = load_partial_predictions(
            valid_predictions_json, valid_instances
        )
        assert valid_predictions == {
            "e.jpg": {
                "filepath": "e.jpg",
                "country": "COUNTRY_E",
                "classifications": {
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                },
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
                "prediction": "K",
                "prediction_score": 0.9,
                "prediction_source": "test",
                "model_version": "0.0.0",
            },
            "f.jpg": {
                "filepath": "f.jpg",
                "classifications": {
                    "classes": ["XYZ"],
                    "scores": [0.8],
                },
                "detections": [],
                "prediction": "XYZ",
                "prediction_score": 0.4,
                "prediction_source": "test",
                "model_version": "0.0.0",
            },
        }
        assert filtered_instances == [
            {"filepath": "g.jpg"},
            {"filepath": "d.jpg"},
            {"filepath": "c.jpg"},
            {"filepath": "b.jpg"},
            {"filepath": "a.jpg"},
        ]

        with pytest.raises(RuntimeError):
            load_partial_predictions(
                valid_predictions_json,
                [{"filepath": "a.jpg"}],
            )

    @pytest.fixture
    def invalid_predictions_json(self, tmp_path) -> Generator[Path, None, None]:
        filepath = tmp_path / "invalid_predictions.json"
        with open(filepath, mode="w", encoding="utf-8") as fp:
            fp.write(
                """{
                    "predictions": [
                        {
                            "filepath": "a.jpg",
                            "countries": {
                                "USA",
                                "MEX"
                            }
                        }
                    ]
                }"""
            )
        yield filepath
        filepath.unlink()

    def test_failed_loading(self, invalid_predictions_json) -> None:
        # Python sets are not JSON serializable.
        with pytest.raises(json.JSONDecodeError):
            load_partial_predictions(invalid_predictions_json, [{"filepath": "a.jpg"}])


class TestSavePredictions:
    """Tests for the saving of predictions."""

    def test_failed_saving(self, tmp_path) -> None:
        # Python sets are not JSON serializable.
        predictions = {
            "predictions": [
                {
                    "filepath": "a.jpg",
                    "countries": {"USA", "MEX"},
                }
            ]
        }
        with pytest.raises(TypeError):
            save_predictions(predictions, tmp_path)
