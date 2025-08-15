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

import numpy as np
import pytest

from speciesnet.utils import file_exists
from speciesnet.utils import limit_float_precision
from speciesnet.utils import load_json
from speciesnet.utils import load_partial_predictions
from speciesnet.utils import load_rgb_image
from speciesnet.utils import ModelInfo
from speciesnet.utils import prepare_instances_dict
from speciesnet.utils import save_predictions
from speciesnet.utils import write_json

# fmt: off
# pylint: disable=line-too-long

AZ_VIA_HTTPS_TEST_DIR = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped"
AZ_VIA_HTTPS_TEST_IMG = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/59420eea-23d2-11e8-a6a3-ec086b02610b.jpg"
HTTPS_TEST_IMG = AZ_VIA_HTTPS_TEST_IMG

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

    def test_https_file(self) -> None:
        assert file_exists(HTTPS_TEST_IMG)
        assert not file_exists(HTTPS_TEST_IMG + "-invalid")

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

    def test_https_image(self) -> None:
        img = load_rgb_image(HTTPS_TEST_IMG)
        assert img
        assert img.size == (2048, 1494)
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


class TestPrecisionLimiting:
    """Tests for precision limiting functionality in JSON operations."""

    def test_limit_float_precision_simple_float(self) -> None:
        """Test precision limiting for simple floats."""
        assert limit_float_precision(3.14159265359, 2) == 3.14
        assert limit_float_precision(3.14159265359, 4) == 3.1416
        assert limit_float_precision(2.0, 2) == 2.0

    def test_limit_float_precision_numpy_float(self) -> None:
        """Test precision limiting for numpy floats."""
        assert limit_float_precision(np.float32(3.14159265359), 2) == 3.14
        assert limit_float_precision(np.float64(3.14159265359), 4) == 3.1416
        assert limit_float_precision(np.float16(2.0), 2) == 2.0

    def test_limit_float_precision_non_float_types(self) -> None:
        """Test that non-float types are unchanged."""
        assert limit_float_precision("string", 2) == "string"
        assert limit_float_precision(42, 2) == 42
        assert limit_float_precision(True, 2) is True
        assert limit_float_precision(None, 2) is None

    def test_limit_float_precision_simple_list(self) -> None:
        """Test precision limiting for lists with floats."""
        input_list = [1.23456, 2.0, "string", 42, 9.87654]
        expected = [1.23, 2.0, "string", 42, 9.88]
        assert limit_float_precision(input_list, 2) == expected

    def test_limit_float_precision_simple_dict(self) -> None:
        """Test precision limiting for dictionaries with floats."""
        input_dict = {
            "float_val": 3.14159,
            "string_val": "test",
            "int_val": 42,
            "another_float": 2.718281828,
        }
        expected = {
            "float_val": 3.14,
            "string_val": "test",
            "int_val": 42,
            "another_float": 2.72,
        }
        assert limit_float_precision(input_dict, 2) == expected

    def test_limit_float_precision_nested_dict_in_list(self) -> None:
        """Test precision limiting for dictionaries nested in lists."""
        input_data = [
            {"score": 0.123456, "name": "item1"},
            {"score": 0.987654, "name": "item2"},
            "string_item",
            42,
        ]
        expected = [
            {"score": 0.123, "name": "item1"},
            {"score": 0.988, "name": "item2"},
            "string_item",
            42,
        ]
        assert limit_float_precision(input_data, 3) == expected

    def test_limit_float_precision_nested_list_in_dict(self) -> None:
        """Test precision limiting for lists nested in dictionaries."""
        input_data = {
            "scores": [0.12345, 0.67890, 0.99999],
            "names": ["A", "B", "C"],
            "metadata": {"threshold": 0.54321, "version": "1.0"},
        }
        expected = {
            "scores": [0.12, 0.68, 1.0],
            "names": ["A", "B", "C"],
            "metadata": {"threshold": 0.54, "version": "1.0"},
        }
        assert limit_float_precision(input_data, 2) == expected

    def test_limit_float_precision_deeply_nested(self) -> None:
        """Test precision limiting for deeply nested structures."""
        input_data = {
            "level1": {
                "level2": {
                    "level3": [
                        {"deep_float": 3.14159265359, "items": [1.23456, 7.89012]},
                        {"another_deep": 2.71828, "values": [9.87654, 5.43210]},
                    ]
                }
            }
        }
        expected = {
            "level1": {
                "level2": {
                    "level3": [
                        {"deep_float": 3.1416, "items": [1.2346, 7.8901]},
                        {"another_deep": 2.7183, "values": [9.8765, 5.4321]},
                    ]
                }
            }
        }
        assert limit_float_precision(input_data, 4) == expected

    def test_limit_float_precision_tuples(self) -> None:
        """Test precision limiting for tuples."""
        input_tuple = (1.23456, "string", 7.89012, 42)
        expected = (1.23, "string", 7.89, 42)
        assert limit_float_precision(input_tuple, 2) == expected

    def test_limit_float_precision_mixed_numpy_types(self) -> None:
        """Test precision limiting with mixed numpy and Python floats."""
        input_data = {
            "python_float": 3.14159,
            "numpy_float32": np.float32(2.71828),
            "numpy_float64": np.float64(1.41421),
            "list_mixed": [1.23456, np.float32(9.87654), "string", np.float64(5.55555)],
        }
        expected = {
            "python_float": 3.14,
            "numpy_float32": 2.72,
            "numpy_float64": 1.41,
            "list_mixed": [1.23, 9.88, "string", 5.56],
        }
        assert limit_float_precision(input_data, 2) == expected

    def test_write_json_with_precision(self, tmp_path) -> None:
        """Test write_json function with precision parameter."""
        test_data = {
            "predictions": [
                {
                    "filepath": "test.jpg",
                    "scores": [0.123456789, 0.987654321],
                    "bbox": [0.111111, 0.222222, 0.333333, 0.444444],
                    "confidence": 0.876543210,
                    "nested": {
                        "value": 1.414213562,
                        "items": [2.718281828, 3.141592654],
                    },
                }
            ]
        }

        output_file = tmp_path / "test_precision.json"
        write_json(test_data, output_file, num_decimals=3)

        # Read the file back and verify precision was limited
        loaded_data = load_json(output_file)
        prediction = loaded_data["predictions"][0]

        assert prediction["scores"] == [0.123, 0.988]
        assert prediction["bbox"] == [0.111, 0.222, 0.333, 0.444]
        assert prediction["confidence"] == 0.877
        assert prediction["nested"]["value"] == 1.414
        assert prediction["nested"]["items"] == [2.718, 3.142]

    def test_write_json_without_precision(self, tmp_path) -> None:
        """Test write_json function without precision parameter.

        Should preserve original precision.
        """
        test_data = {"value": 3.14159265359, "scores": [0.123456789, 0.987654321]}

        output_file = tmp_path / "test_no_precision.json"
        write_json(test_data, output_file)

        loaded_data = load_json(output_file)
        assert loaded_data["value"] == 3.14159265359
        assert loaded_data["scores"] == [0.123456789, 0.987654321]
