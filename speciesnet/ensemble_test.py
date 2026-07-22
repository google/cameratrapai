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
# pylint: disable=protected-access

import warnings

import pytest

from speciesnet.constants import Classification
from speciesnet.ensemble import PredictionType
from speciesnet.ensemble import SpeciesNetEnsemble
from speciesnet.taxonomy_utils import get_ancestor_at_level
from speciesnet.utils import load_rgb_image

# fmt: off
# pylint: disable=line-too-long

BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
BLANK_FC = ";;;;"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
HUMAN_FC = "mammalia;primates;hominidae;homo;sapiens"
VEHICLE = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
VEHICLE_FC = ";;;;"

LION = "ddf59264-185a-4d35-b647-2785792bdf54;mammalia;carnivora;felidae;panthera;leo;lion"
LION_FC = "mammalia;carnivora;felidae;panthera;leo"
PANTHERA_GENUS = "fbb23d07-6677-43db-b650-f99ac452c50f;mammalia;carnivora;felidae;panthera;;panthera species"
PANTHERA_GENUS_FC = "mammalia;carnivora;felidae;panthera;"
FELIDAE_FAMILY = "df8514b0-10a5-411f-8ed6-0f415e8153a3;mammalia;carnivora;felidae;;;cat family"
FELIDAE_FAMILY_FC = "mammalia;carnivora;felidae;;"
CARNIVORA_ORDER = "eeeb5d26-2a47-4d01-a3de-10b33ec0aee4;mammalia;carnivora;;;;carnivorous mammal"
CARNIVORA_ORDER_FC = "mammalia;carnivora;;;"
MAMMALIA_CLASS = "f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal"
MAMMALIA_CLASS_FC = "mammalia;;;;"
ANIMAL_KINGDOM = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"
ANIMAL_KINGDOM_FC = ";;;;"

BROWN_BEAR = "330bb1e9-84d6-4e41-afa9-938aee17ea29;mammalia;carnivora;ursidae;ursus;arctos;brown bear"
BROWN_BEAR_FC = "mammalia;carnivora;ursidae;ursus;arctos"
POLAR_BEAR = "e7f83bf6-df2c-4ce0-97fc-2f233df23ec4;mammalia;carnivora;ursidae;ursus;maritimus;polar bear"
POLAR_BEAR_FC = "mammalia;carnivora;ursidae;ursus;maritimus"
GIANT_PANDA = "85662682-67c1-4ecb-ba05-ba12e2df6b65;mammalia;carnivora;ursidae;ailuropoda;melanoleuca;giant panda"
GIANT_PANDA_FC = "mammalia;carnivora;ursidae;ailuropoda;melanoleuca"
URSUS_GENUS = "5a0f5e3f-c634-4b86-910a-b105cb526a24;mammalia;carnivora;ursidae;ursus;;ursus species"
URSUS_GENUS_FC = "mammalia;carnivora;ursidae;ursus;"
URSIDAE_FAMILY = "ec1a70f4-41c0-4aba-9150-292fb2b7a324;mammalia;carnivora;ursidae;;;bear family"
URSIDAE_FAMILY_FC = "mammalia;carnivora;ursidae;;"

PUMA = "9c564562-9429-405c-8529-04cff7752282;mammalia;carnivora;felidae;puma;concolor;puma"
PUMA_FC = "mammalia;carnivora;felidae;puma;concolor"
SAND_CAT = "e588253d-d61d-4149-a96c-8c245927a80f;mammalia;carnivora;felidae;felis;margarita;sand cat"
SAND_CAT_FC = "mammalia;carnivora;felidae;felis;margarita"

# pylint: enable=line-too-long
# fmt: on


class TestEnsemble:
    """Tests for the ensemble component."""

    @pytest.fixture(scope="class")
    def ensemble(self, model_name: str) -> SpeciesNetEnsemble:
        return SpeciesNetEnsemble(model_name)

    @pytest.fixture
    def mock_ensemble(self, monkeypatch, ensemble) -> SpeciesNetEnsemble:
        taxonomy_map = {
            BLANK_FC: BLANK,
            HUMAN_FC: HUMAN,
            VEHICLE_FC: VEHICLE,
            LION_FC: LION,
            PANTHERA_GENUS_FC: PANTHERA_GENUS,
            FELIDAE_FAMILY_FC: FELIDAE_FAMILY,
            CARNIVORA_ORDER_FC: CARNIVORA_ORDER,
            MAMMALIA_CLASS_FC: MAMMALIA_CLASS,
            ANIMAL_KINGDOM_FC: ANIMAL_KINGDOM,
            BROWN_BEAR_FC: BROWN_BEAR,
            POLAR_BEAR_FC: POLAR_BEAR,
            GIANT_PANDA_FC: GIANT_PANDA,
            URSUS_GENUS_FC: URSUS_GENUS,
            URSIDAE_FAMILY_FC: URSIDAE_FAMILY,
        }
        monkeypatch.setattr(ensemble, "taxonomy_map", taxonomy_map)

        geofence_map = {
            LION_FC: {
                "allow": {
                    "KEN": [],
                    "TZA": [],
                }
            },
            PANTHERA_GENUS_FC: {
                "allow": {
                    "KEN": [],
                    "TZA": [],
                    "USA": ["AK", "CA"],
                }
            },
            FELIDAE_FAMILY_FC: {
                "allow": {
                    "FRA": [],
                    "KEN": [],
                    "TZA": [],
                    "USA": [],
                },
                "block": {
                    "FRA": [],
                    "USA": ["NY"],
                },
            },
            SAND_CAT_FC: {
                "block": {
                    "AUS": [],
                },
            },
            URSIDAE_FAMILY_FC: {
                "block": {
                    "GBR": [],
                },
            },
        }
        monkeypatch.setattr(ensemble, "geofence_map", geofence_map)

        return ensemble

    @pytest.fixture
    def mock_ensemble_no_geofence(
        self, monkeypatch, mock_ensemble
    ) -> SpeciesNetEnsemble:
        monkeypatch.setattr(mock_ensemble, "enable_geofence", False)
        return mock_ensemble

    @pytest.fixture
    def mock_ensemble2(self, monkeypatch, ensemble) -> SpeciesNetEnsemble:

        def prediction_combiner_mock(
            classifications_list: list[dict], *args, **kwargs
        ) -> list[dict]:
            del args  # Unused.
            del kwargs  # Unused.
            return [{
                "prediction": classifications_list[0]["classes"][0],
                "prediction_score": classifications_list[0]["scores"][0],
                "prediction_source": "mock"
            }]

        monkeypatch.setattr(
            ensemble,
            "prediction_combiner",
            prediction_combiner_mock,
        )
        return ensemble

    def test_combine(self, mock_ensemble2) -> None:

        blank_img = load_rgb_image("test_data/blank.jpg")
        blank2_img = load_rgb_image("test_data/blank2.jpg")
        blank3_img = load_rgb_image("test_data/blank3.jpg")
        assert blank_img and blank2_img and blank3_img

        expected_model_version = mock_ensemble2.model_info.version

        filepaths = [
            "a.jpg",
            "b.jpg",
            "c.jpg",
            "d.jpg",
            "e.jpg",
            "f.jpg",
        ]
        classifier_results = {
            "b.jpg": {
                "filepath": "b.jpg",
                "failures": ["CLASSIFIER"],
            },
            "c.jpg": {
                "filepath": "c.jpg",
                "classifications_list": [{
                    "classes": ["X", "Y", "Z"],
                    "scores": [0.5, 0.3, 0.2],
                }],
            },
            "d.jpg": {
                "filepath": "d.jpg",
                "classifications_list": [{
                    "classes": ["R", "S", "T"],
                    "scores": [0.7, 0.2, 0.1],
                }],
            },
            "e.jpg": {
                "filepath": "e.jpg",
                "classifications_list": [{
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                }],
            },
            "f.jpg": {
                "filepath": "f.jpg",
                "classifications_list": [{
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                }],
            },
        }
        detector_results = {
            "a.jpg": {
                "filepath": "a.jpg",
                "failures": ["DETECTOR"],
            },
            "b.jpg": {
                "filepath": "b.jpg",
                "detections": [
                    {
                        "category": "1",
                        "label": "animal",
                        "conf": 0.5,
                        "bbox": [0.0, 0.1, 0.2, 0.3],
                    }
                ],
            },
            "d.jpg": {
                "filepath": "d.jpg",
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            },
            "e.jpg": {
                "filepath": "e.jpg",
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            },
            "f.jpg": {
                "filepath": "f.jpg",
                "detections": [],
            },
        }
        geolocation_results = {
            "a.jpg": {
                "country": "COUNTRY_A",
            },
            "b.jpg": {
                "country": "COUNTRY_B",
            },
            "c.jpg": {
                "country": "COUNTRY_C",
            },
            "e.jpg": {
                "country": "COUNTRY_E",
            },
            "f.jpg": {
                "country": "COUNTRY_F",
            },
        }
        partial_predictions = {
            "f.jpg": {
                "filepath": "f.jpg",
                "classifications_list": [{
                    "classes": ["XYZ"],
                    "scores": [0.8],
                }],
                "detections": [],
                "ensemble_predictions": [{
                    "prediction": "XYZ",
                    "prediction_score": 0.4,
                    "prediction_source": "partial"
                }],
                "model_version": expected_model_version,
            },
        }

        assert mock_ensemble2.combine(
            filepaths,
            classifier_results,
            detector_results,
            geolocation_results,
            partial_predictions,
        ) == [
            {
                "filepath": "a.jpg",
                "failures": ["CLASSIFIER", "DETECTOR"],
                "country": "COUNTRY_A",
                "model_version": expected_model_version,
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
                "model_version": expected_model_version,
            },
            {
                "filepath": "c.jpg",
                "failures": ["DETECTOR"],
                "country": "COUNTRY_C",
                "classifications_list": [{
                    "classes": ["X", "Y", "Z"],
                    "scores": [0.5, 0.3, 0.2],
                }],
                "model_version": expected_model_version,
            },
            {
                "filepath": "d.jpg",
                "failures": ["GEOLOCATION"],
                "classifications_list": [{
                    "classes": ["R", "S", "T"],
                    "scores": [0.7, 0.2, 0.1],
                }],
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
                "ensemble_predictions": [{
                    "prediction": "R",
                    "prediction_score": 0.7,
                    "prediction_source": "mock"
                }],
                "model_version": expected_model_version,
            },
            {
                "filepath": "e.jpg",
                "country": "COUNTRY_E",
                "classifications_list": [{
                    "classes": ["K", "L", "M"],
                    "scores": [0.9, 0.1, 0.0],
                }],
                "detections": [
                    {
                        "category": "2",
                        "label": "human",
                        "conf": 0.7,
                        "bbox": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
                "ensemble_predictions": [{
                    "prediction": "K",
                    "prediction_score": 0.9,
                    "prediction_source": "mock"
                }],
                "model_version": expected_model_version,
            },
            {
                "filepath": "f.jpg",
                "classifications_list": [{
                    "classes": ["XYZ"],
                    "scores": [0.8],
                }],
                "detections": [],
                "ensemble_predictions": [{
                    "prediction": "XYZ",
                    "prediction_score": 0.4,
                    "prediction_source": "partial"
                }],
                "model_version": expected_model_version,
            },
        ]

    def test_multi_combine(self, mock_ensemble2) -> None:
        expected_model_version = mock_ensemble2.model_info.version

        classifications_list = [
            {
                "classes": ["R", "S", "T"],
                "scores": [0.7, 0.2, 0.1],
            }
        ]
        detections = [
            {
                "category": "2",
                "label": "human",
                "conf": 0.7,
                "bbox": [0.1, 0.2, 0.3, 0.4],
            }
        ]
        
        result = mock_ensemble2.combine(
            filepaths=["d.jpg"],
            classifier_results={"d.jpg": {"classifications_list": classifications_list}},
            detector_results={"d.jpg": {"detections": detections}},
            geolocation_results={"d.jpg": {"country": "COUNTRY_D"}},
        )[0]["ensemble_predictions"]

        assert len(result) == 1
        assert result[0]["prediction"] == "R"
        assert result[0]["prediction_score"] == 0.7
        assert result[0]["prediction_source"] == "mock"

    def test_complete_taxonomy(self, ensemble) -> None:

        missing_ancestors = set()

        taxonomy_levels = {
            "kingdom": 0,
            "class": 1,
            "order": 2,
            "family": 3,
            "genus": 4,
            "species": 5,
        }

        def _taxa_from_label(label: str) -> str:
            return ";".join(label.split(";")[1:6])

        def _level_idx_from_label(label: str) -> int:
            label_parts = label.split(";")
            for idx in range(5, 0, -1):
                if label_parts[idx]:
                    return idx
            return 0

        def _ancestor_from_label(label: str, level_idx: int) -> str:
            label_parts = label.split(";")
            return ";".join(label_parts[1 : level_idx + 1]) + (";" * (5 - level_idx))

        with open(
            ensemble.model_info.classifier_labels, mode="r", encoding="utf-8"
        ) as fp:
            classifier_labels = [line.strip() for line in fp.readlines()]

        for label in [Classification.HUMAN, Classification.ANIMAL]:
            taxa = _taxa_from_label(label)
            assert taxa in ensemble.taxonomy_map
            assert ensemble.taxonomy_map[taxa] == label
        for label in [Classification.UNKNOWN]:
            taxa = _taxa_from_label(label)
            assert taxa not in ensemble.taxonomy_map

        for label in classifier_labels:
            if label in [
                Classification.BLANK,
                Classification.VEHICLE,
                Classification.UNKNOWN,
            ]:
                continue

            max_level_idx = _level_idx_from_label(label)

            for taxonomy_level, taxonomy_level_idx in taxonomy_levels.items():
                ancestor = get_ancestor_at_level(
                    label, taxonomy_level, ensemble.taxonomy_map
                )
                if taxonomy_level_idx <= max_level_idx:
                    if not ancestor:
                        missing_ancestors.add(
                            _ancestor_from_label(label, taxonomy_level_idx)
                        )
                else:
                    assert ancestor is None

        if missing_ancestors:
            warnings.warn(
                UserWarning(
                    "Missing from taxonomy: \n" + "\n".join(sorted(missing_ancestors))
                )
            )

    def test_complete_geofence(self, ensemble) -> None:

        missing_ancestors = set()

        def _taxa_from_label(label: str) -> str:
            return ";".join(label.split(";")[1:6])

        def _ancestor_from_label(label: str, level_idx: int) -> str:
            label_parts = label.split(";")
            return ";".join(label_parts[1 : level_idx + 1]) + (";" * (5 - level_idx))

        with open(
            ensemble.model_info.classifier_labels, mode="r", encoding="utf-8"
        ) as fp:
            classifier_labels = [line.strip() for line in fp.readlines()]

        for label in [
            Classification.BLANK,
            Classification.ANIMAL,
            Classification.VEHICLE,
            Classification.UNKNOWN,
        ]:
            taxa = _taxa_from_label(label)
            assert taxa not in ensemble.geofence_map

        for label in [Classification.HUMAN]:
            taxa = _taxa_from_label(label)
            assert taxa in ensemble.geofence_map

        for label in classifier_labels:
            if label in [
                Classification.BLANK,
                Classification.ANIMAL,
                Classification.VEHICLE,
                Classification.UNKNOWN,
            ]:
                continue

            for level_idx in range(1, 6):
                ancestor = _ancestor_from_label(label, level_idx)
                if ancestor not in ensemble.geofence_map:
                    missing_ancestors.add(ancestor)

        if missing_ancestors:
            warnings.warn(
                UserWarning(
                    "Missing from geofence: \n" + "\n".join(sorted(missing_ancestors))
                )
            )
