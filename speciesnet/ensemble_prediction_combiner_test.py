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
# pylint: disable=redefined-outer-name

import pytest
from speciesnet.constants import Classification
from speciesnet.constants import Detection
from speciesnet.ensemble_prediction_combiner import combine_predictions

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


@pytest.fixture
def taxonomy_map():
  return {
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


@pytest.fixture
def geofence_map():
  return {
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


def mock_geofence_fn(*args, **kwargs):
  del args
  del kwargs
  return "geofenced_label", 0.9, "geofence_source"


def mock_roll_up_fn(*args, **kwargs):
  del args
  del kwargs
  return "rollup_label", 0.8, "rollup_source"


def mock_roll_up_fn_return_none(*args, **kwargs):
  del args
  del kwargs


def combine_predictions_for_single_item(
    classifications, 
    detections, 
    country, 
    admin1_region, 
    taxonomy_map, 
    geofence_map, 
    enable_geofence, 
    geofence_fn, 
    roll_up_fn
):
    """Test helper transforming single-item permutations into the list format."""
    results = combine_predictions(
        classifications_list=[classifications],
        detections=detections,
        country=country,
        admin1_region=admin1_region,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=enable_geofence,
        geofence_fn=geofence_fn,
        roll_up_fn=roll_up_fn,
    )
    # The tests expect a flat tuple of (label, score, source) to be asserted
    return results[0]["prediction"], results[0]["prediction_score"], results[0]["prediction_source"]

class TestPredictionEnsembleCombiner:
  """Tests for the ensemble combiner function."""

  def test_combine_predictions_for_single_item_human_detections(
      self, taxonomy_map, geofence_map
  ) -> None:
    # High-confidence HUMAN detection.
    classifications = {"classes": [LION], "scores": [0.1]}
    detections = [{"label": Detection.HUMAN, "conf": 0.8}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.HUMAN, 0.8, "detector")

    # Mid-confidence HUMAN detection + high-confidence HUMAN classification.
    classifications = {"classes": [HUMAN], "scores": [0.7]}
    detections = [{"label": Detection.HUMAN, "conf": 0.3}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.HUMAN, 0.7, "classifier")

    # Mid-confidence HUMAN detection + high-confidence VEHICLE classification.
    classifications = {"classes": [VEHICLE], "scores": [0.7]}
    detections = [{"label": Detection.HUMAN, "conf": 0.3}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.HUMAN, 0.7, "classifier")

  def test_combine_predictions_for_single_item_vehicle_detections(
      self, taxonomy_map, geofence_map
  ) -> None:
    # Mid-confidence VEHICLE detection + high-confidence HUMAN classification.
    classifications = {"classes": [HUMAN], "scores": [0.7]}
    detections = [{"label": Detection.VEHICLE, "conf": 0.3}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.HUMAN, 0.7, "classifier")

    # High-confidence VEHICLE detection.
    classifications = {"classes": [LION], "scores": [0.1]}
    detections = [{"label": Detection.VEHICLE, "conf": 0.8}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.VEHICLE, 0.8, "detector")

    # Mid-confidence VEHICLE detection + high-confidence VEHICLE classification.
    classifications = {"classes": [VEHICLE], "scores": [0.5]}
    detections = [{"label": Detection.VEHICLE, "conf": 0.3}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.VEHICLE, 0.5, "classifier")

  def test_combine_predictions_for_single_item_blank_classifications(
      self, taxonomy_map, geofence_map
  ) -> None:
    # High-confidence BLANK "detection" + high-confidence BLANK classification.
    classifications = {"classes": [BLANK], "scores": [0.7]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.1}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.BLANK, 0.7, "classifier")

    # Extra-high-confidence BLANK classification.
    classifications = {"classes": [BLANK], "scores": [0.995]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.3}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == (Classification.BLANK, 0.995, "classifier")

  def test_combine_predictions_for_single_item_animal_classifications(
      self, taxonomy_map, geofence_map
  ) -> None:
    # Extra-high-confidence ANIMAL classification.
    classifications = {"classes": [LION], "scores": [0.9]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.1}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == ("geofenced_label", 0.9, "geofence_source")

    # High-confidence ANIMAL classification + mid-confidence ANIMAL detection.
    classifications = {"classes": [LION], "scores": [0.7]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.3}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == ("geofenced_label", 0.9, "geofence_source")

  def test_combine_predictions_for_single_item_animal_rollups(
      self, taxonomy_map, geofence_map
  ) -> None:
    # High-confidence ANIMAL rollups.
    classifications = {"classes": [LION], "scores": [0.6]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.1}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn,
    )
    assert result == ("rollup_label", 0.8, "rollup_source")

  def test_combine_predictions_for_single_item_animal_detections(
      self, taxonomy_map, geofence_map
  ) -> None:
    # Mid-confidence ANIMAL detection.
    classifications = {"classes": [LION], "scores": [0.1]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.6}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn_return_none,
    )
    assert result == (Classification.ANIMAL, 0.6, "detector")

  def test_combine_predictions_for_single_item_unknown(
      self, taxonomy_map, geofence_map
  ) -> None:
    # Fallback to UNKNOWN classification.
    classifications = {"classes": [LION], "scores": [0.1]}
    detections = [{"label": Detection.ANIMAL, "conf": 0.1}]
    result = combine_predictions_for_single_item(
        classifications=classifications,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn_return_none,
    )
    assert result == (Classification.UNKNOWN, 0.1, "classifier")

  def test_combine_predictions_for_all_items(
      self, taxonomy_map, geofence_map
  ) -> None:
    classifications_list = [
        {"classes": [HUMAN], "scores": [0.7]},
        {"classes": [LION], "scores": [0.9]},
        {"classes": [VEHICLE], "scores": [0.8]},
    ]
    detections = [
        {"label": Detection.HUMAN, "conf": 0.8, "bbox": [0.1, 0.1, 0.2, 0.2]},
        {"label": Detection.ANIMAL, "conf": 0.9, "bbox": [0.5, 0.5, 0.6, 0.6]},
        {"label": Detection.VEHICLE, "conf": 0.9, "bbox": [0.8, 0.8, 0.9, 0.9]},
    ]

    results = combine_predictions(
        classifications_list=classifications_list,
        detections=detections,
        country=None,
        admin1_region=None,
        taxonomy_map=taxonomy_map,
        geofence_map=geofence_map,
        enable_geofence=True,
        geofence_fn=mock_geofence_fn,
        roll_up_fn=mock_roll_up_fn_return_none,
    )

    assert len(results) == 3

    # First is HUMAN
    assert results[0]["prediction"] == Classification.HUMAN
    assert results[0]["prediction_score"] == 0.8
    assert results[0]["prediction_source"] == "detector"
    assert results[0]["bbox"] == [0.1, 0.1, 0.2, 0.2]

    # Second is LION (gets geofenced)
    assert results[1]["prediction"] == "geofenced_label"
    assert results[1]["prediction_score"] == 0.9
    assert results[1]["prediction_source"] == "geofence_source"
    assert results[1]["bbox"] == [0.5, 0.5, 0.6, 0.6]

    # Third is VEHICLE
    assert results[2]["prediction"] == Classification.VEHICLE
    assert results[2]["prediction_score"] == 0.9
    assert results[2]["prediction_source"] == "detector"
    assert results[2]["bbox"] == [0.8, 0.8, 0.9, 0.9]
