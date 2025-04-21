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

"""Geofence-related utility functions.

Provides functions for checking geofencing rules and rolling up
classification labels based on geographic restrictions and taxonomic levels.
"""

from typing import Optional

from speciesnet.constants import Classification
from speciesnet.taxonomy_utils import get_ancestor_at_level
from speciesnet.taxonomy_utils import get_full_class_string

# Handy type aliases.
PredictionLabelType = str
PredictionScoreType = float
PredictionSourceType = str
PredictionType = tuple[PredictionLabelType, PredictionScoreType, PredictionSourceType]


def should_geofence_animal_classification(
    label: str,
    country: Optional[str],
    admin1_region: Optional[str],
    geofence_map: dict,
    enable_geofence: bool,
) -> bool:
    """Checks whether to geofence animal prediction in a country or admin1_region.

    Args:
        label:
            Animal label to check geofence rules for.
        country:
            Country (in ISO 3166-1 alpha-3 format) to check geofence rules for.
            Optional.
        admin1_region:
            First-level administrative division (in ISO 3166-2 format) to check
            geofence rules for. Optional.
        geofence_map:
            Dictionary mapping full class strings to geofence rules.
        enable_geofence:
            Whether geofencing is enabled.

    Returns:
        A boolean indicating whether to geofence given animal prediction.
    """

    # Do not geofence if geofencing is disabled.
    if not enable_geofence:
        return False

    # Do not geofence if country was not provided.
    if not country:
        return False

    # Do not geofence if full class string is missing from the geofence map.
    full_class_string = get_full_class_string(label)
    if full_class_string not in geofence_map:
        return False

    # Check if we need to geofence based on "allow" rules.
    allow_countries = geofence_map[full_class_string].get("allow")
    if allow_countries:
        if country not in allow_countries:
            # Geofence when country was not explicitly allowed.
            return True
        else:
            allow_admin1_regions = allow_countries[country]
            if (
                admin1_region
                and allow_admin1_regions
                and admin1_region not in allow_admin1_regions
            ):
                # Geofence when admin1_region was not explicitly allowed.
                return True

    # Check if we need to geofence based on "block" rules.
    block_countries = geofence_map[full_class_string].get("block")
    if block_countries:
        if country in block_countries:
            block_admin1_regions = block_countries[country]
            if not block_admin1_regions:
                # Geofence when entire country was blocked.
                return True
            elif admin1_region and admin1_region in block_admin1_regions:
                # Geofence when admin1_region was blocked.
                return True

    # Do not geofence if no rule enforced that.
    return False


def roll_up_labels_to_first_matching_level(  # pylint: disable=too-many-positional-arguments
    labels: list[str],
    scores: list[float],
    country: Optional[str],
    admin1_region: Optional[str],
    target_taxonomy_levels: list[str],
    non_blank_threshold: float,
    taxonomy_map: dict,
    geofence_map: dict,
    enable_geofence: bool,
) -> Optional[PredictionType]:
    """Rolls up prediction labels to the first taxonomy level above given threshold.

    Args:
        labels:
            List of classification labels.
        scores:
            List of classification scores.
        country:
            Country (in ISO 3166-1 alpha-3 format) associated with prediction.
            Optional.
        admin1_region:
            First-level administrative division (in ISO 3166-2 format) associated
            with prediction. Optional.
        target_taxonomy_levels:
            Ordered list of taxonomy levels at which to roll up classification
            labels and check if the cumulative score passes the given threshold.
            Levels must be a subset of: "species", "genus", "family", "order",
            "class", "kingdom".
        non_blank_threshold:
            Min threshold at which the cumulative score is good enough to consider
            the rollup successful.
        taxonomy_map:
            Dictionary mapping taxa to labels.
        geofence_map:
            Dictionary mapping full class strings to geofence rules.
        enable_geofence:
            Whether geofencing is enabled.

    Returns:
        A tuple of <label, score, prediction_source> describing the first taxonomy
        level at which the cumulative score passes the given threshold. If no such
        level exists, return `None`.

    Raises:
        ValueError:
            If the taxonomy level if not one of: "species", "genus", "family",
            "order", "class", "kingdom".
    """

    expected_target_taxonomy_levels = {
        "species",
        "genus",
        "family",
        "order",
        "class",
        "kingdom",
    }
    unknown_target_taxonomy_levels = set(target_taxonomy_levels).difference(
        expected_target_taxonomy_levels
    )
    if unknown_target_taxonomy_levels:
        raise ValueError(
            "Unexpected target taxonomy level(s): "
            f"{unknown_target_taxonomy_levels}. "
            f"Expected only levels from the set: {expected_target_taxonomy_levels}."
        )

    # Accumulate scores at each taxonomy level and, if they pass the desired
    # threshold, return that rollup label.
    for taxonomy_level in target_taxonomy_levels:
        accumulated_scores = {}
        for label, score in zip(labels, scores):
            rollup_label = get_ancestor_at_level(
                label=label, taxonomy_level=taxonomy_level, taxonomy_map=taxonomy_map
            )
            if rollup_label:
                new_score = accumulated_scores.get(rollup_label, 0.0) + score
                accumulated_scores[rollup_label] = new_score

        max_rollup_label = None
        max_rollup_score = 0.0
        for rollup_label, rollup_score in accumulated_scores.items():
            if (
                rollup_score > max_rollup_score
                and not should_geofence_animal_classification(
                    rollup_label, country, admin1_region, geofence_map, enable_geofence
                )
            ):
                max_rollup_label = rollup_label
                max_rollup_score = rollup_score
        if max_rollup_score > non_blank_threshold and max_rollup_label:
            return (
                max_rollup_label,
                max_rollup_score,
                f"classifier+rollup_to_{taxonomy_level}",
            )

    return None


def geofence_animal_classification(
    *,
    labels: list[str],
    scores: list[float],
    country: Optional[str],
    admin1_region: Optional[str],
    taxonomy_map: dict,
    geofence_map: dict,
    enable_geofence: bool,
) -> PredictionType:
    """Geofences animal prediction in a country or admin1_region.

    Under the hood, this also rolls up the labels every time it encounters a
    geofenced label.

    Args:
        labels:
            List of classification labels.
        scores:
            List of classification scores.
        country:
            Country (in ISO 3166-1 alpha-3 format) associated with prediction.
            Optional.
        admin1_region:
            First-level administrative division (in ISO 3166-2 format) associated
            with prediction. Optional.
        taxonomy_map:
            Dictionary mapping taxa to labels.
        geofence_map:
            Dictionary mapping full class strings to geofence rules.
        enable_geofence:
            Whether geofencing is enabled.

    Returns:
        A tuple of <label, score, prediction_source> describing the result of the
        combined geofence and rollup operations.
    """

    if should_geofence_animal_classification(
        labels[0], country, admin1_region, geofence_map, enable_geofence
    ):
        rollup = roll_up_labels_to_first_matching_level(
            labels=labels,
            scores=scores,
            country=country,
            admin1_region=admin1_region,
            target_taxonomy_levels=["family", "order", "class", "kingdom"],
            # Force the rollup to pass the top classification score.
            non_blank_threshold=scores[0] - 1e-10,
            taxonomy_map=taxonomy_map,
            geofence_map=geofence_map,
            enable_geofence=enable_geofence,
        )
        if rollup:
            rollup_label, rollup_score, rollup_source = rollup
            return (
                rollup_label,
                rollup_score,
                "classifier+geofence+" + rollup_source[len("classifier+") :],
            )
        else:
            # Normally, this return statement should never be reached since the
            # animal rollup would eventually succeed (even though that may be at
            # "kingdom" level, as a last resort). The only scenario when this could
            # still be reached is if the method was incorrectly called with a list
            # of non-animal labels (e.g. blanks, vehicles). In this case it's best
            # to return an unknown classification, while propagating the top score.
            return (
                Classification.UNKNOWN,
                scores[0],
                "classifier+geofence+rollup_failed",
            )
    else:
        return labels[0], scores[0], "classifier"
