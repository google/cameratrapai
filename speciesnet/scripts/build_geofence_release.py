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

"""Script to build the geofence release from geofence base with extra manual fixes."""

import copy
import json
from pathlib import Path
from typing import Union

from absl import app
from absl import flags
from absl import logging
import pandas as pd

_BASE = flags.DEFINE_string(
    "base",
    "data/geofence_base.json",
    "Path to the geofence base (JSON). Used as a starting point for constructing the "
    "geofence release.",
)
_FIXES = flags.DEFINE_string(
    "fixes",
    "data/geofence_fixes.csv",
    "Path to the geofence fixes (CSV). Used to correct mistakes in the geofence base.",
)
_TRIM = flags.DEFINE_string(
    "trim",
    None,
    "Path to the labels supported by the model (TXT). Used to trim the geofence "
    "release.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Output path for writing the geofence release (JSON).",
    required=True,
)

# Handy type alias.
StrPath = Union[str, Path]


def load_geofence_base(path: StrPath) -> dict[str, dict]:

    with open(path, mode="r", encoding="utf-8") as fp:
        data = json.load(fp)
    for label, rules in data.items():
        if label.endswith(";"):
            raise ValueError(
                "Base geofence should provide only species-level rules. "
                f"Found higher taxa rule with the label: `{label}`"
            )
        if "block" in rules:
            raise ValueError("Block rules are not accepted in base geofence.")
    return data


def fix_geofence_base(
    geofence_base: dict[str, dict], fixes_path: StrPath
) -> dict[str, dict]:

    geofence = copy.deepcopy(geofence_base)

    fixes = pd.read_csv(fixes_path, keep_default_na=False)
    for idx, fix in fixes.iterrows():
        label = fix["species"].lower()
        label_parts = label.split(";")
        if len(label_parts) != 5:
            raise ValueError(
                "Fixes should always use five-token taxon strings"
            )
        rule = fix["rule"].lower()
        if rule not in {"allow", "block"}:
            raise ValueError(
                "Rule types should be either `allow` or `block`. "
                f"Please correct rule #{idx + 1}:\n{fix}"
            )

        country = fix["country_code"]
        state = fix["admin1_region_code"]

        if rule == "allow":
            if label not in geofence:
                continue  # already allowed
            if "allow" not in geofence[label]:
                continue  # already allowed
            if not state:
                geofence[label]["allow"][country] = geofence[label]["allow"].get(
                    country, []
                )
            else:
                curr_country_rule = geofence[label]["allow"].get(country)
                if curr_country_rule is None:  # missing country rule
                    geofence[label]["allow"][country] = [state]
                else:
                    if not curr_country_rule:  # an empty list
                        continue  # already allowed
                    else:  # not an empty list
                        geofence[label]["allow"][country] = sorted(
                            set(curr_country_rule) | {state}
                        )
        else:  # rule == "block"
            if label not in geofence:
                geofence[label] = {"block": {country: [state] if state else []}}
            if "block" not in geofence[label]:
                geofence[label]["block"] = {country: [state] if state else []}
            if not state:
                geofence[label]["block"][country] = geofence[label]["block"].get(
                    country, []
                )
            else:
                curr_country_rule = geofence[label]["block"].get(country)
                if curr_country_rule is None:  # missing country rule
                    geofence[label]["block"][country] = [state]
                else:
                    if not curr_country_rule:  # an empty list
                        continue  # already blocked
                    else:  # not an empty list
                        geofence[label]["block"][country] = sorted(
                            set(curr_country_rule) | {state}
                        )

    return geofence


def propagate_to_higher_taxa(geofence: dict[str, dict]) -> dict[str, dict]:

    new_geofence = {}

    for label, rule in geofence.items():

        label_parts = label.split(";")

        # Keep original rule.
        new_geofence[label] = rule

        # Propagate to higher taxa.
        for taxa_level_end in range(1, 5):
            new_label = ";".join(label_parts[:taxa_level_end]) + (
                ";" * (5 - taxa_level_end)
            )
            if new_label not in new_geofence:
                new_geofence[new_label] = {"allow": {}}

            # Country-wide "allow" rules at species level get propagated directly, but
            # regional "allow" rules become country wide "allow" rules at genus level
            # and above.
            if "allow" in rule:
                for country in rule["allow"]:
                    new_geofence[new_label]["allow"][country] = []

    return new_geofence


def trim_to_supported_labels(
    geofence: dict[str, dict], labels_path: StrPath
) -> dict[str, dict]:

    with open(labels_path, mode="r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp.readlines()]
        labels = set()
        for line in lines:
            label_parts = line.split(";")[1:6]
            for taxa_level_end in range(1, 6):
                new_label = ";".join(label_parts[:taxa_level_end]) + (
                    ";" * (5 - taxa_level_end)
                )
                labels.add(new_label)

    return {k: v for k, v in geofence.items() if k in labels}


def save_geofence(geofence: dict[str, dict], output_path: StrPath) -> None:

    with open(output_path, mode="w", encoding="utf-8") as fp:
        json.dump(geofence, fp, indent=4, sort_keys=True)


def main(argv: list[str]) -> None:
    del argv  # Unused.

    geofence_base = load_geofence_base(_BASE.value)
    geofence_release = fix_geofence_base(geofence_base, _FIXES.value)
    geofence_release = propagate_to_higher_taxa(geofence_release)
    if _TRIM.value:
        logging.info(
            "Trimming to labels (and their corresponding higher taxa) from `%s`.",
            _TRIM.value,
        )
        geofence_release = trim_to_supported_labels(geofence_release, _TRIM.value)
    else:
        logging.info("No trimming was performed.")
    save_geofence(geofence_release, _OUTPUT.value)


if __name__ == "__main__":
    app.run(main)
