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

"""Script to build the geofence release from geofence base with extra manual fixes.

A geofencing .json file defines a "global geofencing dict".

Keys in a global geofencing dict are five-token SpeciesNet taxonomy strings,
for example:

aves;accipitriformes;accipitridae;accipiter;rhodogaster

Each item in a global geofencing dict is a "taxon geofencing dict", containing
"allow" and/or "block" rules for that taxon.  The only valid keys in a taxon
geofencing dict are "allow" and "block".  In the base geofence file, only "allow"
rules are valid.

"allow" or block" are mapped a to "regional rules dict".  Each key in a regional
rules dict is a three-letter country code, and the correponding value is a list
of regional rules within that country.  Currently that list is empty for all country
codes other than "USA".

Examples:

This taxon would be allowed in RUS, SGP, THA, TWAN, and VNM.  In the USA,
it would be allowed in AK but blocked in all other states.  It would be
blocked in all other countries.

    "aves;accipitriformes;accipitridae;accipiter;soloensis": {
        "allow": {
            "RUS": [],
            "SGP": [],
            "THA": [],
            "TWN": [],
            "USA": [
                "AK"
            ],
            "VNM": []
        }
    }

This taxon would be blocked in ABW and AFG, allowed everywhere else:

    "mammalia;cetartiodactyla;hippopotamidae;;": {
        "block": {
            "ABW": [],
            "AFG": []
        }

Additional conventions:

* If a taxon is not included in the geofence, it's allowed everywhere.
* If a taxon has only an empty allow-list, it's allowed everywhere.
* If allow rules exist for a taxon, any country not on the allow-list for that
  taxon is blocked.
* Block rules "win" over allow rules.  Taxa that are allowed in the base geofence
  may later get blocked

"""
import copy
import json
from pathlib import Path
from typing import Union
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import pandas as pd

from speciesnet.geofence_utils import should_geofence_animal_classification

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


def _taxon_allowed_in_region(
    label: str, country: str, admin1_region: Optional[str], geofence_map: dict
) -> bool:
    """Utility function to check whether a taxon is allowed in a
    region.  This is a thin wrapper for should_geofence_animal_classification.
    """

    # should_geofence_animal_classification accepts only seven-token
    # taxon strings, we want to use five-token strings.
    if len(label.split(";")) not in (5, 7):
        raise ValueError(f"Illegal label {label}")

    if len(label.split(";")) != 7:
        label = ";" + label + ";"

    return not should_geofence_animal_classification(
        label, country, admin1_region, geofence_map, enable_geofence=True
    )


def _validate_taxon_string(taxon: str) -> bool:
    """Validates a five-token taxon string.  Errors in invalid
    taxa, else returns True."""

    tokens = taxon.split(";")

    if (not isinstance(taxon, str)) or (len(tokens) != 5):
        print(f"Invalid taxon string {taxon} in geofence")
        return False

    # You can't specify, e.g., a species without a genus
    found_non_empty_level = False
    for token in tokens[::-1]:
        if len(token) > 0:
            found_non_empty_level = True
        else:
            if found_non_empty_level:
                raise ValueError(f"Illegal taxon {taxon}")

    return True


def _generate_parent_taxon_strings(s):
    """Given a five-token taxon string in a;b;c;d;e format,
    generates the taxon strings for all parents (e.g. a;b;c;d;;).
    """

    tokens = s.split(";")
    n_tokens = len(tokens)
    assert n_tokens == 5
    output_strings = []
    i_token = n_tokens - 1
    while i_token > 0:
        # Skip tokens that are already empty
        if len(tokens[i_token]) == 0:
            i_token -= 1
            continue
        else:
            tokens[i_token] = ""
            output_string = ";".join(tokens)
            output_strings.append(output_string)
    return output_strings


def validate_geofence(geofence: dict[str, dict]) -> bool:
    """Validates a global geofencing dict.  See module header for
    format rules.

    Args:
        path:
            Filename of the base geofence .json file.

    Returns:
        True if the geofencing dict is valid, else False.
    """

    if not isinstance(geofence, dict):
        print("Invalid geofence type")
        return False

    # Basic format validation
    for taxon in geofence.keys():

        # All keys should be five-token taxon strings
        _validate_taxon_string(taxon)

        taxon_rules = geofence[taxon]

        for rule_type in taxon_rules.keys():

            if (not isinstance(rule_type, str)) or (
                rule_type not in ("allow", "block")
            ):
                print(f"Invalid rule type {rule_type} for taxon {taxon}")
                return False

            countries = taxon_rules[rule_type]

            for country_code in countries.keys():
                if (not isinstance(country_code, str)) or (len(country_code) != 3):
                    print(f"Invalid country code {country_code} for taxon {taxon}")
                    return False
                regions = countries[country_code]
                if not isinstance(regions, list):
                    print(f"Invalid rules for {country_code} for taxon {taxon}")
                    return False
                if not all([isinstance(x, str) for x in regions]):
                    print(f"Invalid regions for {country_code} for taxon {taxon}")
                    return False

    # Make sure that if a taxon is explicitly allowed in a region, all of its parents
    # are allowed.
    for taxon in geofence.keys():

        if "allow" not in geofence[taxon]:
            continue

        parent_taxa = _generate_parent_taxon_strings(taxon)

        allowed_countries = geofence[taxon]["allow"]
        for country in allowed_countries.keys():
            allowed_regions = allowed_countries[country]
            if len(allowed_regions) == 0:
                allowed_regions = [None]
            for region in allowed_regions:
                # We know this taxon is allowed in this region
                assert _taxon_allowed_in_region(
                    label=taxon,
                    country=country,
                    admin1_region=region,
                    geofence_map=geofence,
                )
                for parent_taxon in parent_taxa:
                    allowed = _taxon_allowed_in_region(
                        label=parent_taxon,
                        country=country,
                        admin1_region=region,
                        geofence_map=geofence,
                    )
                    if not allowed:
                        raise ValueError(
                            f"Parent taxon {parent_taxon} of {taxon} not allowed in "
                            f"{country}:{region}"
                        )

    return True


def load_geofence_base(path: StrPath) -> dict[str, dict]:
    """Loads the geofence .json file.

    Args:
        path:
            Filename of the base geofence .json file.

    Returns:
        A global geofencing dict.  See module header for format
        information.
    """

    with open(path, mode="r", encoding="utf-8") as fp:
        data = json.load(fp)
    for label, rules in data.items():
        if label.endswith(";"):
            raise ValueError(
                "Base geofence should provide only species-level rules. "
                f"Found higher taxa rule with the label: `{label}`"
            )
        if (len(rules) != 1) or (next(iter(rules)) != "allow"):
            raise ValueError("Only 'allow' rules are accepted in base geofence.")
    return data


def fix_geofence_base(
    geofence_base: dict[str, dict], fixes_path: StrPath
) -> dict[str, dict]:
    """Applies the changes specified in a geofence fixes .csv file
    to the base global geofencing dict, returning an updated global geofencing
    dict.

    Args:
        geofence_base:
            A global geofencing dict, probably loaded via load_geofence_base
        fixes_path:
            Filename of the .csv file defining modifications to the geofencing
            dict.

    Returns:
        An updated global geofencing dict.
    """

    geofence = copy.deepcopy(geofence_base)

    fixes = pd.read_csv(fixes_path, keep_default_na=False, comment="#")
    for idx, fix in fixes.iterrows():
        label = fix["species"].lower()
        label_parts = label.split(";")
        if len(label_parts) != 5:
            raise ValueError("Fixes should always use five-token taxon strings")
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


def propagate_rules(geofence: dict[str, dict]) -> dict[str, dict]:
    """Propagates allow rules up the taxonomy tree, and block rules down the taxonomic tree.
    If species X is allowed in country Y, all taxonomic parents of X also need to be allowed
    in Y; if species A is blocked in country B, all taxonomic children of A need to be blocked in B.

    Args:
        geofence: global geofencing dict.  See module header for format information.

    Returns:
        Modified global geofencing dict.
    """

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
            # regional "allow" rules become country-wide "allow" rules at genus level
            # and above.
            if "allow" in rule:
                for country in rule["allow"]:
                    if country not in new_geofence[new_label]["allow"]:
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
    validate_geofence(geofence_base)

    geofence_release = fix_geofence_base(geofence_base, _FIXES.value)
    geofence_release = propagate_rules(geofence_release)
    if _TRIM.value:
        logging.info(
            "Trimming to labels (and their corresponding higher taxa) from `%s`.",
            _TRIM.value,
        )
        geofence_release = trim_to_supported_labels(geofence_release, _TRIM.value)
    else:
        logging.info("No trimming was performed.")
    validate_geofence(geofence_release)

    save_geofence(geofence_release, _OUTPUT.value)


if __name__ == "__main__":
    app.run(main)
