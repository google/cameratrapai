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

This module also includes tools for building taxonomy_release.txt from the labels
file (e.g. always_crop_99710272_22x8_v12_epoch_00148.labels.txt).

A geofencing .json file defines a "global geofencing dict".

Keys in a global geofencing dict are five-token SpeciesNet taxonomy strings,
for example:

aves;accipitriformes;accipitridae;accipiter;rhodogaster

Each item in a global geofencing dict is a "taxon geofencing dict", containing
"allow" and/or "block" rules for that taxon.  The only valid keys in a taxon
geofencing dict are "allow" and "block".  In the base geofence file, only "allow"
rules are valid.

"allow" or block" are mapped a to "regional rules dict".  Each key in a regional
rules dict is a three-letter country code, and the corresponding value is a list
of regional rules within that country.  Currently that list is empty for all country
codes other than "USA".

Examples:

This taxon would be allowed in RUS, SGP, THA, TWN, and VNM.  In the USA,
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
  may later get blocked.

"""

from collections import defaultdict
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Union
import uuid

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import requests
from tqdm import tqdm

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
    "data/model_package/always_crop_99710272_22x8_v12_epoch_00148.labels.txt",
    "Path to the labels supported by the model (TXT).",
)
_TAXONOMY = flags.DEFINE_string(
    "taxonomy",
    "data/model_package/taxonomy_release.txt",
    "Path to the full taxonomy file (model categories and parents) (TXT).",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    "data/model_package/geofence_release.json",
    "Output path for writing the geofence release (JSON).",
)

# Handy type alias.
StrPath = Union[str, Path]

# Constants used for downloading the Wildlife Insights taxonomy when we need to generate
# taxonomy_release.txt (which requires GUIDs for parent taxa of the taxa that appear
# in the labels file).
wildlife_insights_page_size = 30000
wildlife_insights_taxonomy_url = (
    "https://api.wildlifeinsights.org/api/v1/taxonomy/taxonomies-all?"
    "fields=class,order,family,genus,species,authority,taxonomyType,uniqueIdentifier,"
    f"commonNameEnglish&page[size]={wildlife_insights_page_size}"
)

# These are the only non-taxonomic strings we expect to see in the labels file and/or
# the taxonomy file.
vehicle_string = "e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;vehicle"
blank_string = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
animal_string = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"

# It's not ideal to hard-code these here, but because we have this check in place now, this
# list is not expected to grow.  Also see:
#
# https://github.com/google/cameratrapai/issues/52
known_duplicate_five_token_strings = set()
known_duplicate_five_token_strings.add(
    "aves;passeriformes;muscicapidae;copsychus;malabaricus"
)
known_duplicate_five_token_strings.add("mammalia;artiodactyla;;;")
known_duplicate_five_token_strings.add("aves;pelecaniformes;ardeidae;ardea;alba")

# We read parent taxa of the SpeciesNet categories from the Wildlife Insights taxonomy.
# This is a live taxonomy, so we allow a few hard-coded replacements to help match
# taxonomic entities that are in transition.
taxonomic_replacements = {}
taxonomic_replacements["cetartiodactyla"] = "artiodactyla"

# 50 US state codes (no DC, no US territories).  Used by validate_fixes_file
# to detect the case where a rule has been enumerated for every state.
us_state_codes = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    }
)


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
    """Validates a five-token taxon string.  Errors if invalid
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


def _is_descendant_taxon(child: str, parent: str) -> bool:
    """Returns True iff [child] is a strict taxonomic descendant of [parent]
    in five-token semicolon-delimited taxonomy strings.  Both inputs must
    already be five-token strings; both must be left-anchored (no empty
    tokens before a non-empty one), which is enforced elsewhere.

    Equality is NOT a descendant relationship.
    """

    child_parts = child.split(";")
    parent_parts = parent.split(";")
    if len(child_parts) != 5 or len(parent_parts) != 5:
        return False

    parent_depth = 0
    for p in parent_parts:
        if p:
            parent_depth += 1
        else:
            break
    child_depth = 0
    for p in child_parts:
        if p:
            child_depth += 1
        else:
            break

    if child_depth <= parent_depth:
        return False
    return child_parts[:parent_depth] == parent_parts[:parent_depth]


def validate_fixes_file(fixes_path: StrPath, taxonomy_path: StrPath) -> None:
    """Validates the structural integrity and internal consistency of a
    geofence fixes CSV before any of its rules are applied to the base
    geofence.

    This is independent of validating the resulting geofence: this function
    only looks at the rows in the fixes file.  The following checks are
    enforced:

      * Every non-comment, non-blank row has exactly four columns.
      * Each taxon string is a valid five-token string and appears in
        the release taxonomy.
      * Each rule type is "allow" or "block".
      * Country codes are three characters; admin1 codes, when present,
        are two characters.
      * No two rows are literally identical.
      * No row directly conflicts with another -- specifically, no allow
        row is "covered" by a block row on the same taxon or any ancestor
        taxon at a wider-or-equal scope in the same country.  (Covered
        means: block has empty admin1, or its admin1 matches the allow's.)
      * No (taxon, rule_type, USA) group enumerates all 50 US states;
        the user should collapse those rows into a single country-wide
        rule instead.

    Fails with an AssertionError on the first violation found.
    """

    expected_header = ["species", "rule", "country_code", "admin1_region_code"]

    with open(fixes_path, "r", encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))

    # Drop blank rows and "#"-comment rows; remember 1-based source line numbers
    # so error messages can point at the offending line.
    data_rows: list[tuple[int, list[str]]] = []
    for line_idx, row in enumerate(rows, start=1):
        if not row or all(cell == "" for cell in row):
            continue
        if row[0].startswith("#"):
            continue
        data_rows.append((line_idx, row))

    assert data_rows, f"Fixes CSV `{fixes_path}` contains no data rows"

    header_line, header = data_rows[0]
    assert header == expected_header, (
        f"Fixes CSV header at line {header_line} is {header}, expected "
        f"{expected_header}"
    )

    valid_five_token_taxa = set(validate_release_taxonomy(taxonomy_path))

    # Validate each data row and collect the rules for cross-row checks.
    rules: list[tuple[int, str, str, str, str]] = []
    for line_num, row in data_rows[1:]:
        assert len(row) == 4, f"Line {line_num}: row has {len(row)} columns, expected 4"
        label = row[0].lower()
        rule_type = row[1].lower()
        country = row[2]
        state = row[3]

        label_parts = label.split(";")
        assert len(label_parts) == 5, (
            f"Line {line_num}: taxon `{label}` does not have 5 "
            f"semicolon-separated tokens"
        )

        # Catches taxa with empty intermediate tokens (raises ValueError).
        _validate_taxon_string(label)

        assert (
            label in valid_five_token_taxa
        ), f"Line {line_num}: taxon `{label}` is not in the taxonomy"

        assert rule_type in (
            "allow",
            "block",
        ), f"Line {line_num}: rule type `{rule_type}` must be `allow` or `block`"

        assert (
            len(country) == 3
        ), f"Line {line_num}: country code `{country}` must be 3 characters"

        if state:
            assert len(state) == 2, (
                f"Line {line_num}: admin1 region code `{state}` must be "
                f"2 characters"
            )

        rules.append((line_num, label, rule_type, country, state))

    # No literally identical rules.
    seen: dict[tuple[str, str, str, str], int] = {}
    for line_num, label, rule_type, country, state in rules:
        key = (label, rule_type, country, state)
        assert key not in seen, (
            f"Line {line_num}: rule `{label},{rule_type},{country},{state}` "
            f"is identical to the rule on line {seen[key]}"
        )
        seen[key] = line_num

    # Direct conflicts: an allow row is "covered" by a block row whose
    # taxon is the same or an ancestor in the same country, where the
    # block's scope is wider-or-equal to the allow's:
    #   - block admin1 is empty (country-wide), OR
    #   - block admin1 equals the allow's admin1.
    blocks = [(l, t, c, s) for l, t, r, c, s in rules if r == "block"]
    allows = [(l, t, c, s) for l, t, r, c, s in rules if r == "allow"]

    for a_line, a_taxon, a_country, a_state in allows:
        for b_line, b_taxon, b_country, b_state in blocks:
            if a_country != b_country:
                continue
            if a_taxon != b_taxon and not _is_descendant_taxon(a_taxon, b_taxon):
                continue
            if b_state == "" or b_state == a_state:
                assert False, (
                    f"Conflicting rules in fixes CSV: allow on line "
                    f"{a_line} (`{a_taxon},allow,{a_country},{a_state}`) "
                    f"is covered by block on line {b_line} "
                    f"(`{b_taxon},block,{b_country},{b_state}`)"
                )

    # No (taxon, rule_type, USA) group should enumerate all 50 US states.
    #
    # This check helps keep geofence_fixes.csv concise: a user who lists
    # every state explicitly should replace those rows with a single
    # country-wide rule.  DC and US territories are intentionally NOT
    # counted toward the 50, so a rule that covers every state but
    # deliberately omits one of those would still pass.  Consequently
    # this check prevents some combinations of rules involving DC or
    # territories: if we ever need to apply a rule to all 50 states
    # while exempting DC or a specific territory for a single taxon, we
    # would have to relax this check.  That combination is not currently
    # expected.
    by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    first_line_in_group: dict[tuple[str, str], int] = {}
    for line_num, label, rule_type, country, state in rules:
        if country != "USA" or not state:
            continue
        group = (label, rule_type)
        by_group[group].add(state)
        first_line_in_group.setdefault(group, line_num)

    for group, states in by_group.items():
        if us_state_codes.issubset(states):
            label, rule_type = group
            example_line = first_line_in_group[group]
            assert False, (
                f"Rule `{label},{rule_type},USA,*` is enumerated across all "
                f"50 US states (starting near line {example_line}); replace "
                f"those rows with a single country-wide rule: "
                f"`{label},{rule_type},USA,`"
            )

    # Warn (don't fail) when the CSV contains both a country-wide row
    # and one or more state-specific rows for the same (taxon, rule_type,
    # country).  After the country-wide widening fix, the country-wide
    # row replaces any state list and subsequent state-specific rows
    # become no-ops, so the combination is redundant but not incorrect.
    rows_by_group: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for line_num, label, rule_type, country, state in rules:
        rows_by_group[(label, rule_type, country)].append((line_num, state))

    for (label, rule_type, country), entries in rows_by_group.items():
        cw_lines = [ln for ln, s in entries if not s]
        ss_entries = [(ln, s) for ln, s in entries if s]
        if cw_lines and ss_entries:
            print(
                f"WARNING: fixes CSV has a country-wide row "
                f"`{label},{rule_type},{country},` (line {cw_lines[0]}) "
                f"and {len(ss_entries)} state-specific row(s) for the "
                f"same taxon/rule/country (lines "
                f"{[ln for ln, _ in ss_entries]}, states "
                f"{[s for _, s in ss_entries]}); the country-wide row "
                f"supersedes the state-specific ones at apply time"
            )


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
                # This taxon might also be blocked, which supersedes the allow rule.  If
                # it's actually blocked, don't confirm that its parents are allowed.
                if not _taxon_allowed_in_region(
                    label=taxon,
                    country=country,
                    admin1_region=region,
                    geofence_map=geofence,
                ):
                    continue

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
    geofence_base: dict[str, dict], fixes_path: StrPath, taxonomy_path: StrPath
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
        taxonomy_path:
            Filename of the .txt file containing valid taxonomy entries
            (typically taxonomy_release.txt).

    Returns:
        An updated global geofencing dict.
    """

    # Confirm the fixes CSV is structurally sound and internally consistent
    # before we apply any of its rules.  See validate_fixes_file for the
    # specific checks; failures raise AssertionError.
    validate_fixes_file(fixes_path, taxonomy_path)

    geofence = copy.deepcopy(geofence_base)

    # Read the list of valid taxa
    valid_five_token_taxa = set(validate_release_taxonomy(taxonomy_path))

    fixes = pd.read_csv(fixes_path, keep_default_na=False, comment="#")
    for idx, fix in fixes.iterrows():
        label = fix["species"].lower()
        assert label in valid_five_token_taxa, f"Invalid taxon in fixes file: {label}"
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

        # Snapshot the prior allow/block lists for this (taxon, country)
        # so we can emit warnings when a country-wide row overwrites a
        # state-specific list, when a state-specific row is a no-op
        # against an existing country-wide list, or when a country-wide
        # row coexists with a state-specific rule of the opposite type.
        prior_taxon = geofence.get(label) or {}
        prior_allow = (prior_taxon.get("allow") or {}).get(country)
        prior_block = (prior_taxon.get("block") or {}).get(country)

        if rule == "allow":
            # Mirror the block branch's entry/key creation.  Previously
            # these two cases were silently skipped with "already
            # allowed" -- but per geofence semantics, the first allow
            # rule for a taxon establishes a positive allow list and
            # blocks everywhere else, so silently dropping the row
            # loses the user's intent.  See the same idiom below in the
            # block branch.
            if label not in geofence:
                print(
                    f"WARNING: allow row `{label},allow,{country},{state}` "
                    f"introduces the first entry for this taxon; the taxon "
                    f"was previously allowed everywhere (no entry) and is "
                    f"now restricted to the locations specified by this "
                    f"and any other allow rows for it"
                )
                geofence[label] = {"allow": {country: [state] if state else []}}
            if "allow" not in geofence[label]:
                print(
                    f"WARNING: allow row `{label},allow,{country},{state}` "
                    f"adds the first allow entry for a taxon that previously "
                    f"had only block rules; the taxon is now restricted to "
                    f"the locations specified by this and any other allow "
                    f"rows for it"
                )
                geofence[label]["allow"] = {country: [state] if state else []}
            if not state:
                # Country-wide allow: replace any existing state list with
                # [] (country-wide).  Previously this used .get(country, [])
                # which silently failed to widen a state-restricted entry.
                if prior_allow:
                    print(
                        f"WARNING: country-wide row `{label},allow,"
                        f"{country},` overwrites existing state-restricted "
                        f"allow.{country} = {prior_allow}"
                    )
                if prior_block:
                    print(
                        f"WARNING: country-wide row `{label},allow,"
                        f"{country},` coexists with state-restricted "
                        f"block.{country} = {prior_block} (i.e. allowed "
                        f"everywhere in {country} except those states); "
                        f"verify this is intentional"
                    )
                geofence[label]["allow"][country] = []
            else:
                curr_country_rule = geofence[label]["allow"].get(country)
                if curr_country_rule is None:  # missing country rule
                    geofence[label]["allow"][country] = [state]
                else:
                    if not curr_country_rule:  # an empty list
                        print(
                            f"WARNING: state-specific row `{label},allow,"
                            f"{country},{state}` is a no-op; allow.{country}"
                            f" = [] (country-wide) is already in effect"
                        )
                        continue
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
                # Country-wide block: replace any existing state list with
                # [].  Same widening fix as for the allow branch.
                if prior_block:
                    print(
                        f"WARNING: country-wide row `{label},block,"
                        f"{country},` overwrites existing state-restricted "
                        f"block.{country} = {prior_block}"
                    )
                if prior_allow:
                    print(
                        f"WARNING: country-wide row `{label},block,"
                        f"{country},` is being applied while state-"
                        f"restricted allow.{country} = {prior_allow} "
                        f"exists; block wins, so those allow entries "
                        f"become ineffective"
                    )
                geofence[label]["block"][country] = []
            else:
                curr_country_rule = geofence[label]["block"].get(country)
                if curr_country_rule is None:  # missing country rule
                    geofence[label]["block"][country] = [state]
                else:
                    if not curr_country_rule:  # an empty list
                        print(
                            f"WARNING: state-specific row `{label},block,"
                            f"{country},{state}` is a no-op; block.{country}"
                            f" = [] (country-wide) is already in effect"
                        )
                        continue
                    else:  # not an empty list
                        geofence[label]["block"][country] = sorted(
                            set(curr_country_rule) | {state}
                        )

    return geofence


def propagate_rules(geofence: dict[str, dict], labels_path: str) -> dict[str, dict]:
    """Propagates allow rules up the taxonomy tree, and block rules down the taxonomic tree.
    If species X is allowed in country Y, all taxonomic parents of X also need to be allowed
    in Y; if species A is blocked in country B, all taxonomic children of A need to be blocked
    in B.

    Args:
        geofence: global geofencing dict.  See module header for format information.
        labels_path: text file containing labels in seven-token format.

    Returns:
        Global geofencing dict.  Does not modify "geofence" in place.
    """

    new_geofence = {}

    # Propagate allow rules up
    for label, rule in geofence.items():

        label_parts = label.split(";")

        # Keep original rule.
        new_geofence[label] = rule

        # Propagate to higher taxa.
        for taxa_level_end in range(1, 5):
            new_label = ";".join(label_parts[:taxa_level_end]) + (
                ";" * (5 - taxa_level_end)
            )

            # By convention, create an allow rule for all parent taxa, even if
            # we're about adding a block rule for the child taxon.
            if new_label not in new_geofence:
                new_geofence[new_label] = {"allow": {}}

            # Country-wide "allow" rules at species level get propagated directly, but
            # regional "allow" rules become country-wide "allow" rules at genus level
            # and above.
            if "allow" in rule:
                for country in rule["allow"]:
                    if country not in new_geofence[new_label]["allow"]:
                        new_geofence[new_label]["allow"][country] = []

    # Create a list of block rules we need to propagate down the tree.
    #
    # "block_rules" will be formatted like a global geofence dict,
    # but it will only contain rules of type "block".
    block_rules = {}

    for label, rule in geofence.items():

        if "block" not in rule:
            continue

        for country in rule["block"]:
            country_rule = rule["block"][country]
            if label not in block_rules:
                block_rules[label] = {}
                block_rules[label]["block"] = {}
            assert country not in block_rules[label]["block"]
            block_rules[label]["block"][country] = country_rule

    # Read the label list
    labels = _read_label_list(labels_path)

    # We're *probably* going to trim this geofencing dict to the set
    # of allowable labels later, so it won't really matter whether we
    # also propagate block rules through labels that exist in the base
    # geofence but aren't valid SpeciesNet classes.  However, if we want
    # this function to return a valid geofencing dict, we also need to
    # propagate block rules over all the labels that exist in the geofence.
    labels_in_geofence = set(geofence.keys())

    labels = set(labels) | labels_in_geofence

    # We also need to add all parent taxa of all labels
    all_parent_taxa = set()
    for label in labels:
        parent_taxa = _generate_parent_taxon_strings(label)
        for taxon in parent_taxa:
            all_parent_taxa.add(taxon)

    for taxon in all_parent_taxa:
        labels.add(taxon)

    labels = sorted(list(labels))

    # Propagate block rules down the taxonomy.
    #
    # "new_block_rules" will be formatted like a global geofence dict,
    # but it will only include rules of type "block".
    new_block_rules = {}

    def _merge_country_rule_lists(source, target):
        """Add rules from source into target.

        An empty list ([]) at a country key means "the whole country"
        and is strictly more restrictive than any state-restricted list,
        so a country-wide contribution from EITHER source or target
        always wins.

        Args:
            source: a regional rules dict, e.g. "{'RUS':[],'USA':['AZ']}"
            target: a regional rules dict, modified in place
        """
        assert isinstance(source, dict)
        assert isinstance(target, dict)
        for country in source:
            assert (isinstance(country, str)) and (len(country) == 3)
            s_list = source[country]
            assert isinstance(s_list, list)
            if country not in target:
                target[country] = list(s_list)
                continue
            t_list = target[country]
            assert isinstance(t_list, list)
            # Country-wide on either side wins.
            if not s_list or not t_list:
                target[country] = []
                continue
            # Both are state-restricted lists: union them.
            for region in s_list:
                if region not in t_list:
                    t_list.append(region)

    # "label" is a five-token taxon string
    for label in labels:

        # taxon_with_block_rule is a five-token taxon string
        for taxon_with_block_rule in block_rules.keys():

            # Don't add a new copy of the same block rule
            if label == taxon_with_block_rule:
                continue

            # "taxon_prefix" is a semicolon-delimited list, but it may have
            # anywhere from one to five tokens.  We append the ";" back
            # so the prefix match below is anchored to a token boundary
            # (otherwise e.g. genus "mus" could spuriously match a species
            # whose name happens to contain the substring "mus").
            taxon_prefix = taxon_with_block_rule.rstrip(";") + ";"

            # If "taxon_prefix" is the start of "label", that means "label"
            # is a taxonomic child of "taxon_with_block_rule".
            if label.startswith(taxon_prefix):

                print(
                    "Adding block rule to {} because of parent {}".format(
                        label, taxon_with_block_rule
                    )
                )

                if label not in new_block_rules:
                    new_block_rules[label] = {}
                    new_block_rules[label]["block"] = {}

                _merge_country_rule_lists(
                    block_rules[taxon_with_block_rule]["block"],
                    new_block_rules[label]["block"],
                )

    print(f"Adding {len(new_block_rules)} new block rules during propagation")

    for label in new_block_rules:
        # Make sure new_geofence[label] has a "block" dict to merge into,
        # without clobbering an existing "allow" entry.  The previous
        # version reassigned the whole entry to {"block": {}} when the
        # label had only an "allow", silently dropping the allow data.
        if label not in new_geofence:
            new_geofence[label] = {"block": {}}
        elif "block" not in new_geofence[label]:
            new_geofence[label]["block"] = {}
        _merge_country_rule_lists(
            new_block_rules[label]["block"], new_geofence[label]["block"]
        )

    return new_geofence


def _read_label_list(labels_path: StrPath) -> set[str]:
    """Create a list of five-token labels from a file of seven-token labels.
    The resulting list includes all taxa in the input list, as well as all
    parents of those taxa.

    Args:
        geofence: global geofencing dict.  See module header for format information.
        labels_path: text file containing labels in seven-token format.

    Returns:
        Set of five-token labels.
    """

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

    return labels


def download_wildlife_insights_taxonomy(
    output_path: Optional[StrPath], overwrite: bool = False
) -> StrPath:
    """Download the Wildlife Insights taxonomy file from the taxonomy
    API to a local .json file.

    Args:
        output_path:
            Path to write the taxonomy to; defaults to a file in system temp space.
        overwrite:
            Overwrite the taxonomy file if it exists.

    Returns:
        Path where the taxonomy file was downloaded (or already existed).
    """

    if not output_path:
        temp_dir = tempfile.gettempdir()
        wi_temp_path = os.path.join(temp_dir, "speciesnet")
        os.makedirs(wi_temp_path, exist_ok=True)
        output_path = os.path.join(wi_temp_path, "wi_taxonomy.json")

    if os.path.isfile(output_path) and (not overwrite):
        print(f"Bypassing download of existing file {output_path}")
        return output_path

    response = requests.get(wildlife_insights_taxonomy_url, stream=True, timeout=600)
    response.raise_for_status()
    with open(output_path, mode="wb") as fp:
        for chunk in response.iter_content(chunk_size=8192):
            fp.write(chunk)

    with open(output_path, mode="r", encoding="utf-8") as fp:
        wildlife_insights_taxonomy = json.load(fp)

    # We haven't implemented paging, make sure that's not an issue
    if wildlife_insights_taxonomy["meta"]["totalItems"] > wildlife_insights_page_size:
        raise NotImplementedError("Paging not implemented yet for WI taxonomy download")

    return output_path


def validate_release_taxonomy(taxonomy_path: StrPath):
    """Verify that taxonomy_path (typically taxonomy_release.txt) represents
    a valid taxonomy file.

    Specifically, verify that:

    * All lines are well-formatted seven-token taxonomy strings
    * Only expected duplicated taxa exist
    * Only expected non-taxonomic strings exist

    Args:
        taxonomy_path: .csv file to alidate, typically taxonomy_releaase.txt

    Returns:
        list of valid five-token taxon strings
    """

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [s.strip() for s in lines]

    known_non_taxonomic_strings = set()
    known_non_taxonomic_strings.add(blank_string)
    known_non_taxonomic_strings.add(vehicle_string)
    known_non_taxonomic_strings.add(animal_string)

    five_token_taxa = set()
    for line in lines:
        # Verify that this line is well-formatted
        assert line.islower()
        tokens = line.split(";")
        assert len(tokens) == 7
        five_token_taxon_string = ";".join(tokens[1:-1])
        # If this is a non-taxonomic string, make sure it's expected
        if len(five_token_taxon_string.replace(";", "")) == 0:
            assert line in known_non_taxonomic_strings
            continue
        # If this is a duplicate, make sure it's expected
        if five_token_taxon_string in five_token_taxa:
            assert five_token_taxon_string in known_duplicate_five_token_strings
        five_token_taxa.add(five_token_taxon_string)

    return sorted(list(five_token_taxa))


def generate_release_taxonomy_from_label_list(
    labels_path: StrPath, output_path: StrPath
):
    """Generates taxonomy_release.txt (the list of taxa that might
    be produced by the ensemble) from the labels.txt file (the list of
    model classes).  The taxonomy list is just the set of categories in
    the labels file, plus their parent taxa, which have to be retrieved
    from the public WI taxonomy.

    This is not related to geofencing, but it is part of the release process,
    which is why it's in this module.

    Args:
        labels_path: text file containing labels in seven-token format.
        output_path: the release taxonomy file, containing labels in seven-token
            format.
    """

    ## Download the WI taxonomy

    taxonomy_path = download_wildlife_insights_taxonomy(output_path=None)

    with open(taxonomy_path, mode="r", encoding="utf-8") as fp:
        wi_taxonomy = json.load(fp)

    # wi_taxononmy['data'] is a list of items that look like:
    """
        {'id': 2000003,
        'class': 'Mammalia',
        'order': 'Rodentia',
        'family': 'Abrocomidae',
        'genus': 'Abrocoma',
        'species': 'bennettii',
        'authority': 'Waterhouse, 1837',
        'commonNameEnglish': "Bennett's Chinchilla Rat",
        'taxonomyType': 'biological',
        'uniqueIdentifier': '7a6c93a5-bdf7-4182-82f9-7a67d23f7fe1'}
    """

    ## Map five-token taxon strings to entities in the WI taxonomy

    # For example:
    #
    # 'mammalia;chiroptera;emballonuridae;diclidurus;ingens': {'id': 2001323 ...}
    five_token_taxon_string_to_wi_taxon_info = {}

    for item in wi_taxonomy["data"]:
        fields = []
        levels = ["class", "order", "family", "genus", "species"]
        for level in levels:
            if item[level] is None:
                fields.append("")
            else:
                s = item[level].lower().strip()
                if s in taxonomic_replacements:
                    s = taxonomic_replacements[s]
                fields.append(s)
        # If there is no value for the class, this is not an animal, it's
        # something like "fire" or "no cv result"
        if len(fields[0]) == 0:
            print(f"Skipping non-animal taxon {item['commonNameEnglish']}")
            continue

        # This is a five-token taxon string
        taxon_string = ";".join(fields)

        # We only want to map each five-token taxon string to a single item,
        # so if this five-token taxon string has already appeared, choose
        # the more recently-updated item.
        if taxon_string in five_token_taxon_string_to_wi_taxon_info:
            old_item = five_token_taxon_string_to_wi_taxon_info[taxon_string]
            # The field "updatedAt" is a datetime string, e.g.
            # "2025-11-07T22:12:45.929Z".  Between "item" and
            # "old_item", set five_token_taxon_string_to_wi_taxon_info[taxon_string]
            # to the one with the more recent value for "updatedAt".
            if item["updatedAt"] > old_item["updatedAt"]:
                print(
                    "Warning: replacing {} with {} for {}".format(
                        old_item["commonNameEnglish"],
                        item["commonNameEnglish"],
                        taxon_string,
                    )
                )
                five_token_taxon_string_to_wi_taxon_info[taxon_string] = item
        else:
            five_token_taxon_string_to_wi_taxon_info[taxon_string] = item

    ## Read the labels file

    # The labels file is a list of seven-token taxon strings, e.g.:
    #
    # 32809fe1-a183-45a0-9dcf-f9d900193a6e;aves;coraciiformes;momotidae;momotus;mexicanus;russet-crowned motmot
    with open(labels_path, mode="r", encoding="utf-8") as fp:
        seven_token_taxon_strings_in_label_file = [
            line.strip() for line in fp.readlines()
        ]

    # The output file (typically taxonomy_release.txt) need to contain the parents of everything in the
    # label file, so, map the parents of every taxon in the labels file into the WI taxonomy.

    non_taxonomic_strings_in_labels_file = set()

    # This maps the five-token taxon strings in the labels file to seven-token taxon strings, e.g.:
    #
    # {'aves;coraciiformes;momotidae;momotus;mexicanus':
    # '32809fe1-a183-45a0-9dcf-f9d900193a6e;aves;coraciiformes;momotidae;momotus;mexicanus;russet-crowned motmot'}
    five_token_taxon_strings_in_labels_file = {}
    for seven_token_taxon_string in seven_token_taxon_strings_in_label_file:
        tokens = seven_token_taxon_string.split(";")
        assert len(tokens) == 7
        taxon_string = ";".join(tokens[1:6])
        if len(taxon_string.replace(";", "")) == 0:
            print(f"Ignoring non-taxonomic string {seven_token_taxon_string}\n")
            non_taxonomic_strings_in_labels_file.add(seven_token_taxon_string)
            continue
        if taxon_string in five_token_taxon_strings_in_labels_file:
            print(
                f"Warning: taxon {taxon_string} appears multiple times in the labels file:"
            )
            print(f"{seven_token_taxon_string}")
            print(f"{five_token_taxon_strings_in_labels_file[taxon_string]}")
            print("")
        five_token_taxon_strings_in_labels_file[taxon_string] = seven_token_taxon_string

    ## Find taxa in the labels file whose parents aren't also in the labels file
    #
    # ...and fill in those parents from the WI taxonomy.

    # Keys are the taxon strings we'll add from the WI taxonomy, representing all
    # the parents of the taxa in the labels file that were not already in the labels
    # file.  Values are the corresponding seven-token strings.
    five_token_parent_identifier_strings = {}

    for taxon_string in tqdm(five_token_taxon_strings_in_labels_file):
        # [taxon_string] is a five-token taxon string representing one of SpeciesNet's
        # output categories.  Generate all of its parents, and if they're not also a
        # SpeciesNet category, find the corresponding taxon information from the WI
        # taxonomy.
        parent_taxa = _generate_parent_taxon_strings(taxon_string)
        for parent_taxon_string in parent_taxa:

            # If this parent taxon is itself a SpeciesNet output category, skip it
            if parent_taxon_string in five_token_taxon_strings_in_labels_file:
                continue

            # If this parent taxon was already added because it had another child,
            # skip it
            if parent_taxon_string in five_token_parent_identifier_strings:
                continue

            # A small number of taxa in the Wildlife Insights taxonomy are
            # "missing", in the sense that x's children are present, but x is
            # not.  We still include these in the SpeciesNet taxonomy file,
            # but we make up a GUID dynamically.
            if parent_taxon_string not in five_token_taxon_string_to_wi_taxon_info:
                print(
                    f"Warning: expected taxon {parent_taxon_string} not in WI taxonomy"
                )
                guid = str(uuid.uuid4())
                taxon_info = {}
                taxon_info["uniqueIdentifier"] = guid
                taxon_info["commonNameEnglish"] = ""
            else:
                taxon_info = five_token_taxon_string_to_wi_taxon_info[
                    parent_taxon_string
                ]
            guid = taxon_info["uniqueIdentifier"]
            if taxon_info["commonNameEnglish"] is not None:
                common = taxon_info["commonNameEnglish"].lower().strip()
            else:
                common = ""
            identifier_string = guid + ";" + parent_taxon_string + ";" + common
            five_token_parent_identifier_strings[parent_taxon_string] = (
                identifier_string
            )

    seven_token_parent_identifier_strings = (
        five_token_parent_identifier_strings.values()
    )
    print(
        "Adding {} new parent identifier strings".format(
            len(seven_token_parent_identifier_strings)
        )
    )

    # Merge the taxon strings from the labels file with the parent taxon strings
    # that we need to add
    merged_seven_token_taxon_strings = set(
        seven_token_taxon_strings_in_label_file
    ) | set(seven_token_parent_identifier_strings)

    # Verify that none of the strings we're adding were already in the labels file
    assert len(merged_seven_token_taxon_strings) == len(
        seven_token_taxon_strings_in_label_file
    ) + len(seven_token_parent_identifier_strings)

    ## Make sure that five-token strings are unique, except for known duplicates
    #
    # This will not include non-taxonomic strings ('vehicle','blank','animal').
    five_token_taxon_to_seven_token_taxon = {}

    for i_taxon, seven_token_taxon_string in enumerate(
        merged_seven_token_taxon_strings
    ):
        tokens = seven_token_taxon_string.split(";")
        assert len(tokens) == 7
        five_token_taxon_string = ";".join(tokens[1:-1])
        # Skip non-taxonomic entities
        if len(five_token_taxon_string.replace(";", "")) == 0:
            print(f"Skipping non-taxonomic category {seven_token_taxon_string}")
            continue
        if (five_token_taxon_string in five_token_taxon_to_seven_token_taxon) and (
            five_token_taxon_string not in known_duplicate_five_token_strings
        ):
            raise ValueError(
                "Five-token taxon string {} is mapped to multiple seven-token strings:\n{}\n{}\n".format(
                    five_token_taxon_string,
                    seven_token_taxon_string,
                    five_token_taxon_to_seven_token_taxon[five_token_taxon_string],
                )
            )
        five_token_taxon_to_seven_token_taxon[five_token_taxon_string] = (
            seven_token_taxon_string
        )

    assert len(set(five_token_taxon_to_seven_token_taxon.values())) == len(
        five_token_taxon_to_seven_token_taxon
    )

    ## Add non-taxonomic strings that belong in the output taxonomy file

    # The WI taxonomy includes a variety of other non-taxonomic labels; these
    # are the only ones we want to include in the output labels file.
    assert vehicle_string in non_taxonomic_strings_in_labels_file
    assert blank_string in non_taxonomic_strings_in_labels_file
    non_taxonomic_strings_in_output_taxonomy = set(non_taxonomic_strings_in_labels_file)
    non_taxonomic_strings_in_output_taxonomy.add(animal_string)

    output_taxa = list(five_token_taxon_to_seven_token_taxon.values()) + list(
        non_taxonomic_strings_in_output_taxonomy
    )
    assert len(output_taxa) == len(five_token_taxon_to_seven_token_taxon) + len(
        non_taxonomic_strings_in_output_taxonomy
    )

    ## Write taxonomy file

    # Sort taxa by name, rather than by GUID.
    output_taxa_sorted = sorted(output_taxa, key=lambda x: x.split(";")[1:])

    print(f"Writing taxonomy file to {output_path}")
    with open(output_path, mode="w", encoding="utf-8") as f:
        for s in output_taxa_sorted:
            f.write(s + "\n")

    validate_release_taxonomy(output_path)


def trim_to_supported_labels(
    geofence: dict[str, dict], labels_path: str
) -> dict[str, dict]:
    """Trim the geofence rules to labels that are used in the labels file.

    Args:
        geofence: global geofencing dict.  See module header for format information.
        labels_path: text file containing labels in seven-token format.

    Returns:
        Global geofencing dict trimmed to specified labels.  Does not modify
        "geofence" in place.

    """

    labels = _read_label_list(labels_path)
    return {k: v for k, v in geofence.items() if k in labels}


def save_geofence(geofence: dict[str, dict], output_path: StrPath) -> None:

    with open(output_path, mode="w", encoding="utf-8") as fp:
        json.dump(geofence, fp, indent=4, sort_keys=True)


def main(argv: list[str]) -> None:
    del argv  # Unused.

    geofence_base = load_geofence_base(_BASE.value)
    validate_geofence(geofence_base)

    geofence_release = fix_geofence_base(
        geofence_base=geofence_base,
        fixes_path=_FIXES.value,
        taxonomy_path=_TAXONOMY.value,
    )
    geofence_release = propagate_rules(
        geofence=geofence_release, labels_path=_TRIM.value
    )
    geofence_release = trim_to_supported_labels(
        geofence=geofence_release, labels_path=_TRIM.value
    )

    validate_geofence(geofence_release)

    save_geofence(geofence_release, _OUTPUT.value)


if __name__ == "__main__":
    app.run(main)
