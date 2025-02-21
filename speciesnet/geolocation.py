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

"""Geolocation utilities."""

__all__ = [
    "find_admin1_region",
]

from typing import Optional

import reverse_geocoder as rg

US_COUNTRY_CODES_ALPHA_2_TO_ALPHA_3 = {  # ISO 3166-1 Alpha-2 => ISO 3166-1 Alpha-3
    "US": "USA",  # United States of America
    "AS": "ASM",  # American Samoa
    "GU": "GUM",  # Guam
    "MP": "MNP",  # Northern Mariana Islands
    "PR": "PRI",  # Puerto Rico
    "UM": "UMI",  # United States Minor Outlying Islands
    "VI": "VIR",  # United States Virgin Islands
}
US_COUNTRY_CODES_ALPHA_2 = US_COUNTRY_CODES_ALPHA_2_TO_ALPHA_3.keys()
US_COUNTRY_CODES_ALPHA_3 = US_COUNTRY_CODES_ALPHA_2_TO_ALPHA_3.values()

US_ADMIN1_REGION_TO_ABBREVIATION = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    # "District of Columbia" appears with an alternative name in the reverse geocoder.
    "Washington, D.C.": "DC",
}


ADMIN1_REGION_SUPPORTED_COUNTRIES = US_COUNTRY_CODES_ALPHA_3


def find_admin1_region(
    country: Optional[str] = None,
    admin1_region: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Optional[str]:
    """Finds the first-level administrative division for a given (lat, lon) location.

    Skips doing any work if both `country` and `admin1_region` are already specified.

    Currently implemented only for USA and its territories:
        * If the country is "USA" and the (lat, lon) coordinates fall within a USA state
          or USA territory, the admin1 region will be the two-letter country code of
          that state or territory (e.g. "NY" or "PR").
        * If the country is a USA territory country code (e.g. "PRI") and the (lat, lon)
          coordinates fall within that territory, the admin1 region will be `None`.

    Args:
        country:
            Country in ISO 3166-1 alpha-3 format. Optional.
        admin1_region:
            First-level administrative division in ISO 3166-2 format. Optional.
        latitude:
            Float value representing latitude. Optional.
        longitude:
            Float value representing longitude. Optional.

    Returns:
        First-level administrative division in ISO 3166-2 format. If no matching admin1
        region is found (e.g. due to missing country, latitude or longitude, or due to
        country not supported), return `None`.
    """

    if country and admin1_region:
        return admin1_region

    if (
        country not in ADMIN1_REGION_SUPPORTED_COUNTRIES
        or latitude is None
        or longitude is None
    ):
        return None

    if country in US_COUNTRY_CODES_ALPHA_3:
        if country != "USA":
            return None

        result = rg.search((latitude, longitude), mode=1, verbose=False)[0]
        if result["cc"] not in US_COUNTRY_CODES_ALPHA_2:
            return None
        if result["cc"] == "US":
            if (
                "admin1" in result
                and result["admin1"] in US_ADMIN1_REGION_TO_ABBREVIATION
            ):
                return US_ADMIN1_REGION_TO_ABBREVIATION[result["admin1"]]
            else:
                return None
        else:
            return result["cc"]

    return None
