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

from speciesnet.geolocation import find_admin1_region as admin1


class TestFindAdmin1Region:
    """Tests for the first-level administrative division."""

    def test_missing_country(self) -> None:
        assert admin1(country=None, latitude=None, longitude=None) is None
        assert admin1(country=None, latitude=0, longitude=0) is None
        assert admin1(country=None, admin1_region="AK") is None

    def test_given_admin1_region(self) -> None:
        assert admin1(country="USA", admin1_region="HI") == "HI"
        assert (
            admin1(country="USA", admin1_region="CA", latitude=37, longitude=-119)
            == "CA"
        )
        assert (
            admin1(country="USA", admin1_region="CA", latitude=46.01, longitude=-69.19)
            == "CA"
        )
        assert (
            admin1(country="USA", admin1_region="DC", latitude=46.01, longitude=-69.19)
            == "DC"
        )

    def test_usa_states(self) -> None:
        assert admin1(country="USA", latitude=None, longitude=None) is None
        assert admin1(country="USA", latitude=0, longitude=None) is None
        assert admin1(country="USA", latitude=None, longitude=0) is None
        assert admin1(country="USA", latitude=0, longitude=0) is None
        assert admin1(country="USA", latitude=-23, longitude=133) is None
        assert admin1(country="USA", latitude=45, longitude=0) is None

        assert admin1(country="USA", latitude=65, longitude=-150) == "AK"
        assert admin1(country="USA", latitude=19, longitude=-155) == "HI"
        assert admin1(country="USA", latitude=37, longitude=-119) == "CA"
        assert admin1(country="USA", latitude=46.01, longitude=-69.19) == "ME"
        assert admin1(country="USA", latitude=38.63, longitude=-103.43) == "CO"
        assert admin1(country="USA", latitude=38.92, longitude=-77.01) == "DC"

    def test_usa_territories(self) -> None:
        assert admin1(country="ASM", latitude=-14.23, longitude=-169.46) is None
        assert admin1(country="USA", latitude=-14.23, longitude=-169.46) == "AS"
        assert admin1(country="GUM", latitude=13.42, longitude=144.71) is None
        assert admin1(country="USA", latitude=13.42, longitude=144.71) == "GU"
        assert admin1(country="MNP", latitude=16.34, longitude=145.66) is None
        assert admin1(country="USA", latitude=16.34, longitude=145.66) == "MP"
        assert admin1(country="PRI", latitude=18.22, longitude=-66.67) is None
        assert admin1(country="USA", latitude=18.22, longitude=-66.67) == "PR"
        assert admin1(country="UMI", latitude=19.28, longitude=166.64) is None
        # "UM" is not currently supported by the reverse geocoder.
        # assert admin1(country="USA", latitude=19.28, longitude=166.64) == "UM"
        assert admin1(country="USA", latitude=19.28, longitude=166.64) is None
        assert admin1(country="VIR", latitude=18.35, longitude=-64.71) is None
        assert admin1(country="USA", latitude=18.35, longitude=-64.71) == "VI"

    def test_other_countries(self) -> None:
        assert admin1(country="AUS", latitude=None, longitude=None) is None
        assert admin1(country="AUS", latitude=-23, longitude=133) is None
        assert admin1(country="BRA", latitude=0, longitude=0) is None
        assert admin1(country="BRA", latitude=-8, longitude=-56) is None
