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

"""The SpeciesNet package."""

from speciesnet.classifier import *
from speciesnet.constants import *
from speciesnet.detector import *
from speciesnet.display import *
from speciesnet.ensemble import *
from speciesnet.geolocation import *
from speciesnet.multiprocessing import *
from speciesnet.utils import *

DEFAULT_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"

# This represents the model URLs that will be tested via pytest;
# this does not indicate that only these models will work with
# the speciesnet package.
SUPPORTED_MODELS = [
    "kaggle:google/speciesnet/pyTorch/v4.0.2a/1",
    "kaggle:google/speciesnet/pyTorch/v4.0.2b/1",
]
