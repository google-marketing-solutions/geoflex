# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All the individual geo-experimentation methodologies used by GeoFleX."""

from geoflex.methodology import _base
from geoflex.methodology import gbr
from geoflex.methodology import synthetic_controls
from geoflex.methodology import tbr
from geoflex.methodology import tbrmm
from geoflex.methodology import testing_methodology


Methodology = _base.Methodology
register_methodology = _base.register_methodology
get_methodology = _base.get_methodology
list_methodologies = _base.list_methodologies
assign_geos = _base.assign_geos
analyze_experiment = _base.analyze_experiment
design_is_eligible_for_data = _base.design_is_eligible_for_data
is_pseudo_experiment = _base.is_pseudo_experiment
