"""All the individual geo-experimentation methodologies used by GeoFleX."""

from geoflex.methodology import _base
from geoflex.methodology import testing_methodology

Methodology = _base.Methodology
register_methodology = _base.register_methodology
get_methodology = _base.get_methodology
list_methodologies = _base.list_methodologies
assign_geos = _base.assign_geos
analyze_experiment = _base.analyze_experiment
design_is_eligible_for_data = _base.design_is_eligible_for_data
