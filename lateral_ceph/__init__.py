"""Lateral-cephalogram landmarks (HRNet, 29 points), ported from
orthosage_python2_api's /predict. Mounted into api.py via `router`."""

from .router import router, load_models, status as router_status  # noqa: F401
