"""Intraoral-photo condition detection (gingivitis, recession, caries, plaque),
ported from BinaModel_project. Mounted into api.py via `router`."""

from .router import router, load_models, status as router_status  # noqa: F401
