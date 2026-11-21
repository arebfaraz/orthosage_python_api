"""Intraoral photo view classification + FDI tooth segmentation (+ gingivitis
grading on frontal views), ported from orthosage_occlusogram (IOP-Compass API).
Mounted into api.py via `router`."""

from .router import router, load_models, status as router_status  # noqa: F401
