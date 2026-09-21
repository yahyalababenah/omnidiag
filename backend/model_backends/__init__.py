"""
OmniDiag — model-family backends.

Importing this package imports every module in it, so a backend registers
itself (@register_backend) just by existing as a file here — adding a model
family touches no other file. See backend/model_backends/base.py.
"""

import importlib
import pkgutil

from backend.model_backends.base import (  # noqa: F401  (public API)
    BackendCapabilities,
    ModelBackend,
    ShapResult,
    UnknownModelFamilyError,
    get_backend,
    register_backend,
    registered_families,
)

for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")
