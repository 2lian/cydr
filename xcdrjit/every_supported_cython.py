"""Loader for the `_every_supported_cython` example extension module."""

from __future__ import annotations

import numpy as np
import pyximport


pyximport.install(
    language_level=3,
    setup_args={"include_dirs": np.get_include()},
)

from xcdrjit._every_supported_cython import (  # noqa: E402
    compute_serialized_size_every_supported_schema,
    deserialize_every_supported_schema,
    serialize_every_supported_schema,
)


__all__ = [
    "compute_serialized_size_every_supported_schema",
    "deserialize_every_supported_schema",
    "serialize_every_supported_schema",
]
