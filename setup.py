from __future__ import annotations

from Cython.Build import cythonize
from setuptools import Extension, setup

import numpy as np


extensions = [
    Extension(
        "xcdrjit._every_supported_cython",
        ["xcdrjit/_every_supported_cython.pyx"],
        include_dirs=[np.get_include()],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    )
)
