"""Build the optional Cython hot loop (``pdc._match``).

The package runs correctly in pure Python without this; building it only speeds up the
per-stage matcher.  Build in place with:

    python3 setup.py build_ext --inplace

or install (``pip install -e .``) with Cython + a C compiler available.
"""

from setuptools import setup, Extension

import numpy as np

try:
    from Cython.Build import cythonize
    HAVE_CYTHON = True
except ImportError:
    HAVE_CYTHON = False

ext = ".pyx" if HAVE_CYTHON else ".c"
extensions = [
    Extension(
        "pdc._match",
        ["pdc/_match" + ext],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]
if HAVE_CYTHON:
    extensions = cythonize(extensions, language_level=3)

setup(ext_modules=extensions)
