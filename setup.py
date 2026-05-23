# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "cholrot._core_ext",
        ["src/cholrot/_core_ext.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
