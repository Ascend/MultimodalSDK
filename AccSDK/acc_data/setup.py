#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import os
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        build_type = os.environ.get("CMAKE_BUILD_TYPE", "")
        cfg = build_type
        enable_tracer = int(os.environ.get("ACCDATA_ENABLE_TRACER", 0))
        tracer_switch = "ON" if enable_tracer else "OFF"

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # ACCDATA_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}accdata{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DENABLE_TRACER={tracer_switch}",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if "CMAKE_PREFIX_PATH" in os.environ:
            cmake_args += [f"-DCMAKE_PREFIX_PATH={os.environ['CMAKE_PREFIX_PATH']}"]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DACCDATA_VERSION_INFO={self.distribution.get_version()}"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_path = os.environ.get("ACCDATA_BUILD_PATH", "")
        if build_path == "":
            build_path = Path(self.build_temp) / ext.name
        else:
            build_path = Path(build_path)
        if not build_path.exists():
            build_path.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_path, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_path, check=True
        )
        subprocess.run(
            ["cmake", "--install", "."], cwd=build_path, check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="accdata",
    version="0.0.1",
    author="Huawei Technologies Co.",
    author_email="",
    description="An AI Accelerate Data Process Kit",
    packages=setuptools.find_packages('src/python'),
    package_dir={'': 'src/python'},
    include_package_data=True,
    ext_modules=[CMakeExtension("accdata")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
)
