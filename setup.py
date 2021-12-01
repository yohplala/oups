#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from setuptools import setup


setup(
    name="oups",
    version="0.1",
    author="Yoh Plala",
    author_email="yoh.plala@gmail.com",
    url="https://github.com/yohplala/oups",
    description="Ordered Updatable Parquet Store",
    python_requires=">=3.8",
    tests_require=["pytest"],
    install_requires=["pandas>=1.3.1",
                      "fastparquet>=0.7.1"]
)
