#!/usr/bin/env python
#-*- coding:utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = "auto_phase_diagram",
    version = "0.4.1",
    keywords = ("phase diagram", "themodynamics"),
    description = "Automatic analysis and plot phase diagram from DFT calculation",
    license = "MIT Licence",

    url = "https://github.com/renpj/auto_phase_diagram",
    author = "PJ Ren",
    author_email = "openrpj@gmail.com",

    packages = find_packages(),
    #include_package_data = True,
    package_data = {"auto_phase_diagram": ["*.xlsx","*.vsz"]}, # use MANIFEST.in maybe
    entry_points={
        'console_scripts': [
            'plot_phase_diagram = auto_phase_diagram.__main__:main'
        ]
    },
    platforms = "any",
    install_requires = [
        "numpy",
        "pandas",
        "xlrd", 
        "xlwt"
    ] 
)
