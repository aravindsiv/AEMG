import os
import sys
from setuptools import setup, find_packages

setup (
    name = "AEMG",
    version = "0.1",
    packages = find_packages(),
    author = "Aravind Sivaramakrishnan, Dhruv Metha Ramesh, Ewerton Rocha Vieira",
    url = "https://github.com/aravindsiv/AEMG.git",
    description = "Autoencoding dynamics for applying Morse Graphs to high-dimensional dynamical systems",
    long_description = open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'pandas', 'seaborn', 'tqdm', 'torch', 'torchvision', 'wandb'],
)