"""
easyAI
=================

For the sake of simplicity and the python motto, easyAI allows deep learning models to be used as a reusable package. Using python as the engine without the need for any data science framework.

Available submodules:
- core: Core works as the engine of the library consisting in base clases which allows the creation of complex models.
- classtools: Ofers tools for class creation.
- Arquitectures: Diversity of Deep Learning arquitectures.

Usage:
>>> from Arquitectures import Perceptron
>>> p = Perceptron(2) # Number of inputs/entries 
>>> history = p.fit(X, y, verbose=True)
"""

from tomllib import load

tomlpath: str = "/".join(
    __file__.split("/")[:-1].__add__(["..", "..", "pyproject.toml"])
)

with open(tomlpath, "rb") as f:
    toml = load(f)

VERSION = toml["tool"]["poetry"]["version"]

# Version package definition
__version__ = VERSION 

# Registry config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .configs.default import config 
from random import seed 

seed(config["seed"])

# Appending package path
from sys import path
from os.path import dirname, abspath

path.append(dirname(abspath(__file__)))

# Submodules import
from . import arquitectures
from . import layers

# Subpackages import
from .core import objects
from .core import activations
from .core import loss
from .core import optimizers

# Package API definition through __all__
__all__ = ["activations", "loss", "objects", "optimizers", "arquitectures", "layers"]
