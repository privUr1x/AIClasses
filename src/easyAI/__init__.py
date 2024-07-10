"""
easyAI
=================

For the sake of simplicity and the python motto, easyAI allows deep learning models to be used as a reusable package. Using python as the engine without the need for any data science framework.

Submódulos disponibles:
- core: Core funciona como el motor de la librería partiendo de la base formada por algunas clases que permiten la creación de modelos complejos.
- classtools: Ofrece herramientas para la creación de clases.
- Arquitectures: Diversidad de modelos de deep learning utilizables.

Ejemplo de uso:
>>> from Arquitectures import Perceptron
>>> p = Perceptron(2) # Number of inputs/entries 
>>> p.fit(X, y, verbose = True)
"""

# Definición de la versión del paquete
__version__ = "0.1.6"

# Configuración del registro para el paquete
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadinedo path del paquete
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importación de submódulos
from .Arquitectures import Perceptron, MLP
from .classtools import Verifiers

# Importación de subpaquetes
from .core import objects
from .core import activations
from .core import loss_func

# Definir el API del paquete mediante __all__
__all__ = ["MLP", "Perceptron", "activations", "loss_func"]
