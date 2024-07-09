"""
AIClasses
=================

Breve descripción del paquete y su funcionalidad principal.

Submódulos disponibles:
- submodulo1: Descripción breve del submódulo 1.
- submodulo2: Descripción breve del submódulo 2.
- subpaquete: Descripción breve del subpaquete.

Ejemplo de uso:
>>> from nombre_del_paquete import Clase1
>>> obj = Clase1()
>>> obj.metodo()
"""

# Definición de la versión del paquete
__version__ = '1.0.0'

# Configuración del registro para el paquete
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importación de submódulos
from .submodulo1 import Clase1, funcion1
from .submodulo2 import Clase2, funcion2

# Importación de subpaquetes
from .subpaquete import modulo3

# Definir el API del paquete mediante __all__
__all__ = ['Clase1', 'funcion1', 'Clase2', 'funcion2', 'modulo3']

# Comentario: La cadena de documentación proporciona una descripción general del paquete.
# Comentario: __version__ establece la versión actual del paquete.
# Comentario: logging configura el sistema de registro para el paquete.
# Comentario: Los import statements exponen los submódulos y subpaquetes principales.
# Comentario: __all__ define explícitamente qué se exporta cuando se usa from nombre_del_paquete import *.


