"""
puente_sprint2.py · Carga el paquete `src` del Sprint 2 bajo el alias `s2`.

El Sprint 3 NO duplica lógica: la limpieza, el preprocesador, las métricas y
las utilidades de E/S viven en el Sprint 2 y aquí solo se importan. Se usa
importlib con un nombre de paquete propio ("sprint2_src") para evitar el
choque con el paquete `src` del propio Sprint 3.
"""
from __future__ import annotations

import importlib.util
import sys

from . import config as C3

_NOMBRE = "sprint2_src"


def _cargar_paquete_sprint2():
    if _NOMBRE in sys.modules:
        return sys.modules[_NOMBRE]
    init = C3.DIR_SPRINT2 / "src" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _NOMBRE, init, submodule_search_locations=[str(init.parent)]
    )
    paquete = importlib.util.module_from_spec(spec)
    sys.modules[_NOMBRE] = paquete
    spec.loader.exec_module(paquete)
    return paquete


_s2 = _cargar_paquete_sprint2()

import importlib as _il

# Módulos del Sprint 2 expuestos al Sprint 3 (importación explícita de submódulos)
config2   = _il.import_module(f"{_NOMBRE}.config")
utils     = _il.import_module(f"{_NOMBRE}.utils")
features  = _il.import_module(f"{_NOMBRE}.features")
cleaning  = _il.import_module(f"{_NOMBRE}.cleaning")
selection = _il.import_module(f"{_NOMBRE}.selection")
pipeline  = _il.import_module(f"{_NOMBRE}.pipeline")
metrics   = _il.import_module(f"{_NOMBRE}.metrics")
