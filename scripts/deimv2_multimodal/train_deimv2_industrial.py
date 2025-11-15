#!/usr/bin/env python3
"""
Script de entrenamiento DEIMv2 para detección de defectos industriales
"""

import os
import sys
from pathlib import Path

# Agregar DEIMv2 al path
DEIMV2_PATH = Path(__file__).parent.parent.parent / "DEIMv2"
sys.path.insert(0, str(DEIMV2_PATH))

# Cambiar al directorio de DEIMv2 para que encuentre los archivos
os.chdir(DEIMV2_PATH)

# Modificar sys.argv con los argumentos correctos
config_path = Path(__file__).parent / "configs" / "deimv2_industrial_defects.yml"

sys.argv = [
    'train.py',
    '-c', str(config_path.resolve()),
    '--use-amp',
    '--seed', '0'
]

# Ejecutar el script original de DEIMv2
exec(open('train.py').read())