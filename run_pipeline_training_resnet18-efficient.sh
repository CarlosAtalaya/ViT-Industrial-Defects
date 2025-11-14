#!/bin/bash

# Script para ejecutar todos los pipelines de baselines secuencialmente
# ResNet-18 → EfficientNet-B0

set -e  # Salir si hay algún error

echo "================================================================================"
echo "EJECUCIÓN AUTOMATIZADA DE TODOS LOS BASELINES"
echo "================================================================================"
echo ""
echo "Este script ejecutará secuencialmente:"
echo "  1. ResNet-18 + Faster R-CNN"
echo "  2. EfficientNet-B0 + Faster R-CNN"
echo ""
echo "================================================================================"
echo ""

# Obtener directorio raíz del repositorio (donde está este script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Directorio raíz: $SCRIPT_DIR"
echo ""

# ==============================================================================
# BASELINE 1: ResNet-18 + Faster R-CNN
# ==============================================================================

echo ""
echo "================================================================================"
echo "INICIANDO BASELINE 1: ResNet-18 + Faster R-CNN"
echo "================================================================================"
echo ""

cd "$SCRIPT_DIR/scripts/resnet18"

if [ ! -f "run_pipeline.sh" ]; then
    echo "ERROR: No se encontró run_pipeline.sh en scripts/resnet18/"
    exit 1
fi

# Dar permisos de ejecución si no los tiene
chmod +x run_pipeline.sh

# Ejecutar pipeline de ResNet-18
echo "Ejecutando: scripts/resnet18/run_pipeline.sh"
./run_pipeline.sh

RESNET_EXIT_CODE=$?

if [ $RESNET_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: El pipeline de ResNet-18 falló con código de salida $RESNET_EXIT_CODE"
    echo "================================================================================"
    exit $RESNET_EXIT_CODE
fi

echo ""
echo "================================================================================"
echo "✓ BASELINE 1 COMPLETADO: ResNet-18 + Faster R-CNN"
echo "================================================================================"
echo ""

# ==============================================================================
# BASELINE 2: EfficientNet-B0 + Faster R-CNN
# ==============================================================================

echo ""
echo "================================================================================"
echo "INICIANDO BASELINE 2: EfficientNet-B0 + Faster R-CNN"
echo "================================================================================"
echo ""

cd "$SCRIPT_DIR/scripts/efficientnet"

if [ ! -f "run_pipeline.sh" ]; then
    echo "ERROR: No se encontró run_pipeline.sh en scripts/efficientnet/"
    exit 1
fi

# Dar permisos de ejecución si no los tiene
chmod +x run_pipeline.sh

# Ejecutar pipeline de EfficientNet
echo "Ejecutando: scripts/efficientnet/run_pipeline.sh"
./run_pipeline.sh

EFFICIENTNET_EXIT_CODE=$?

if [ $EFFICIENTNET_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: El pipeline de EfficientNet falló con código de salida $EFFICIENTNET_EXIT_CODE"
    echo "================================================================================"
    exit $EFFICIENTNET_EXIT_CODE
fi

echo ""
echo "================================================================================"
echo "✓ BASELINE 2 COMPLETADO: EfficientNet-B0 + Faster R-CNN"
echo "================================================================================"
echo ""

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

cd "$SCRIPT_DIR"

echo ""
echo "================================================================================"
echo "✓✓✓ TODOS LOS BASELINES COMPLETADOS EXITOSAMENTE ✓✓✓"
echo "================================================================================"
echo ""
echo "Baselines ejecutados:"
echo "  ✓ ResNet-18 + Faster R-CNN"
echo "  ✓ EfficientNet-B0 + Faster R-CNN"
echo ""
echo "Puedes comparar los resultados ejecutando:"
echo "  python scripts/compare_differents_architectures_experiments.py"
echo ""
echo "================================================================================"