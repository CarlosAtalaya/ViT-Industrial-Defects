#!/bin/bash

# =============================================================================
# SCRIPT HELPER PARA RECALCULAR MÉTRICAS CON SCORE THRESHOLD ALTO
# =============================================================================
# Este script facilita la ejecución del recálculo de métricas con un threshold
# más alto para verificar si el modelo está realmente bien o está overfitteado.
# =============================================================================

set -e

echo "================================================================================"
echo "  RECÁLCULO DE MÉTRICAS CON SCORE THRESHOLD ALTO"
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------

# Rutas del proyecto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/deimv2_multimodal"

# Dataset
DATASET_PATH="${PROJECT_ROOT}/curated_dataset_splitted_20251101_provisional_1st_version"
TEST_ANN_FILE="${DATASET_PATH}/test/test.json"

# Experimento por defecto (mejor modelo)
EXPERIMENT_DIR="${SCRIPT_DIR}/outputs/deimv2_1024_300epochs"
DETECTIONS_FILE="${EXPERIMENT_DIR}/test_detections_filtered.json"

# Score threshold por defecto
SCORE_THRESHOLD="${1:-0.75}"

# -----------------------------------------------------------------------------
# VALIDACIÓN
# -----------------------------------------------------------------------------

if [ ! -f "$DETECTIONS_FILE" ]; then
    echo "❌ ERROR: No se encontró el archivo de detecciones:"
    echo "   $DETECTIONS_FILE"
    echo ""
    echo "💡 Asegúrate de que el experimento ha sido evaluado previamente."
    exit 1
fi

if [ ! -f "$TEST_ANN_FILE" ]; then
    echo "❌ ERROR: No se encontró el archivo de anotaciones:"
    echo "   $TEST_ANN_FILE"
    exit 1
fi

# -----------------------------------------------------------------------------
# EJECUCIÓN
# -----------------------------------------------------------------------------

echo "📊 Configuración:"
echo "   Experimento: $EXPERIMENT_DIR"
echo "   Score threshold: $SCORE_THRESHOLD"
echo "   IoU threshold: 0.5 (por defecto)"
echo ""

echo "🚀 Ejecutando recálculo de métricas..."
echo ""

python3 "${SCRIPT_DIR}/recalculate_metrics_with_threshold.py" \
    --detections-file "$DETECTIONS_FILE" \
    --test-ann-file "$TEST_ANN_FILE" \
    --score-threshold "$SCORE_THRESHOLD" \
    --iou-threshold 0.5

echo ""
echo "================================================================================"
echo "  ✅ RECÁLCULO COMPLETADO"
echo "================================================================================"
echo ""
echo "📁 Archivos generados:"
echo "   📊 test_evaluation_results_comparable_th${SCORE_THRESHOLD}.json"
echo "   📋 test_detections_filtered_th${SCORE_THRESHOLD}.json"
echo ""
echo "💡 Compara estos resultados con los originales (threshold 0.15) para verificar"
echo "   si el modelo mantiene buen rendimiento con un threshold más estricto."
echo ""

