#!/bin/bash

# =============================================================================
# PIPELINE COMPLETO DE EVALUACIÓN PARA DEIMV2
# Detección de Defectos Industriales con Vision Transformers
# =============================================================================

set -e  # Salir si hay error

echo "================================================================================"
echo "  PIPELINE DE EVALUACIÓN - DEIMV2 INDUSTRIAL"
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
TEST_IMG_FOLDER="${DATASET_PATH}/test/images"
TEST_ANN_FILE="${DATASET_PATH}/test/test.json"

# Configuración de entrenamiento
CONFIG_FILE="${SCRIPT_DIR}/configs/deimv2_industrial_defects.yml"

# Buscar último checkpoint (o especificar manualmente)
if [ -z "$1" ]; then
    # Buscar automáticamente el último checkpoint
    OUTPUT_BASE="${SCRIPT_DIR}/outputs"
    
    if [ ! -d "$OUTPUT_BASE" ]; then
        echo "❌ ERROR: No se encontró directorio de outputs en $OUTPUT_BASE"
        exit 1
    fi
    
    # Buscar el checkpoint más reciente
    LATEST_RUN=$(ls -td ${OUTPUT_BASE}/deimv2_industrial_run* 2>/dev/null | head -1)
    
    if [ -z "$LATEST_RUN" ]; then
        echo "❌ ERROR: No se encontraron runs de entrenamiento"
        echo "Ejecuta primero: python train_deimv2_industrial.py"
        exit 1
    fi
    
    # Buscar checkpoint_best.pth o el último checkpoint
    if [ -f "${LATEST_RUN}/checkpoint_best.pth" ]; then
        CHECKPOINT="${LATEST_RUN}/checkpoint_best.pth"
    else
        CHECKPOINT=$(ls -t ${LATEST_RUN}/checkpoint*.pth 2>/dev/null | head -1)
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo "❌ ERROR: No se encontró checkpoint en $LATEST_RUN"
        exit 1
    fi
    
    EXPERIMENT_DIR="$LATEST_RUN"
else
    # Usar checkpoint especificado
    CHECKPOINT="$1"
    EXPERIMENT_DIR="$(dirname "$CHECKPOINT")"
fi

echo "📁 Directorio de experimento: $EXPERIMENT_DIR"
echo "💾 Checkpoint: $CHECKPOINT"
echo ""

# Verificar archivos necesarios
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint no encontrado: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config no encontrado: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$TEST_IMG_FOLDER" ]; then
    echo "❌ ERROR: Directorio de test no encontrado: $TEST_IMG_FOLDER"
    exit 1
fi

if [ ! -f "$TEST_ANN_FILE" ]; then
    echo "❌ ERROR: Anotaciones de test no encontradas: $TEST_ANN_FILE"
    exit 1
fi

echo "✅ Verificación de archivos completada"
echo ""

# -----------------------------------------------------------------------------
# PIPELINE DE EVALUACIÓN
# -----------------------------------------------------------------------------

# 1. VISUALIZAR MÉTRICAS DE ENTRENAMIENTO
echo "================================================================================"
echo "PASO 1: VISUALIZAR MÉTRICAS DE ENTRENAMIENTO"
echo "================================================================================"
echo ""

LOG_FILE="${EXPERIMENT_DIR}/log.txt"

if [ -f "$LOG_FILE" ]; then
    python3 "${SCRIPT_DIR}/plot_deimv2_training_metrics.py" \
        --log-path "$LOG_FILE"
    echo ""
else
    echo "⚠️  Log de entrenamiento no encontrado, saltando visualización"
    echo ""
fi

# 2. EVALUACIÓN EN TEST SET
echo "================================================================================"
echo "PASO 2: EVALUACIÓN EN TEST SET (COCO mAP)"
echo "================================================================================"
echo ""

python3 "${SCRIPT_DIR}/evaluate_deimv2.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --test-img-folder "$TEST_IMG_FOLDER" \
    --test-ann-file "$TEST_ANN_FILE"

echo ""

# 3. VISUALIZAR PREDICCIONES
echo "================================================================================"
echo "PASO 3: VISUALIZAR PREDICCIONES EN TEST"
echo "================================================================================"
echo ""

python3 "${SCRIPT_DIR}/visualize_deimv2_predictions.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --img-folder "$TEST_IMG_FOLDER" \
    --ann-file "$TEST_ANN_FILE" \
    --num-images 30 \
    --random \
    --score-threshold 0.75

echo ""

# -----------------------------------------------------------------------------
# RESUMEN FINAL
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "  ✅ PIPELINE DE EVALUACIÓN COMPLETADO"
echo "================================================================================"
echo ""
echo "Resultados guardados en: $EXPERIMENT_DIR"
echo ""
echo "Archivos generados:"
echo "  📊 training_metrics.png - Gráficas de entrenamiento"
echo "  📈 test_evaluation_results.json - Métricas mAP en test"
echo "  🖼️  visualizations_test/ - Predicciones visualizadas"
echo ""

# Mostrar métricas si existen
RESULTS_FILE="${EXPERIMENT_DIR}/test_evaluation_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "Métricas de Test:"
    python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
    print(f\"  mAP@0.50:0.95: {data['metrics']['mAP']:.4f}\")
    print(f\"  AP@0.50:      {data['metrics']['AP50']:.4f}\")
    print(f\"  AP@0.75:      {data['metrics']['AP75']:.4f}\")
"
fi

echo ""
echo "Para ejecutar de nuevo con otro checkpoint:"
echo "  ./run_evaluation_deimv2.sh /ruta/al/checkpoint.pth"
echo ""