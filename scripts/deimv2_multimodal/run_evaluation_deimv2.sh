#!/bin/bash

# =============================================================================
# PIPELINE COMPLETO DE EVALUACIÓN PARA DEIMV2
# Detección de Defectos Industriales con Vision Transformers
# 
# ACTUALIZADO: Usa evaluate_deimv2_comparable.py para métricas comparables
# =============================================================================

set -e  # Salir si hay error

echo "================================================================================"
echo "  PIPELINE DE EVALUACIÓN - DEIMV2 INDUSTRIAL (COMPARABLE CON BASELINES)"
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

# Thresholds de evaluación (ajusta según necesites)
SCORE_THRESHOLD=0.15  # Mismo que baselines CNN
IOU_THRESHOLD=0.5     # IoU estándar

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
    
    # Permitir override de thresholds
    if [ ! -z "$2" ]; then
        SCORE_THRESHOLD="$2"
    fi
    if [ ! -z "$3" ]; then
        IOU_THRESHOLD="$3"
    fi
fi

echo "📁 Directorio de experimento: $EXPERIMENT_DIR"
echo "💾 Checkpoint: $CHECKPOINT"
echo "📊 Score threshold: $SCORE_THRESHOLD"
echo "📐 IoU threshold: $IOU_THRESHOLD"
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
        --log-path "$LOG_FILE" \
        --output-dir "$EXPERIMENT_DIR"
    echo ""
else
    echo "⚠️  Log de entrenamiento no encontrado, saltando visualización"
    echo ""
fi

# 2. EVALUACIÓN EN TEST SET (MÉTRICAS COMPARABLES)
echo "================================================================================"
echo "PASO 2: EVALUACIÓN EN TEST SET (MÉTRICAS COMPARABLES CON BASELINES)"
echo "================================================================================"
echo ""

python3 "${SCRIPT_DIR}/evaluate_deimv2_comparable.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --test-img-folder "$TEST_IMG_FOLDER" \
    --test-ann-file "$TEST_ANN_FILE" \
    --score-threshold "$SCORE_THRESHOLD" \
    --iou-threshold "$IOU_THRESHOLD"

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
    --score-threshold "$SCORE_THRESHOLD"

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
echo "  📈 test_evaluation_results_comparable.json - Métricas comparables con baselines"
echo "  🗂️  test_detections_filtered.json - Detecciones filtradas (score >= $SCORE_THRESHOLD)"
echo "  🖼️  visualizations_test/ - Predicciones visualizadas"
echo ""

# Mostrar métricas si existen
RESULTS_FILE="${EXPERIMENT_DIR}/test_evaluation_results_comparable.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "Métricas de Test (Comparables con ResNet/EfficientNet):"
    python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
    print(f\"  mAP (IoU=${IOU_THRESHOLD}): {data['mAP']:.4f}\")
    print(f\"  Score threshold: {data['score_threshold']}\")
    print(f\"  IoU threshold: {data['iou_threshold']}\")
    print(f\"\\n  Métricas por clase:\")
    for cls, ap in data['AP_per_class'].items():
        prec = data['precision_per_class'][cls]
        rec = data['recall_per_class'][cls]
        print(f\"    {cls}: AP={ap:.4f}, Prec={prec:.4f}, Rec={rec:.4f}\")
"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "OPCIONES DE USO:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Evaluar con checkpoint específico:"
echo "   ./run_evaluation_deimv2.sh /ruta/al/checkpoint.pth"
echo ""
echo "2. Evaluar con thresholds personalizados:"
echo "   ./run_evaluation_deimv2.sh /ruta/al/checkpoint.pth 0.25 0.5"
echo "   (Formato: checkpoint score_threshold iou_threshold)"
echo ""
echo "3. Recalcular métricas desde detecciones ya guardadas:"
echo "   python recalculate_metrics_from_detections.py \\"
echo "     --detections-file $EXPERIMENT_DIR/test_detections.json \\"
echo "     --ann-file $TEST_ANN_FILE \\"
echo "     --score-threshold 0.20 \\"
echo "     --iou-threshold 0.5"
echo ""
echo "4. Comparar múltiples thresholds (sin re-ejecutar inferencia):"
echo "   python recalculate_metrics_from_detections.py \\"
echo "     --detections-file $EXPERIMENT_DIR/test_detections.json \\"
echo "     --ann-file $TEST_ANN_FILE \\"
echo "     --compare-thresholds"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"