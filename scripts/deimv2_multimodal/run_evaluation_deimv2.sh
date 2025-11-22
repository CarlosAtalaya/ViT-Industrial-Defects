#!/bin/bash

# =============================================================================
# PIPELINE COMPLETO DE EVALUACIÓN PARA DEIMV2 - VERSIÓN ACTUALIZADA
# Detección de Defectos Industriales con Vision Transformers @ 1024×1024
# =============================================================================

set -e  # Salir si hay error

echo "================================================================================"
echo "  PIPELINE DE EVALUACIÓN - DEIMV2 INDUSTRIAL (1024×1024)"
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

# Configuración de entrenamiento (con resolución 1024×1024)
CONFIG_FILE="${SCRIPT_DIR}/configs/deimv2_industrial_defects.yml"

# -----------------------------------------------------------------------------
# BUSCAR CHECKPOINT
# -----------------------------------------------------------------------------

if [ -z "$1" ]; then
    # Buscar automáticamente el último checkpoint
    OUTPUT_BASE="${SCRIPT_DIR}/outputs"
    
    if [ ! -d "$OUTPUT_BASE" ]; then
        echo "❌ ERROR: No se encontró directorio de outputs en $OUTPUT_BASE"
        exit 1
    fi
    
    echo "🔍 Buscando último experimento..."
    echo ""
    
    # Buscar el run más reciente (prioriza runs con 1024 en el nombre)
    LATEST_RUN=$(ls -td ${OUTPUT_BASE}/deimv2_1024* ${OUTPUT_BASE}/deimv2_*_run* 2>/dev/null | head -1)
    
    if [ -z "$LATEST_RUN" ]; then
        echo "❌ ERROR: No se encontraron runs de entrenamiento"
        echo "Ejecuta primero: python train_deimv2_industrial.py"
        exit 1
    fi
    
    echo "📂 Experimento encontrado: $(basename $LATEST_RUN)"
    
    # Buscar el mejor checkpoint (prioridad: best_stg1.pth > checkpoint0080.pth > último)
    if [ -f "${LATEST_RUN}/best_stg1.pth" ]; then
        CHECKPOINT="${LATEST_RUN}/best_stg1.pth"
        echo "✓ Usando best_stg1.pth (mejor mAP en validación)"
    elif [ -f "${LATEST_RUN}/checkpoint0080.pth" ]; then
        CHECKPOINT="${LATEST_RUN}/checkpoint0080.pth"
        echo "✓ Usando checkpoint0080.pth (última época)"
    else
        CHECKPOINT=$(ls -t ${LATEST_RUN}/checkpoint*.pth 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            echo "❌ ERROR: No se encontró checkpoint en $LATEST_RUN"
            exit 1
        fi
        echo "✓ Usando: $(basename $CHECKPOINT)"
    fi
    
    EXPERIMENT_DIR="$LATEST_RUN"
else
    # Usar checkpoint especificado manualmente
    CHECKPOINT="$1"
    EXPERIMENT_DIR="$(dirname "$CHECKPOINT")"
    echo "💾 Usando checkpoint especificado: $CHECKPOINT"
fi

echo ""
echo "📁 Directorio de experimento: $EXPERIMENT_DIR"
echo "💾 Checkpoint: $(basename $CHECKPOINT)"
echo ""

# -----------------------------------------------------------------------------
# VERIFICAR ARCHIVOS NECESARIOS
# -----------------------------------------------------------------------------

echo "🔍 Verificando archivos..."

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
# EXTRAER INFO DEL CHECKPOINT
# -----------------------------------------------------------------------------

# Intentar extraer época del nombre del checkpoint
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
if [[ $CHECKPOINT_NAME =~ checkpoint([0-9]+) ]]; then
    EPOCH="${BASH_REMATCH[1]}"
    echo "📊 Checkpoint de época: $EPOCH"
elif [[ $CHECKPOINT_NAME == "best_stg1.pth" ]]; then
    echo "📊 Checkpoint: Mejor modelo (validación)"
else
    echo "📊 Checkpoint: Desconocido"
fi
echo ""

# -----------------------------------------------------------------------------
# PIPELINE DE EVALUACIÓN
# -----------------------------------------------------------------------------

# PASO 1: VISUALIZAR MÉTRICAS DE ENTRENAMIENTO
echo "================================================================================"
echo "PASO 1: VISUALIZAR MÉTRICAS DE ENTRENAMIENTO"
echo "================================================================================"
echo ""

LOG_FILE="${EXPERIMENT_DIR}/log.txt"

if [ -f "$LOG_FILE" ]; then
    echo "📈 Generando gráficas de training..."
    python3 "${SCRIPT_DIR}/plot_deimv2_training_metrics.py" \
        --log-path "$LOG_FILE" 2>/dev/null || echo "⚠️  Error al generar gráficas (puede requerir actualización del parser)"
    echo ""
else
    echo "⚠️  Log de entrenamiento no encontrado, saltando visualización"
    echo ""
fi

# PASO 2: EVALUACIÓN EN TEST SET
echo "================================================================================"
echo "PASO 2: EVALUACIÓN EN TEST SET (Protocolo COCO)"
echo "================================================================================"
echo ""
echo "🔬 Evaluando modelo en test set (205 imágenes)..."
echo "⏱️  Esto puede tomar 1-2 minutos..."
echo ""

python3 "${SCRIPT_DIR}/evaluate_deimv2.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --test-img-folder "$TEST_IMG_FOLDER" \
    --test-ann-file "$TEST_ANN_FILE"

echo ""

# PASO 3: VISUALIZAR PREDICCIONES
echo "================================================================================"
echo "PASO 3: VISUALIZAR PREDICCIONES EN TEST"
echo "================================================================================"
echo ""
echo "🖼️  Generando visualizaciones de predicciones..."
echo ""

# Generar visualizaciones con diferentes thresholds
python3 "${SCRIPT_DIR}/visualize_deimv2_predictions.py" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_FILE" \
    --img-folder "$TEST_IMG_FOLDER" \
    --ann-file "$TEST_ANN_FILE" \
    --num-images 30 \
    --random \
    --score-threshold 0.15

echo ""

# -----------------------------------------------------------------------------
# RESUMEN FINAL
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "  ✅ PIPELINE DE EVALUACIÓN COMPLETADO"
echo "================================================================================"
echo ""
echo "📂 Resultados guardados en: $EXPERIMENT_DIR"
echo ""
echo "📁 Archivos generados:"
echo "  📊 training_metrics.png           - Gráficas de entrenamiento (si disponible)"
echo "  📈 test_evaluation_results.json   - Métricas mAP en test"
echo "  🖼️  visualizations_test/          - Predicciones visualizadas (30 imágenes)"
echo ""

# Mostrar métricas si existen
RESULTS_FILE="${EXPERIMENT_DIR}/test_evaluation_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📊 MÉTRICAS FINALES EN TEST SET"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    python3 -c "
import json
import sys

try:
    with open('$RESULTS_FILE') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    print('  Métricas Principales:')
    print(f\"    mAP@0.50:0.95:  {metrics.get('mAP', 0):.4f}  (39.5% es baseline con 640px)\")
    print(f\"    AP@0.50:        {metrics.get('AP50', 0):.4f}  (49.9% es baseline con 640px)\")
    print(f\"    AP@0.75:        {metrics.get('AP75', 0):.4f}  (38.4% es baseline con 640px)\")
    print()
    
    if 'AP_small' in metrics:
        print('  Métricas por Tamaño:')
        print(f\"    AP Small:       {metrics.get('AP_small', 0):.4f}  (objetos < 32²)\")
        print(f\"    AP Medium:      {metrics.get('AP_medium', 0):.4f}  (objetos 32²-96²)\")
        print(f\"    AP Large:       {metrics.get('AP_large', 0):.4f}  (objetos > 96²)\")
        print()
    
    if 'Recall' in metrics:
        print('  Recall:')
        print(f\"    AR@100:         {metrics.get('Recall', 0):.4f}\")
        print()
    
    # Comparación con baseline
    baseline_map = 0.395
    current_map = metrics.get('mAP', 0)
    improvement = ((current_map - baseline_map) / baseline_map) * 100 if baseline_map > 0 else 0
    
    print('  Comparación vs Baseline (640px):')
    if improvement > 0:
        print(f\"    Mejora: +{improvement:.1f}% 🚀\")
    elif improvement < 0:
        print(f\"    Diferencia: {improvement:.1f}%\")
    else:
        print(f\"    Similar al baseline\")
    print()
    
except Exception as e:
    print(f\"⚠️  Error al leer métricas: {e}\", file=sys.stderr)
"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "⚠️  Archivo de resultados no encontrado"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  💡 SIGUIENTES PASOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Revisar visualizaciones:"
echo "   cd $EXPERIMENT_DIR/visualizations_test"
echo "   # Ver predicciones para analizar errores"
echo ""
echo "2. Comparar con baselines CNN:"
echo "   cd scripts/resnet18"
echo "   python evaluate_model.py --checkpoint ... --score-threshold 0.5"
echo ""
echo "3. Si resultados son buenos (mAP ≥ 0.45):"
echo "   - Documentar en TFG"
echo "   - Proceder con FASE 2 (extensión multimodal)"
echo ""
echo "4. Para re-evaluar con otro checkpoint:"
echo "   ./run_evaluation_deimv2.sh /ruta/al/checkpoint.pth"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"