#!/bin/bash

# Script de ejemplo para ejecutar el pipeline completo de entrenamiento y evaluación
# ResNet-18 + Faster R-CNN para detección de defectos industriales

set -e  # Salir si hay algún error

echo "=========================================="
echo "PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN"
echo "Detección de Defectos Industriales"
echo "Arquitectura: ResNet-18 + Faster R-CNN"
echo "=========================================="
echo ""

# Configuración
DATASET_PATH="/home/carlos/Escritorio/Proyectos_Personales/TFG_25-26/TFG-ViT/ViT/ViT-Industrial-Defects/curated_dataset_splitted_20251101_provisional_1st_version"
EPOCHS=50
BATCH_SIZE=8
LR=0.005
NUM_WORKERS=4

# Verificar que existe el dataset
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: No se encontró el dataset en $DATASET_PATH"
    exit 1
fi

# 1. ENTRENAMIENTO
echo "=========================================="
echo "PASO 1: ENTRENAMIENTO"
echo "=========================================="
echo ""

python3 train_resnet18_fasterrcnn.py \
    --dataset-path "$DATASET_PATH" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --num-workers $NUM_WORKERS \
    --pretrained-backbone \
    --output-dir results/training

# Encontrar el último directorio de experimento
EXPERIMENT_DIR=$(ls -td results/training/resnet18_fasterrcnn_* | head -1)
echo ""
echo "Experimento guardado en: $EXPERIMENT_DIR"

# 2. VISUALIZAR MÉTRICAS DE ENTRENAMIENTO
echo ""
echo "=========================================="
echo "PASO 2: VISUALIZAR MÉTRICAS"
echo "=========================================="
echo ""

python3 plot_training_metrics.py \
    --history-path "$EXPERIMENT_DIR/training_history.json"

# 3. EVALUACIÓN EN TEST SET
echo ""
echo "=========================================="
echo "PASO 3: EVALUACIÓN EN TEST SET"
echo "=========================================="
echo ""

python3 evaluate_model.py \
    --checkpoint "$EXPERIMENT_DIR/checkpoints/best_checkpoint.pth" \
    --dataset-path "$DATASET_PATH" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --score-threshold 0.5 \
    --iou-threshold 0.5

# 4. VISUALIZAR PREDICCIONES
echo ""
echo "=========================================="
echo "PASO 4: VISUALIZAR PREDICCIONES"
echo "=========================================="
echo ""

python3 visualize_predictions.py \
    --checkpoint "$EXPERIMENT_DIR/checkpoints/best_checkpoint.pth" \
    --dataset-path "$DATASET_PATH" \
    --split test \
    --num-images 30 \
    --random \
    --score-threshold 0.5

echo ""
echo "=========================================="
echo "PIPELINE COMPLETADO"
echo "=========================================="
echo ""
echo "Resultados guardados en: $EXPERIMENT_DIR"
echo ""
echo "Contenido del directorio:"
ls -lh "$EXPERIMENT_DIR"
echo ""
echo "Checkpoints disponibles:"
ls -lh "$EXPERIMENT_DIR/checkpoints/"
echo ""