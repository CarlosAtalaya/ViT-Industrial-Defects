# Detección de Defectos Industriales con ResNet-18 + Faster R-CNN

Sistema de detección de defectos en componentes industriales usando ResNet-18 como backbone en un modelo Faster R-CNN para detección multiclase.

## 📋 Descripción

Este proyecto implementa un pipeline completo de entrenamiento y evaluación para detección de defectos industriales con las siguientes características:

- **Arquitectura**: ResNet-18 (preentrenado en ImageNet) + Faster R-CNN
- **Tarea**: Detección multiclase de defectos (6 categorías + NORMAL)
- **Categorías de defectos**:
  - NORMAL (sin defectos)
  - ROTURA_FRACTURA
  - PERFORACIONES
  - RAYONES_ARANAZOS
  - DEFORMACIONES
  - CONTAMINACION

- **Dataset**: Formato COCO con anotaciones en bounding boxes
- **Métricas**: mAP (mean Average Precision), Precision, Recall por clase

## 🗂️ Estructura de Archivos

```
.
├── industrial_defects_dataset.py    # Dataset loader (COCO format)
├── train_resnet18_fasterrcnn.py    # Script de entrenamiento
├── evaluate_model.py                # Evaluación con métricas mAP
├── visualize_predictions.py         # Visualización de predicciones
├── plot_training_metrics.py         # Gráficas de métricas de entrenamiento
├── run_pipeline.sh                  # Script para ejecutar pipeline completo
└── README.md                        # Este archivo
```

## 📊 Estadísticas del Dataset

### Train Set (715 imágenes)
- NORMAL: 210 imágenes
- ROTURA_FRACTURA: 118 imágenes
- PERFORACIONES: 106 imágenes
- RAYONES_ARANAZOS: 105 imágenes
- DEFORMACIONES: 94 imágenes
- CONTAMINACION: 85 imágenes

### Val Set (102 imágenes)
- NORMAL: 30 imágenes
- ROTURA_FRACTURA: 17 imágenes
- PERFORACIONES: 15 imágenes
- RAYONES_ARANAZOS: 15 imágenes
- DEFORMACIONES: 13 imágenes
- CONTAMINACION: 13 imágenes

### Test Set (205 imágenes)
- NORMAL: 60 imágenes
- ROTURA_FRACTURA: 34 imágenes
- RAYONES_ARANAZOS: 32 imágenes
- PERFORACIONES: 31 imágenes
- DEFORMACIONES: 26 imágenes
- CONTAMINACION: 24 imágenes

## 🚀 Instalación

### Requisitos

```bash
# PyTorch (con CUDA si está disponible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Otras dependencias
pip install numpy matplotlib pillow tqdm
```

### Verificar instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 💻 Uso

### Opción 1: Pipeline Completo (Recomendado)

Ejecutar el pipeline completo de entrenamiento, evaluación y visualización:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Este script ejecutará automáticamente:
1. Entrenamiento del modelo
2. Visualización de métricas de entrenamiento
3. Evaluación en conjunto de test
4. Visualización de predicciones en imágenes de test

### Opción 2: Ejecución Manual por Pasos

#### 1. Entrenamiento

```bash
python train_resnet18_fasterrcnn.py \
    --dataset-path curated_dataset_splitted_20251101_provisional_1st_version \
    --epochs 20 \
    --batch-size 4 \
    --lr 0.005 \
    --num-workers 4 \
    --pretrained-backbone \
    --output-dir results/training
```

**Parámetros principales:**
- `--dataset-path`: Ruta al dataset
- `--epochs`: Número de épocas (default: 20)
- `--batch-size`: Tamaño del batch (default: 4)
- `--lr`: Learning rate inicial (default: 0.005)
- `--pretrained-backbone`: Usar ResNet-18 preentrenado en ImageNet
- `--lr-step-size`: Reducir LR cada N épocas (default: 5)
- `--lr-gamma`: Factor de reducción de LR (default: 0.1)

**Outputs:**
- `results/training/resnet18_fasterrcnn_TIMESTAMP/`
  - `config.json`: Configuración del experimento
  - `training_history.json`: Métricas por época
  - `checkpoints/`:
    - `best_checkpoint.pth`: Mejor modelo (menor val_loss)
    - `last_checkpoint.pth`: Último checkpoint
    - `checkpoint_epoch_N.pth`: Checkpoints periódicos

#### 2. Visualizar Métricas de Entrenamiento

```bash
python plot_training_metrics.py \
    --history-path results/training/resnet18_fasterrcnn_TIMESTAMP/training_history.json
```

Genera gráficas de:
- Pérdida total (train/val)
- Pérdida del clasificador
- Pérdida de regresión de bbox
- Pérdida de objectness (RPN)
- Pérdida de RPN bbox regression
- Learning rate schedule

#### 3. Evaluación en Test Set

```bash
python evaluate_model.py \
    --checkpoint results/training/resnet18_fasterrcnn_TIMESTAMP/checkpoints/best_checkpoint.pth \
    --dataset-path curated_dataset_splitted_20251101_provisional_1st_version \
    --batch-size 4 \
    --score-threshold 0.5 \
    --iou-threshold 0.5
```

**Parámetros:**
- `--checkpoint`: Ruta al checkpoint del modelo
- `--score-threshold`: Umbral de confianza para filtrar predicciones (default: 0.5)
- `--iou-threshold`: Umbral de IoU para considerar True Positive (default: 0.5)

**Métricas calculadas:**
- **mAP** (mean Average Precision): Métrica principal
- **AP por clase**: Average Precision para cada categoría
- **Precision por clase**: Precisión final
- **Recall por clase**: Recall final

**Output:**
- `test_evaluation_results.json`: Resultados en formato JSON

#### 4. Visualizar Predicciones

```bash
python visualize_predictions.py \
    --checkpoint results/training/resnet18_fasterrcnn_TIMESTAMP/checkpoints/best_checkpoint.pth \
    --dataset-path curated_dataset_splitted_20251101_provisional_1st_version \
    --split test \
    --num-images 20 \
    --random \
    --score-threshold 0.5
```

**Parámetros:**
- `--split`: Conjunto a visualizar (train/val/test)
- `--num-images`: Número de imágenes a visualizar (-1 para todas)
- `--random`: Seleccionar imágenes aleatoriamente
- `--score-threshold`: Umbral de confianza

**Output:**
- `visualizations_test/`: Imágenes con predicciones y ground truth lado a lado

## 📈 Interpretación de Resultados

### Métricas de Entrenamiento

Durante el entrenamiento, se monitorizan las siguientes pérdidas:

1. **loss_classifier**: Error en la clasificación de objetos detectados
2. **loss_box_reg**: Error en la regresión de bounding boxes
3. **loss_objectness**: Error de la RPN en detectar si hay objetos
4. **loss_rpn_box_reg**: Error de la RPN en ajustar propuestas de cajas

Una buena convergencia se observa cuando:
- Las pérdidas disminuyen gradualmente
- La pérdida de validación sigue la pérdida de entrenamiento
- No hay overfitting (val_loss aumenta mientras train_loss baja)

### Métricas de Evaluación (mAP)

- **mAP > 0.7**: Excelente rendimiento
- **mAP 0.5-0.7**: Buen rendimiento
- **mAP 0.3-0.5**: Rendimiento aceptable
- **mAP < 0.3**: Necesita mejora

**Nota**: El mAP depende del umbral de IoU. IoU=0.5 es estándar para COCO.

## 🔧 Hiperparámetros Recomendados

### Para Dataset Pequeño (<1000 imágenes)

```bash
--epochs 30
--batch-size 4
--lr 0.005
--lr-step-size 10
--pretrained-backbone  # IMPORTANTE: siempre usar preentrenado
```

### Para Dataset Mediano (1000-5000 imágenes)

```bash
--epochs 25
--batch-size 8
--lr 0.005
--lr-step-size 8
```

### Si tienes problemas de memoria GPU

```bash
--batch-size 2  # Reducir batch size
--num-workers 2  # Reducir workers
```

## 🎯 Próximos Pasos para tu TFG

### 1. Experimentos Adicionales con CNNs Clásicas

#### EfficientNet

Crear un script similar pero con EfficientNet como backbone:

```python
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model_efficientnet_fasterrcnn(num_classes):
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Adaptar para Faster R-CNN...
```

#### Otros backbones clásicos a probar:
- ResNet-50 (más profundo)
- MobileNetV3 (más ligero)
- VGG-16 (arquitectura clásica)

### 2. Comparación con Vision Transformers

Para tu línea de investigación principal, el siguiente paso sería:

#### DINOv2 + Detection Head

```python
# Ejemplo conceptual
from transformers import AutoModel

backbone = AutoModel.from_pretrained("facebook/dinov2-base")
# Adaptar para detección...
```

**Ventajas de ViT vs CNN:**
- Mejor capacidad de atención global
- Mejores features para defectos pequeños
- Transfer learning más efectivo

### 3. Análisis Comparativo

Crear una tabla comparativa con:

| Modelo | Backbone | mAP | Params | Inference Time | GPU Memory |
|--------|----------|-----|--------|----------------|------------|
| ResNet-18 + Faster R-CNN | ResNet-18 | ? | 11M | ? | ? |
| ResNet-50 + Faster R-CNN | ResNet-50 | ? | 25M | ? | ? |
| EfficientNet + Faster R-CNN | EfficientNet-B0 | ? | 5M | ? | ? |
| DINOv2 + Detection Head | ViT-B | ? | 86M | ? | ? |

### 4. Mejoras del Dataset

- **Augmentation adicional**: rotaciones, flips, cambios de brillo
- **Balanceo de clases**: técnicas de re-sampling o loss weighting
- **Detección de falsos positivos**: análisis de errores comunes

### 5. Análisis de Errores

Crear un notebook para:
- Visualizar casos donde el modelo falla
- Analizar confusiones entre categorías
- Identificar patrones en errores (tamaño, ubicación, etc.)

## 📝 Notas Importantes

### Sobre ResNet-18 vs Hugging Face

El código inicial que mencionaste usa `AutoModelForImageClassification`, que es para **clasificación de imágenes**, no detección de objetos. La diferencia es:

- **Clasificación**: Una etiqueta por imagen (ej: "esta imagen contiene un perro")
- **Detección**: Múltiples objetos con ubicación (ej: "hay un perro en [x,y,w,h] y un gato en [x2,y2,w2,h2]")

Para detección, usamos:
1. **Backbone** (ResNet-18): Extrae features de la imagen
2. **RPN** (Region Proposal Network): Propone regiones candidatas
3. **ROI Head**: Clasifica y refina las regiones

### Normalización de Imágenes

Usamos la normalización estándar de ImageNet:
```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

Esto es importante porque ResNet-18 fue preentrenado con estas estadísticas.

### Formato de Anotaciones

El dataset usa formato COCO con bounding boxes en formato `[x, y, width, height]`, que se convierten a `[x_min, y_min, x_max, y_max]` para PyTorch.

## 🐛 Troubleshooting

### Error: CUDA out of memory
```bash
# Reducir batch size
--batch-size 2

# O usar CPU
CUDA_VISIBLE_DEVICES="" python train_resnet18_fasterrcnn.py ...
```

### Error: Invalid bbox (width or height <= 0)
El dataset loader filtra automáticamente bboxes inválidos. Verifica que tus anotaciones sean correctas.

### Pérdida no converge
- Verificar learning rate (probar 0.001 o 0.01)
- Verificar que el dataset esté correctamente cargado
- Aumentar número de épocas
- Verificar que el backbone esté preentrenado

### mAP muy bajo (<0.2)
- Aumentar épocas de entrenamiento
- Verificar score_threshold (probar 0.3)
- Revisar quality del dataset (anotaciones correctas)
- Usar backbone preentrenado

## 📚 Referencias

- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2015)
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2016)
- **TorchVision Detection**: https://pytorch.org/vision/stable/models.html#object-detection
- **COCO Format**: https://cocodataset.org/#format-data

## 📧 Contacto

Para dudas sobre el código o sugerencias de mejora, no dudes en contactar.

---

**Éxito con tu TFG! 🎓🚀**