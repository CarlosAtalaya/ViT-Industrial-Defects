# Baseline CNN: EfficientNet-B0 + Faster R-CNN

Este directorio contiene la implementación del modelo base basado en redes convolucionales (CNN) para la comparativa con arquitecturas ViT.

## 🎯 Objetivo
Proporcionar un punto de referencia (*baseline*) utilizando una arquitectura eficiente estándar en la industria para evaluar la ganancia de rendimiento de los modelos DINOv3/DEIMv2.

## 🏗️ Arquitectura
- **Backbone:** EfficientNet-B0 (Preentrenado en ImageNet).
- **Detector:** Faster R-CNN con Feature Pyramid Network (FPN).
- **Optimizador:** AdamW.

## 🚀 Ejecución
Para entrenar y evaluar el modelo completo:

```bash
bash run_pipeline.sh

El script run_pipeline.sh orquesta:

Entrenamiento (train_efficientnet_fasterrcnn.py).

Diagnóstico de curvas de aprendizaje (diagnose_model.py).

Evaluación de métricas COCO (evaluate_model.py).