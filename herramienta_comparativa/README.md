# 🔬 Dashboard de Comparación de Arquitecturas

Herramienta interactiva para visualizar y comparar los resultados de experimentación del TFG sobre **Detección de Defectos Industriales con Vision Transformers**.

## 🚀 Uso Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Lanzar el dashboard
streamlit run dashboard.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📋 Contenido del Dashboard

### 🏠 Inicio
- Contexto del proyecto y metodología de investigación
- Descripción detallada de cada arquitectura evaluada
- Información especial sobre DEIMv2 y Vision Transformers

### 📜 Línea Temporal
- Evolución cronológica de las 3 fases de experimentación:
  - **Fase 1 (Octubre 2024)**: Baseline con CNNs
  - **Fase 2 (Noviembre 2024)**: Exploración de Vision Transformers
  - **Fase 3 (Diciembre 2024)**: Validación experimental

### 🔬 Explorador
- Análisis detallado de cada experimento individual
- Configuración de entrenamiento y mejor checkpoint
- Métricas de evaluación: AP, Precision y Recall por clase
- Curvas de entrenamiento

### 📊 Comparativa
- Comparación directa entre arquitecturas
- Filtros: Todos, Mejores por arquitectura, Solo 1024x1024
- Gráficos de mAP, AP, Precision y Recall por clase

### 📝 Conclusiones
- Tabla resumen de todos los experimentos
- Análisis del impacto de resolución en CNNs vs ViTs
- Hallazgos principales y recomendaciones

## 📁 Estructura de Datos

```
data/
├── experiments_metadata.json    # Metadatos de todos los experimentos
├── fase1_baseline/              # CNNs con resolución nativa
│   ├── resnet18_nativa/         # mAP: 0.077
│   └── efficientnet_nativa/     # mAP: 0.162 ⭐ (mejor EfficientNet)
├── fase2_vit/                   # Vision Transformers
│   ├── deimv2_640_87ep/         # mAP: 0.499
│   ├── deimv2_1024_80ep/        # mAP: 0.624
│   ├── deimv2_1024_120ep/       # mAP: 0.766
│   └── deimv2_1024_300ep/       # mAP: 0.785 ⭐ (mejor global)
└── fase3_comparacion_justa/     # CNNs @ 1024x1024
    ├── resnet18_1024/           # mAP: 0.080 ⭐ (mejor ResNet)
    └── efficientnet_1024/       # mAP: 0.122 (peor que nativa)
```

## 📊 Resultados Principales

| Arquitectura | Mejor Configuración | mAP@0.5 |
|--------------|---------------------|---------|
| ResNet-18 | 1024x1024 | 0.080 |
| EfficientNet-B0 | Nativa | 0.162 |
| **DEIMv2 (ViT)** | **1024x1024, 300ep** | **0.785** ⭐ |

**Conclusión:** La arquitectura Vision Transformer (DEIMv2) supera significativamente a las CNNs tradicionales para la detección de defectos industriales.

## 📦 Exportar Datos para Herramienta Independiente

Para hacer la herramienta completamente independiente del repositorio, exporta todos los datos necesarios:

```bash
python3 export_data.py
```

Este script exporta:
- **Imágenes seleccionadas** → `data/images_selected/`
- **Predicciones JSON** → `data/predictions/` (resnet18_predictions.json, efficientnet_predictions.json, deimv2_predictions.json)
- **Ground Truth** → `data/ground_truth.json` (solo anotaciones de imágenes seleccionadas)

Una vez exportados los datos, el dashboard funcionará de forma independiente usando los datos en `data/` en lugar de buscar en el repositorio completo.

## 🛠️ Requisitos

- Python 3.8+
- Streamlit >= 1.28.0
- Pandas >= 2.0.0
- Plotly >= 5.18.0
- Pillow >= 10.0.0
- Matplotlib >= 3.7.0

---
*TFG 2025-26 - Detección de Defectos Industriales con Vision Transformers*
