# Etapa 5: Análisis Exhaustivo del Dataset Curado Final

## Descripción

Esta etapa representa el análisis científico riguroso del dataset final curado y preparado para entrenamiento de modelos de detección de defectos. El objetivo es validar la calidad del dataset, identificar posibles problemas y documentar todas las métricas relevantes para futuras referencias y reproducibilidad.

## Prerrequisitos

- Python 3.8+
- Dataset curado final: `curated_dataset_splitted_20251101_provisional_1st_version/`

### Dependencias

```bash
pip install pandas numpy matplotlib seaborn pillow
```

## Scripts Incluidos

### 1. `01_analizar_tamanios_imagenes.py`

**Propósito:** Analiza la distribución de tamaños de imágenes para determinar la resolución óptima de entrenamiento.

**Análisis realizados:**
- Distribución de ancho, alto y lado más corto
- Cálculo de percentiles (P10, P25, P50, P75, P90)
- Aspect ratios de las imágenes
- Identificación de imágenes extremas (muy pequeñas/grandes)
- Recomendación de resolución compatible con ViT (múltiplos de 14)

**Uso:**
```bash
python 01_analizar_tamanios_imagenes.py
```

**Outputs:**
- `analysis_imagessizes_plots/image_size_distributions.png`
- `analysis_imagessizes_plots/width_vs_height_scatter.png`
- Recomendación de resolución para `Resize` transform

---

### 2. `02_analisis_balance_dataset.py`

**Propósito:** Análisis exhaustivo del balance y distribución del dataset entre splits.

**Análisis realizados:**
- Distribución de categorías por split (train/val/test)
- Tamaños de imágenes: width, height, area, short_edge
- Tamaños de bounding boxes: width, height, area, aspect_ratio
- Detección de problemas críticos (bboxes < 32px)
- Proporción de imágenes augmentadas vs originales
- Distribución por dataset de origen (MVTec/VISION)

**Uso:**
```bash
python 02_analisis_balance_dataset.py \
    --dataset-root ../path/to/outputs/dataset_info \
    --output-dir ./balanced-dataset-analysis
```

**Outputs:**
- `category_distribution.csv` / `.png`
- `category_proportions.png`
- `bbox_distribution.png`
- `bbox_stats.csv`
- `image_sizes_distribution.png`
- `image_sizes_stats.csv`
- `augmentation_distribution.png`
- `augmentation_stats.csv`
- `source_dataset_distribution.csv` / `.png`
- `BALANCE_REPORT.txt`
- `resumen_ejecutivo.txt`

---

### 3. `03_inspeccionar_dataset.py`

**Propósito:** Inspección detallada de cada split del dataset en formato COCO.

**Análisis realizados:**
- Verificación de existencia de archivos de imagen
- Estadísticas por categoría unificada
- Distribución de áreas y aspect ratios de bboxes
- Análisis de segmentaciones (presencia y área)
- Detección de imágenes augmentadas

**Uso:**
```bash
# Para cada split
python 03_inspeccionar_dataset.py \
    --coco-json ../path/to/train/train.json \
    --images-dir ../path/to/train/images \
    --out-dir ./inspect_train

python 03_inspeccionar_dataset.py \
    --coco-json ../path/to/val/val.json \
    --images-dir ../path/to/val/images \
    --out-dir ./inspect_val

python 03_inspeccionar_dataset.py \
    --coco-json ../path/to/test/test.json \
    --images-dir ../path/to/test/images \
    --out-dir ./inspect_test
```

**Outputs por split:**
- `dataset_report_{split}.txt` - Reporte general
- `images_table.csv` - Tabla de imágenes
- `images_with_area.csv` - Imágenes con área calculada
- `images_per_unified_category.csv` - Distribución por categoría
- `annotations_table.csv` - Tabla de anotaciones
- `bboxes_stats.csv` - Estadísticas de bounding boxes
- `hist_bbox_area.png` - Histograma de áreas de bbox
- `hist_bbox_aspect_ratio.png` - Histograma de aspect ratios
- `hist_short_edge.png` - Histograma de lado más corto

---

## Métricas Clave del Dataset Final

### Resumen General

| Métrica | Valor |
|---------|-------|
| **Total imágenes** | 1,022 |
| **Total anotaciones** | 1,354 |
| **Train** | 715 imgs (944 anns) |
| **Val** | 102 imgs (145 anns) |
| **Test** | 205 imgs (265 anns) |

### Distribución de Categorías

| Categoría | Train | Val | Test | Total |
|-----------|-------|-----|------|-------|
| PERFORACIONES | 242 | 26 | 60 | 328 |
| NORMAL | 210 | 30 | 60 | 300 |
| ROTURA_FRACTURA | 138 | 33 | 40 | 211 |
| DEFORMACIONES | 136 | 21 | 38 | 195 |
| RAYONES_ARANAZOS | 111 | 17 | 34 | 162 |
| CONTAMINACION | 107 | 18 | 33 | 158 |

### Estadísticas de Bounding Boxes

| Split | N BBoxes | Width (med) | Height (med) | Area (med) | AR (med) |
|-------|----------|-------------|--------------|------------|----------|
| Train | 944 | 222.4 px | 132.5 px | 6,082 px² | 1.00 |
| Val | 145 | 224.0 px | 112.0 px | 7,435 px² | 1.00 |
| Test | 265 | 314.0 px | 175.0 px | 7,914 px² | 1.00 |

### Tamaños de Imágenes

| Split | N Imgs | Width (mean) | Height (mean) | Area (mean) |
|-------|--------|--------------|---------------|-------------|
| Train | 715 | 1,647.5 px | 1,347.4 px | 3.09 MP |
| Val | 102 | 1,735.2 px | 1,456.7 px | 3.56 MP |
| Test | 205 | 1,703.7 px | 1,415.1 px | 3.37 MP |

### Augmentación

| Split | Original | Augmentado | % Augmentado |
|-------|----------|------------|--------------|
| Train | 458 (64.1%) | 257 (35.9%) | 35.9% |
| Val | 63 (61.8%) | 39 (38.2%) | 38.2% |
| Test | 133 (64.9%) | 72 (35.1%) | 35.1% |

---

## Problemas Detectados y Recomendaciones

### 1. Bounding Boxes Pequeños

**Problema:** ~20% de bboxes tienen dimensión < 32px

- Train: 23.4% width < 32px, 22.2% height < 32px
- Val: 16.6% width < 32px, 20.0% height < 32px
- Test: 17.0% width < 32px, 16.2% height < 32px

**Implicación:** Faster R-CNN usa anchors mínimos de 32px por defecto.

**Recomendación:**
```python
# Modificar AnchorGenerator para incluir anchors más pequeños
anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128, 256),),  # Añadir 16px
    aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)
)
```

### 2. Aspect Ratios Extremos

**Problema:** ~11-15% de bboxes tienen aspect ratio extremo (<0.25 o >4.0)

**Recomendación:**
- Usar Feature Pyramid Networks (FPN) para multi-scale detection
- Considerar Vision Transformers (DINOv2) que manejan mejor estos casos

### 3. Variabilidad de Resolución

**Problema:** Rango de resoluciones muy amplio (262x192 a 3840x3620 px)

**Recomendación:** 
- Resolución de entrenamiento: **1400×1400 px** (múltiplo de 14 para ViT)
- Utilizar padding adaptativo en lugar de distorsión

---

## Estructura de Salidas

```
Final_analysis_curated_dataset_1st_version/outputs/
├── analysis_imagessizes_plots/
│   ├── image_size_distributions.png
│   └── width_vs_height_scatter.png
│
├── balanced-dataset-analysis-20251114/
│   ├── augmentation_distribution.png
│   ├── augmentation_stats.csv
│   ├── BALANCE_REPORT.txt
│   ├── bbox_distribution.png
│   ├── bbox_stats.csv
│   ├── category_distribution.csv
│   ├── category_distribution.png
│   ├── category_proportions.png
│   ├── image_sizes_distribution.png
│   ├── image_sizes_stats.csv
│   ├── resumen_ejecutivo.txt
│   ├── source_dataset_distribution.csv
│   └── source_dataset_distribution.png
│
└── dataset_info/
    ├── inspect_train/
    │   ├── annotations_table.csv
    │   ├── bboxes_stats.csv
    │   ├── dataset_report_train.txt
    │   ├── hist_bbox_area.png
    │   ├── hist_bbox_aspect_ratio.png
    │   ├── hist_short_edge.png
    │   ├── images_per_unified_category.csv
    │   ├── images_table.csv
    │   └── images_with_area.csv
    ├── inspect_val/
    │   └── ... (mismos archivos)
    └── inspect_test/
        └── ... (mismos archivos)
```

---

## Conclusión

El análisis exhaustivo valida que el dataset curado está listo para entrenamiento con las siguientes características:

✅ **Balance de clases optimizado** (ratio 2.08:1)  
✅ **Estratificación validada** entre splits  
✅ **Trazabilidad completa** de origen y augmentación  
⚠️ **Considerar ajustes** de anchors para objetos pequeños  
⚠️ **Normalizar resolución** a 1400×1400 px para ViT  

El dataset es apto para entrenamiento de modelos de detección de defectos industriales con arquitecturas basadas en Vision Transformers (DINOv2, DINOv3) y detectores tradicionales (Faster R-CNN con anchors ajustados).

