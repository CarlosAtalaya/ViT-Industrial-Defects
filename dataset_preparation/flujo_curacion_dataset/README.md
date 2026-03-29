# Flujo de Curación del Dataset - Guía de Scripts

Este directorio contiene todos los scripts organizados paso a paso para reproducir el proceso completo de curación del dataset de detección de defectos industriales.

**Descarga de VISION-Datasets y MVTec AD:** [../DESCARGA_DATASETS_ORIGEN.md](../DESCARGA_DATASETS_ORIGEN.md).

## Estructura del Flujo

```
flujo_curacion_dataset/
├── README.md                               # Este archivo
├── etapa1_exploracion/                     # Exploración de datasets originales
│   ├── 01_explorar_vision_dataset.py
│   ├── 02_analizar_mvtec_dataset.py
│   ├── 03_taxonomia_propuesta.txt
│   └── README.md
├── etapa2_curacion_inicial/                # Curación y unificación
│   ├── 01_dataset_curator.py
│   └── README.md
├── etapa3_analisis_exhaustivo/             # Análisis de calidad
│   ├── 01_analisis_profundo_dataset.py
│   └── README.md
├── etapa4_recuracion_final/                # Pipeline final de curación
│   ├── 01_recurate_dataset.py              # Subetapa 4.1: Re-curación
│   ├── 02_unify_taxonomy.py                # Subetapa 4.2: Unificación
│   ├── 03_balance_dataset.py               # Subetapa 4.3: Balanceo
│   ├── 04_create_splits.py                 # Subetapa 4.4: Splits
│   └── README.md
├── etapa5_analisis_final/                  # ✅ Análisis exhaustivo del dataset final
│   ├── 01_analizar_tamanios_imagenes.py    # Análisis de resoluciones
│   ├── 02_analisis_balance_dataset.py      # Balance por categoría y split
│   ├── 03_inspeccionar_dataset.py          # Inspección COCO detallada
│   └── README.md
└── outputs/                                # Carpeta para resultados
```

---

## Etapa 1: Exploración de Datasets Originales

**Objetivo:** Analizar los datasets VISION-Datasets y MVTec AD para identificar categorías, defectos y posibilidades de unificación.

### Paso 1.1: Explorar VISION-Datasets

```bash
cd etapa1_exploracion/
python 01_explorar_vision_dataset.py
```

**Requiere:** 
- Path al dataset VISION-Datasets (modificar `dataset_path` en el script)
- Dependencias: `opencv-python`, `matplotlib`, `numpy`

**Genera:**
- `vision_dataset_report.json` - Reporte completo del dataset
- Visualizaciones de muestras por componente

### Paso 1.2: Analizar MVTec AD

```bash
python 02_analizar_mvtec_dataset.py
```

**Requiere:**
- Path al dataset MVTec AD (modificar `MVTEC_PATH` en el script)
- El archivo `samples.json` en el directorio del dataset

**Genera:**
- `mvtec_ad_report.json` - Reporte completo
- `mvtec_analysis_charts.png` - Visualizaciones

### Paso 1.3: Revisar Taxonomía Propuesta

El archivo `03_taxonomia_propuesta.txt` contiene:
- Mapeo de defectos entre datasets
- Categorías MVTec y componentes VISION seleccionados
- Taxonomía unificada de 5 tipos de defectos

---

## Etapa 2: Curación Inicial del Dataset

**Objetivo:** Crear el dataset curado unificando imágenes de ambas fuentes.

### Paso 2.1: Ejecutar Curador

```bash
cd etapa2_curacion_inicial/
python 01_dataset_curator.py \
  --vision-path /path/to/VISION-Datasets \
  --mvtec-path /path/to/mvtec-ad \
  --output-path ../outputs/curated_dataset_inicial
```

**Requiere:**
- Paths a ambos datasets originales
- Dependencias: `Pillow`, `uuid`, `argparse`

**Genera:**
- `images/` - Imágenes filtradas en formato PNG
- `annotations.coco.json` - Anotaciones COCO unificadas
- `metadata/dataset_info.json` - Metadatos del proceso

### Configuración del Curador

El script tiene parámetros configurables:

```python
# Defectos objetivo (en el script)
defectos_objetivo = {
    "ROTURA_FRACTURA": {...},
    "CONTAMINACION": {...},
    "RAYONES_ARANAZOS": {...},
    "PERFORACIONES": {...},
    "DEFORMACIONES": {...}
}

# Categorías MVTec a incluir
categorias_mvtec = ["transistor", "metal_nut", "cable", "capsule", "hazelnut"]

# Componentes VISION a incluir
componentes_vision = ["PCB_1", "PCB_2", "Console", "Electronics", "Cable", "Lens"]
```

---

## Etapa 3: Análisis Exhaustivo del Dataset Curado

**Objetivo:** Analizar la calidad del dataset curado, detectar problemas y proyectar distribución post-unificación.

### Paso 3.1: Ejecutar Análisis Profundo

```bash
cd etapa3_analisis_exhaustivo/
python 01_analisis_profundo_dataset.py \
  --dataset ../outputs/curated_dataset_inicial \
  --output ../outputs/analisis_resultados
```

**Requiere:**
- Dataset curado de la Etapa 2
- Dependencias: `pandas`, `imagehash`, `Pillow`, `matplotlib`, `seaborn`

**Genera:**
- `analysis_report.json` - Reporte JSON completo
- `category_distribution.csv` y `.png`
- `source_distribution.csv` y `.png`
- `unified_distribution_preview.csv`
- `duplicates.csv` - Pares de imágenes duplicadas
- `consistency_issues.csv` - Problemas de consistencia
- `image_resolutions.csv` - Resoluciones de todas las imágenes
- `resolution_scatter.png`

### Análisis Realizados

| Análisis | Descripción |
|----------|-------------|
| `analyze_category_distribution()` | Distribución de categorías originales |
| `analyze_source_distribution()` | Balance MVTec vs VISION |
| `detect_duplicates()` | Detección con perceptual hashing |
| `analyze_image_quality()` | Resoluciones, formatos, corrupción |
| `analyze_annotation_consistency()` | Verificación de coherencia |
| `analyze_defect_mapping()` | Proyección a taxonomía unificada |

---

## Dependencias Completas

```bash
# Crear entorno virtual
python -m venv venv_curacion
source venv_curacion/bin/activate  # Linux/Mac
# venv_curacion\Scripts\activate   # Windows

# Instalar dependencias
pip install numpy pandas Pillow opencv-python matplotlib seaborn imagehash
```

**requirements.txt:**
```
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
imagehash>=4.3.1
```

---

## Flujo Completo de Ejecución

```bash
# 1. Activar entorno
source venv_curacion/bin/activate

# 2. Etapa 1 - Exploración
cd etapa1_exploracion/
python 01_explorar_vision_dataset.py
python 02_analizar_mvtec_dataset.py
# Revisar outputs y 03_taxonomia_propuesta.txt

# 3. Etapa 2 - Curación
cd ../etapa2_curacion_inicial/
python 01_dataset_curator.py \
  --vision-path /path/to/VISION-Datasets \
  --mvtec-path /path/to/mvtec-ad \
  --output-path ../outputs/curated_dataset

# 4. Etapa 3 - Análisis
cd ../etapa3_analisis_exhaustivo/
python 01_analisis_profundo_dataset.py \
  --dataset ../outputs/curated_dataset \
  --output ../outputs/analisis

# 5. Revisar resultados en outputs/
```

---

## Resultados Esperados

Tras completar las 3 etapas:

| Métrica | Valor Esperado |
|---------|----------------|
| Total imágenes | ~1,900 |
| Imágenes VISION | ~14% |
| Imágenes MVTec | ~86% |
| Categorías originales | 15 |
| Categorías unificadas | 6 |
| Duplicados | ~50 pares |
| Imágenes corruptas | 0 |

---

## Etapa 4: Recuración Final y Preparación del Dataset

**Objetivo:** Pipeline completo que transforma el dataset curado inicial en un dataset listo para entrenamiento de Vision Transformers.

### Paso 4.1: Re-curación del Dataset

```bash
cd etapa4_recuracion_final/
python 01_recurate_dataset.py \
  --source ../outputs/curated_dataset \
  --output ../outputs/curated_v1 \
  --duplicates ../outputs/analisis/duplicates.csv
```

**Operaciones:**
- Elimina categoría "hazelnut" (fuera del scope electrónico)
- Remueve duplicados detectados en Etapa 3
- Filtra solo componentes relevantes

**Resultado:** 1,907 → 1,393 imágenes

### Paso 4.2: Unificación de Taxonomía

```bash
python 02_unify_taxonomy.py \
  --source ../outputs/curated_v1 \
  --output ../outputs/curated_v2
```

**Operaciones:**
- Mapea 15 categorías originales → 6 categorías unificadas
- Aplica esquema híbrido COCO-compatible
- Añade metadata de trazabilidad

**Taxonomía Final:**
| ID | Categoría | Criticidad |
|----|-----------|------------|
| 0 | NORMAL | NINGUNA |
| 1 | DEFORMACIONES | ALTA |
| 2 | ROTURA_FRACTURA | CRÍTICA |
| 3 | RAYONES_ARANAZOS | MEDIA |
| 4 | PERFORACIONES | CRÍTICA |
| 5 | CONTAMINACION | ALTA |

### Paso 4.3: Balanceo del Dataset

```bash
python 03_balance_dataset.py \
  --source ../outputs/curated_v2 \
  --output ../outputs/curated_v3_balanced \
  --seed 42
```

**Operaciones:**
- Under-sampling de NORMAL (1,039 → 300)
- Over-sampling con augmentación conservadora
- Logra ratio máx/min de 2.08:1 (óptimo < 3:1)

### Paso 4.4: Creación de Splits

```bash
python 04_create_splits.py \
  --source ../outputs/curated_v3_balanced \
  --output ../outputs/curated_FINAL \
  --seed 42
```

**Operaciones:**
- Splits estratificados: Train 70% / Val 10% / Test 20%
- Estratificación dual (categoría + dataset origen)
- Validación Chi-cuadrado y no-leakage

**Dataset Final:**
```
curated_FINAL/
├── train/images/ + train.json    # 715 imágenes
├── val/images/ + val.json        # 102 imágenes
├── test/images/ + test.json      # 205 imágenes
└── metadata/
```

---

## Dependencias Completas

```bash
# Crear entorno virtual
python -m venv venv_curacion
source venv_curacion/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
imagehash>=4.3.1
pycocotools>=2.0.7
scikit-learn>=1.3.0
albumentations>=1.3.1
scipy>=1.11.0
```

---

## Flujo Completo de Ejecución

```bash
# 1. Activar entorno
source venv_curacion/bin/activate

# 2. Etapa 1 - Exploración
cd etapa1_exploracion/
python 01_explorar_vision_dataset.py
python 02_analizar_mvtec_dataset.py

# 3. Etapa 2 - Curación Inicial
cd ../etapa2_curacion_inicial/
python 01_dataset_curator.py --vision-path ... --mvtec-path ... --output-path ../outputs/curated_dataset

# 4. Etapa 3 - Análisis
cd ../etapa3_analisis_exhaustivo/
python 01_analisis_profundo_dataset.py --dataset ../outputs/curated_dataset --output ../outputs/analisis

# 5. Etapa 4 - Recuración Final (Pipeline completo)
cd ../etapa4_recuracion_final/
python 01_recurate_dataset.py --source ../outputs/curated_dataset --output ../outputs/curated_v1 --duplicates ../outputs/analisis/duplicates.csv
python 02_unify_taxonomy.py --source ../outputs/curated_v1 --output ../outputs/curated_v2
python 03_balance_dataset.py --source ../outputs/curated_v2 --output ../outputs/curated_v3_balanced --seed 42
python 04_create_splits.py --source ../outputs/curated_v3_balanced --output ../outputs/curated_FINAL --seed 42

# 6. Etapa 5 - Análisis Final
cd ../etapa5_analisis_final/

# Análisis de tamaños de imágenes (recomendación de resolución)
python 01_analizar_tamanios_imagenes.py

# Inspección detallada por split
python 03_inspeccionar_dataset.py \
  --coco-json ../outputs/curated_FINAL/train/train.json \
  --images-dir ../outputs/curated_FINAL/train/images \
  --out-dir ../outputs/analisis_final/inspect_train

python 03_inspeccionar_dataset.py \
  --coco-json ../outputs/curated_FINAL/val/val.json \
  --images-dir ../outputs/curated_FINAL/val/images \
  --out-dir ../outputs/analisis_final/inspect_val

python 03_inspeccionar_dataset.py \
  --coco-json ../outputs/curated_FINAL/test/test.json \
  --images-dir ../outputs/curated_FINAL/test/images \
  --out-dir ../outputs/analisis_final/inspect_test

# Análisis exhaustivo de balance
python 02_analisis_balance_dataset.py \
  --dataset-root ../outputs/analisis_final \
  --output-dir ../outputs/analisis_final/balance_report

# 7. Dataset final listo + análisis completo
```

---

## Etapa 5: Análisis Exhaustivo del Dataset Final

**Objetivo:** Validar la calidad del dataset final y generar métricas exhaustivas antes del entrenamiento.

### Paso 5.1: Análisis de Tamaños de Imágenes

```bash
cd etapa5_analisis_final/
python 01_analizar_tamanios_imagenes.py
```

**Genera:**
- Histogramas de distribución de resoluciones
- Scatter plot ancho vs alto por split
- Recomendación de resolución para ViT (múltiplos de 14)

### Paso 5.2: Inspección Detallada por Split

```bash
python 03_inspeccionar_dataset.py \
  --coco-json /path/to/train.json \
  --images-dir /path/to/train/images \
  --out-dir ./inspect_train
```

**Genera por cada split:**
- `dataset_report.txt` - Reporte general
- `images_table.csv` - Info de todas las imágenes
- `annotations_table.csv` - Info de todas las anotaciones
- `bboxes_stats.csv` - Estadísticas de bounding boxes
- `hist_bbox_area.png` - Histograma de áreas
- `hist_bbox_aspect_ratio.png` - Histograma de aspect ratios
- `hist_short_edge.png` - Histograma de lado más corto

### Paso 5.3: Análisis de Balance

```bash
python 02_analisis_balance_dataset.py \
  --dataset-root ../outputs/analisis_final \
  --output-dir ./balance_report
```

**Genera:**
- Distribución de categorías por split
- Estadísticas de bboxes (tamaño, area, aspect ratio)
- Detección de bboxes pequeños (< 32px)
- Proporción de augmentación
- Distribución por dataset de origen
- Reporte crítico con problemas detectados

### Métricas Clave Verificadas

| Métrica | Valor | Estado |
|---------|-------|--------|
| Total imágenes | 1,022 | ✅ |
| Ratio máx/min | 2.08:1 | ✅ Óptimo |
| Bboxes < 32px | ~20% | ⚠️ Ajustar anchors |
| Archivos faltantes | 0 | ✅ |
| Leakage entre splits | Ninguno | ✅ |

---

## Resultados Finales

| Etapa | Operación | Imágenes |
|-------|-----------|----------|
| Original | Datasets combinados | ~9,142 |
| Etapa 2 | Filtrado inicial | 1,907 |
| Etapa 4.1 | Re-curación | 1,393 |
| Etapa 4.3 | Balanceo | 1,354 + 368 aug |
| **Etapa 4.4** | **Splits** | **1,022** |
| Etapa 5 | Análisis final | Validado ✅ |

### Dataset Final

```
Total: 1,022 imágenes
├── Train: 715 (70%) - 944 anotaciones
├── Val: 102 (10%) - 145 anotaciones
└── Test: 205 (20%) - 265 anotaciones

Categorías: 6 (NORMAL, DEFORMACIONES, ROTURA_FRACTURA, 
               RAYONES_ARANAZOS, PERFORACIONES, CONTAMINACION)

Validaciones:
├── Ratio máx/min: 2.08:1 ✅
├── Chi² test splits: p > 0.99 ✅
├── Leakage: Ninguno ✅
├── Integridad: 100% ✅
└── Augmentación: 36% ✅
```

### Recomendaciones para Entrenamiento

1. **Resolución:** 1400×1400 px (múltiplo de 14 para ViT)
2. **Anchors:** Añadir tamaño 16px para bboxes pequeños
3. **Arquitectura:** DINOv2/DINOv3 recomendado para objetos pequeños

---

**Documento actualizado:** Diciembre 2025  
**Versión:** 3.0 (Pipeline completo con análisis final)

