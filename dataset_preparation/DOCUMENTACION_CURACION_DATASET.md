# Documentación: Proceso de Curación del Dataset para Detección de Defectos Industriales

**Proyecto:** TFG - Vision Transformers para Detección de Anomalías Industriales  
**Fecha de documentación:** Diciembre 2025  
**Dataset resultante:** `curated_dataset_splitted_20251101_provisional_1st_version/`

---

## Índice

1. [Introducción](#1-introducción)
2. [Etapa 1: Exploración de Datasets Originales](#2-etapa-1-exploración-de-datasets-originales)
3. [Etapa 2: Propuesta e Implementación de Curación Inicial](#3-etapa-2-propuesta-e-implementación-de-curación-inicial)
4. [Etapa 3: Análisis Exhaustivo del Dataset Curado](#4-etapa-3-análisis-exhaustivo-del-dataset-curado)
5. [Etapa 4: Recuración Final y Preparación del Dataset](#5-etapa-4-recuración-final-y-preparación-del-dataset)
6. [Etapa 5: Análisis Exhaustivo del Dataset Final](#6-etapa-5-análisis-exhaustivo-del-dataset-final)
7. [Resumen de Métricas y Resultados Finales](#7-resumen-de-métricas-y-resultados-finales)
8. [Estructura de Carpetas del Proyecto](#8-estructura-de-carpetas-del-proyecto)

---

## 1. Introducción

Este documento describe el proceso completo de curación de un dataset unificado para entrenamiento de modelos de visión (Vision Transformers) destinados a la detección de defectos industriales. El dataset final combina imágenes de dos fuentes públicas reconocidas:

- **VISION-Datasets**: Dataset supervisado con anotaciones COCO detalladas para componentes electrónicos industriales
- **MVTec AD**: Benchmark estándar para detección de anomalías industriales con máscaras pixel-level

### Objetivo del Dataset Curado

Crear un dataset unificado con:
- Taxonomía de defectos coherente y normalizada
- Formato de anotaciones COCO estándar
- Balance de clases optimizado para entrenamiento de ViT
- Trazabilidad completa del origen de cada imagen

---

## 2. Etapa 1: Exploración de Datasets Originales

**Carpeta:** `Analisis-datasets-VISION-mvtecad/`

### 2.1 Objetivo

Realizar un análisis exploratorio profundo de ambos datasets originales para:
- Identificar categorías y tipos de defectos disponibles
- Analizar estructuras de anotación y formatos
- Detectar correspondencias semánticas entre defectos de ambos datasets
- Evaluar la viabilidad de unificación

### 2.2 Scripts Utilizados

| Script | Propósito |
|--------|-----------|
| `dataset_exploration.py` | Explorador del dataset VISION-Datasets |
| `analyze_mvtec_dataset.py` | Analizador específico para MVTec AD |
| `mvtec_inspector.py` | Inspector detallado de categorías MVTec |

### 2.3 Análisis del Dataset VISION-Datasets

**Características identificadas:**
- **Total componentes:** 14 tipos diferentes
- **Total imágenes:** 3,788 (train: 1,760 | val: 2,028)
- **Tipos de defectos únicos:** 44 categorías
- **Formato:** COCO JSON con bounding boxes

**Componentes analizados:**

| Componente | Train | Val | Categorías de Defectos |
|------------|-------|-----|------------------------|
| PCB_1 | 94 | 80 | missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper |
| PCB_2 | 160 | 150 | defect1-defect7 |
| Console | 190 | 194 | Collision, Dirty, Gap, Scratch |
| Cable | 82 | 262 | break, thunderbolt |
| Lens | 132 | 126 | Fiber, Flash Particle, Hole, Surface Damage, Tear |
| Electronics | 72 | 62 | damage |
| Capacitor | 70 | 84 | 0 (sin defectos etiquetados específicamente) |
| ... | ... | ... | ... |

### 2.4 Análisis del Dataset MVTec AD

**Características identificadas:**
- **Total muestras:** 5,354
- **Categorías:** 15 tipos de objetos
- **Tipos de defectos:** 49 únicos
- **Splits:** train (3,629) | test (1,725)
- **Filosofía:** Semi-supervisado (train solo normales, test mixto)
- **Formato:** Máscaras pixel-level + etiquetas binarias

**Categorías principales:**

| Categoría | Total | Normal | Anomalías | Defectos |
|-----------|-------|--------|-----------|----------|
| transistor | 313 | 273 | 40 | bent_lead, cut_lead, damaged_case, misplaced |
| metal_nut | 335 | 242 | 93 | bent, color, flip, scratch |
| cable | 374 | 282 | 92 | bent_wire, cable_swap, cut_*, missing_* |
| capsule | 351 | 242 | 109 | crack, faulty_imprint, poke, scratch, squeeze |
| hazelnut | 501 | 431 | 70 | crack, cut, hole, print |

### 2.5 Mapeo de Defectos Identificado

Se identificaron correspondencias semánticas entre defectos de ambos datasets:

```
DEFECTOS PROPUESTOS PARA UNIFICACIÓN:
=====================================

1. ROTURA/FRACTURA (Criticidad: CRÍTICA)
   - VISION: break, defect
   - MVTec: broken, broken_large, crack

2. CONTAMINACIÓN (Criticidad: ALTA)
   - VISION: Dirty, impurities
   - MVTec: contamination, metal_contamination

3. RAYONES/ARAÑAZOS (Criticidad: MEDIA)
   - VISION: Scratch, s_scratch, t_scratch
   - MVTec: scratch, scratch_head

4. PERFORACIONES (Criticidad: CRÍTICA)
   - VISION: Hole, missing_hole
   - MVTec: hole, cut

5. DEFORMACIONES (Criticidad: ALTA)
   - VISION: short, spur
   - MVTec: bent, bent_lead, bent_wire
```

### 2.6 Outputs Generados

| Archivo | Descripción |
|---------|-------------|
| `vision_dataset_report.json` | Reporte completo de VISION-Datasets |
| `mvtec_ad_report.json` | Reporte completo de MVTec AD |
| `datasets_technical_summary.txt` | Resumen técnico comparativo |
| `detailed_categories_analysis.txt` | Análisis detallado de categorías |
| `posibles_categorias-defectos.txt` | Definición de taxonomía objetivo |

---

## 3. Etapa 2: Propuesta e Implementación de Curación Inicial

**Carpeta:** `curated_dataset_20250921_115859/`

### 3.1 Propuesta de Curación

Se elaboró un documento de propuesta (`Curation-dataset-proposal.md`) que define:

#### Taxonomía Unificada de 6 Categorías

```yaml
TAXONOMIA_DEFECTOS_V2:
  NORMAL:
    codigo: 0
    descripcion: "Componente sin defectos visibles"
    
  DEFORMACIONES:
    codigo: 1
    descripcion: "Alteración de geometría o forma original"
    criticidad: ALTA
    
  ROTURA_FRACTURA:
    codigo: 2
    descripcion: "Discontinuidad estructural, grietas o roturas"
    criticidad: CRITICA
    
  RAYONES_ARANAZOS:
    codigo: 3
    descripcion: "Marcas superficiales por abrasión o contacto"
    criticidad: MEDIA
    
  PERFORACIONES:
    codigo: 4
    descripcion: "Agujeros, cortes o ausencia de material"
    criticidad: CRITICA
    
  CONTAMINACION:
    codigo: 5
    descripcion: "Presencia de material extraño o suciedad"
    criticidad: ALTA
```

#### Filtros de Selección

**Categorías MVTec seleccionadas:**
- transistor
- metal_nut
- cable
- capsule
- hazelnut

**Componentes VISION seleccionados:**
- PCB_1, PCB_2
- Console
- Electronics
- Cable
- Lens

### 3.2 Implementación del Curador

El script `dataset_curator_20250921.py` implementa la clase `DatasetCurator` con las siguientes funcionalidades:

```python
class DatasetCurator:
    """
    Funcionalidades principales:
    - Carga y procesamiento de anotaciones MVTec (samples.json)
    - Carga y procesamiento de anotaciones VISION (COCO JSON)
    - Filtrado por defectos objetivo
    - Conversión de imágenes a PNG
    - Generación de anotaciones COCO unificadas
    - Gestión de metadatos y trazabilidad
    """
```

**Flujo de procesamiento:**

```
1. Setup estructura de salida
2. Cargar anotaciones MVTec AD
3. Filtrar muestras MVTec por categorías y defectos objetivo
4. Procesar y copiar imágenes MVTec
5. Cargar anotaciones VISION por componente
6. Filtrar anotaciones VISION por defectos objetivo
7. Procesar y copiar imágenes VISION
8. Generar annotations.coco.json unificado
9. Guardar metadatos del proceso
```

### 3.3 Resultados de Curación Inicial

**Estadísticas del dataset generado:**

| Métrica | Valor |
|---------|-------|
| **Total imágenes** | 1,907 |
| Imágenes VISION | 267 (14.0%) |
| Imágenes MVTec | 1,640 (86.0%) |
| **Total anotaciones** | 2,125 |
| **Total categorías** | 15 (etiquetas originales) |

**Distribución por tipo de defecto (categorías originales):**

| Categoría | Count | Porcentaje |
|-----------|-------|------------|
| normal | 1,470 | 69.18% |
| break | 128 | 6.02% |
| missing_hole | 124 | 5.84% |
| Scratch | 60 | 2.82% |
| Dirty | 60 | 2.82% |
| short | 47 | 2.21% |
| scratch | 46 | 2.16% |
| crack | 41 | 1.93% |
| spur | 38 | 1.79% |
| Hole | 28 | 1.32% |
| bent | 25 | 1.18% |
| hole | 18 | 0.85% |
| cut | 17 | 0.80% |
| bent_wire | 13 | 0.61% |
| bent_lead | 10 | 0.47% |

### 3.4 Outputs Generados

| Archivo/Carpeta | Descripción |
|-----------------|-------------|
| `images/` | 1,907 imágenes en formato PNG |
| `annotations.coco.json` | Anotaciones COCO unificadas |
| `metadata/dataset_info.json` | Metadatos del proceso de curación |

---

## 4. Etapa 3: Análisis Exhaustivo del Dataset Curado

**Carpeta:** `curation-method_20251101/analysis_curated_dataset_20251101/`

### 4.1 Objetivo

Realizar un análisis científico riguroso del dataset curado para:
- Validar la calidad de las imágenes
- Detectar duplicados
- Verificar consistencia de anotaciones
- Proyectar la distribución tras unificación de taxonomía

### 4.2 Script de Análisis

El script `analysis_previous_curated_dataset_20250921.py` implementa la clase `DatasetDeepAnalyzer` con los siguientes análisis:

```python
class DatasetDeepAnalyzer:
    """
    Análisis implementados:
    - analyze_category_distribution(): Distribución de categorías
    - analyze_source_distribution(): Distribución por dataset origen
    - detect_duplicates(): Detección con perceptual hashing
    - analyze_image_quality(): Resoluciones, formatos, corrupción
    - analyze_annotation_consistency(): Verificación de coherencia
    - analyze_defect_mapping(): Proyección a taxonomía unificada
    - generate_visualizations(): Gráficos de análisis
    """
```

### 4.3 Resultados del Análisis

#### Resumen General

| Métrica | Valor |
|---------|-------|
| Total imágenes | 1,907 |
| Total anotaciones | 2,125 |
| Total categorías | 15 |
| **Duplicados encontrados** | 49 |
| **Imágenes corruptas** | 0 |
| **Issues de consistencia** | 3 |

#### Distribución por Origen

| Dataset | Imágenes | Anotaciones |
|---------|----------|-------------|
| MVTec | 1,640 (86.0%) | 1,640 |
| VISION | 267 (14.0%) | 485 |

#### Proyección a Taxonomía Unificada

| Categoría Unificada | Count | Porcentaje |
|---------------------|-------|------------|
| NORMAL | 1,470 | 69.18% |
| PERFORACIONES | 187 | 8.80% |
| ROTURA_FRACTURA | 169 | 7.95% |
| DEFORMACIONES | 133 | 6.26% |
| RAYONES_ARANAZOS | 106 | 4.99% |
| CONTAMINACION | 60 | 2.82% |

#### Análisis de Resoluciones

| Estadística | Ancho (px) | Alto (px) |
|-------------|------------|-----------|
| Media | 1,161 | 1,071 |
| Desv. Estándar | 669 | 457 |
| Mínimo | 262 | 192 |
| Máximo | 3,840 | 3,620 |
| Mediana | 1,024 | 1,024 |

#### Duplicados Detectados

Se detectaron **49 pares de imágenes duplicadas** (todas del dataset VISION), con distancia de hash perceptual < 5:

- Todos los duplicados son intra-dataset (VISION ↔ VISION)
- No se detectaron duplicados inter-dataset (MVTec ↔ VISION)
- Las distancias varían entre 2 y 4 (muy similares)

### 4.4 Problemas Identificados

1. **Desbalance severo de clases:**
   - NORMAL representa el 69.18% del dataset
   - CONTAMINACION solo representa el 2.82%
   - Ratio max/min: ~24.5:1 (muy superior al recomendado de 3:1)

2. **Fragmentación de etiquetas:**
   - 15 categorías originales deben mapearse a 6 unificadas
   - Ejemplo: "Scratch" (VISION) y "scratch" (MVTec) son la misma categoría

3. **Duplicados:**
   - 49 pares de imágenes muy similares
   - Pueden causar data leakage si no se manejan

4. **Variabilidad de resolución:**
   - Rango de 262x192 a 3840x3620
   - Requiere normalización para entrenamiento ViT

### 4.5 Visualizaciones Generadas

| Archivo | Descripción |
|---------|-------------|
| `category_distribution.png` | Distribución de categorías originales |
| `source_distribution.png` | Distribución por dataset de origen |
| `resolution_scatter.png` | Dispersión de resoluciones de imagen |

### 4.6 Outputs del Análisis

| Archivo | Descripción |
|---------|-------------|
| `analysis_report.json` | Reporte completo en formato JSON |
| `category_distribution.csv` | Distribución de categorías |
| `source_distribution.csv` | Distribución por origen |
| `unified_distribution_preview.csv` | Proyección post-unificación |
| `duplicates.csv` | Lista de pares duplicados |
| `consistency_issues.csv` | Issues de consistencia encontrados |
| `image_resolutions.csv` | Resoluciones de todas las imágenes |

---

## 5. Resumen de Métricas y Resultados

### 5.1 Evolución del Dataset

| Etapa | Imágenes | Categorías | Estado |
|-------|----------|------------|--------|
| Originales combinados | ~9,142 | 64 tipos | Sin curar |
| Post-filtrado inicial | 1,907 | 15 | Curado básico |
| Proyección unificada | 1,907 | 6 | Taxonomía normalizada |

### 5.2 Distribución de Defectos Objetivo

```
DISTRIBUCIÓN POST-ETAPA 3:
==========================

Defecto              | Count | %     | Criticidad
---------------------|-------|-------|------------
NORMAL               | 1,470 | 69.2% | -
PERFORACIONES        |   187 |  8.8% | CRITICA
ROTURA_FRACTURA      |   169 |  8.0% | CRITICA
DEFORMACIONES        |   133 |  6.3% | ALTA
RAYONES_ARANAZOS     |   106 |  5.0% | MEDIA
CONTAMINACION        |    60 |  2.8% | ALTA

Total defectos:        655 (30.8%)
Total normales:      1,470 (69.2%)
```

### 5.3 Calidad del Dataset

| Aspecto | Estado | Detalle |
|---------|--------|---------|
| Integridad de imágenes | ✅ Excelente | 0 imágenes corruptas |
| Consistencia anotaciones | ⚠️ Bueno | 3 issues menores |
| Duplicados | ⚠️ Atención | 49 pares a tratar |
| Balance de clases | ❌ Crítico | Ratio 24.5:1 |
| Formato | ✅ Estándar | COCO JSON |

---

## 5. Etapa 4: Recuración Final y Preparación del Dataset

**Carpeta:** `curation-method_20251101/recuration_dataset_curated_20250921/`

### 5.1 Objetivo

Esta etapa representa el pipeline final de curación científica, dividido en 4 subetapas que transforman el dataset curado inicial en un dataset listo para entrenamiento de modelos ViT:

1. **Subetapa 4.1** - Re-curación con eliminación de componentes irrelevantes
2. **Subetapa 4.2** - Unificación de taxonomía (15 → 6 categorías)
3. **Subetapa 4.3** - Balanceo de clases mediante under/over-sampling
4. **Subetapa 4.4** - Creación de splits estratificados train/val/test

### 5.2 Subetapa 4.1: Re-curación del Dataset

**Script:** `recurate_dataset.py`  
**Output:** `curated_dataset_v1_20251101/`

#### Objetivos
- Eliminar categoría "hazelnut" (fuera del scope de componentes electrónicos)
- Remover duplicados detectados en Etapa 3
- Filtrar solo componentes relevantes para el dominio industrial/electrónico

#### Filtros Aplicados

```yaml
MVTec AD - Categorías permitidas:
  - transistor     # Componente electrónico
  - metal_nut      # Componente mecánico industrial
  - cable          # Conectividad electrónica
  - capsule        # Componente industrial

VISION - Componentes permitidos:
  - PCB_1, PCB_2   # Circuitos impresos
  - Electronics    # Componentes electrónicos
  - Console        # Dispositivos electrónicos
  - Cable          # Cables de conexión
  - Lens           # Óptica para sensores
```

#### Resultados

| Métrica | Valor |
|---------|-------|
| Imágenes originales | 1,907 |
| Imágenes filtradas | **1,393** |
| Eliminadas (hazelnut) | 484 |
| Eliminadas (duplicados) | 30 |
| Imágenes MVTec | 1,156 (83.0%) |
| Imágenes VISION | 237 (17.0%) |

**Tasa de retención:** 73.0%

### 5.3 Subetapa 4.2: Unificación de Taxonomía

**Script:** `unify_taxonomy.py`  
**Output:** `curated_dataset_v2_20251101/`

#### Taxonomía Unificada (6 Categorías)

```yaml
TAXONOMIA_FINAL:
  0 - NORMAL:
    descripcion: "Componente funcional sin defectos detectables"
    criticidad: NINGUNA
    labels_originales: [good, normal]
    
  1 - DEFORMACIONES:
    descripcion: "Alteración geométrica o estructural del componente"
    criticidad: ALTA
    labels_originales: [bent, bent_lead, bent_wire, short, spur]
    
  2 - ROTURA_FRACTURA:
    descripcion: "Discontinuidad estructural, grietas o roturas"
    criticidad: CRÍTICA
    labels_originales: [crack, break, broken, broken_large, broken_small, defect]
    
  3 - RAYONES_ARANAZOS:
    descripcion: "Daño superficial por abrasión o contacto"
    criticidad: MEDIA
    labels_originales: [scratch, Scratch, s_scratch, t_scratch, scratch_head, scratch_neck]
    
  4 - PERFORACIONES:
    descripcion: "Agujeros, cortes o ausencia de material"
    criticidad: CRÍTICA
    labels_originales: [hole, Hole, missing_hole, cut, cut_inner_insulation, cut_outer_insulation, cut_lead]
    
  5 - CONTAMINACION:
    descripcion: "Presencia de material extraño o impurezas"
    criticidad: ALTA
    labels_originales: [contamination, metal_contamination, Dirty, impurities]
```

#### Mapeo Realizado

| Categoría Original | → | Categoría Unificada | Count |
|-------------------|---|---------------------|-------|
| normal | → | NORMAL | 1,039 |
| bent_lead | → | DEFORMACIONES | 84 |
| crack | → | ROTURA_FRACTURA | 151 |
| scratch | → | RAYONES_ARANAZOS | 104 |
| bent | → | DEFORMACIONES | 84 |
| cut | → | PERFORACIONES | 82 |
| hole | → | PERFORACIONES | 82 |
| bent_wire | → | DEFORMACIONES | 84 |
| short | → | DEFORMACIONES | 84 |
| missing_hole | → | PERFORACIONES | 82 |
| spur | → | DEFORMACIONES | 84 |
| Dirty | → | CONTAMINACION | 58 |
| Scratch | → | RAYONES_ARANAZOS | 104 |
| break | → | ROTURA_FRACTURA | 151 |
| Hole | → | PERFORACIONES | 82 |

**Reducción:** 15 categorías originales → 6 categorías unificadas (factor 2.5x)

### 5.4 Subetapa 4.3: Balanceo del Dataset

**Script:** `balance_dataset.py`  
**Output:** `curated_dataset_v3_balanced_20251101/`

#### Estrategia de Balanceo

```yaml
Estrategia: Híbrida (Under-sampling + Over-sampling con augmentación)
Seed: 42 (reproducibilidad)

Distribución Objetivo:
  NORMAL: 300           # Under-sampled (-739)
  DEFORMACIONES: 133    # Original
  ROTURA_FRACTURA: 169  # Original
  RAYONES_ARANAZOS: 150 # Over-sampled
  PERFORACIONES: 187    # Original
  CONTAMINACION: 120    # Over-sampled
```

#### Augmentación Conservadora

```python
# Transformaciones aplicadas (preservan características de defectos)
augmentation_pipeline = [
    HorizontalFlip(p=0.5),
    Rotate(limit=5, p=0.5),           # Solo ±5 grados
    RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.5
    ),
    GaussNoise(var_limit=(5.0, 15.0), p=0.3)
]
```

#### Resultados del Balanceo

| Categoría | Original | Target | Final | Δ |
|-----------|----------|--------|-------|---|
| NORMAL | 1,039 | 300 | 300 | -739 |
| DEFORMACIONES | 84 | 133 | 195 | +111 |
| ROTURA_FRACTURA | 151 | 169 | 211 | +60 |
| RAYONES_ARANAZOS | 104 | 150 | 162 | +58 |
| PERFORACIONES | 82 | 187 | 328 | +246 |
| CONTAMINACION | 58 | 120 | 158 | +100 |

**Total imágenes augmentadas generadas:** 368  
**Ratio máx/min final:** 2.08:1 ✅ (óptimo < 3:1)

### 5.5 Subetapa 4.4: Creación de Splits Estratificados

**Script:** `create_splits.py`  
**Output:** `curated_dataset_v4_splitted_20251101/` (Dataset Final)

#### Configuración de Splits

```yaml
Split Ratios:
  train: 70%
  val: 10%
  test: 20%

Estratificación: Dual (categoría + dataset origen)
Seed: 42
```

#### Validaciones Realizadas

1. **Test Chi-cuadrado** - Verifica que cada split mantiene la distribución global
2. **Leakage Check** - Garantiza no hay imágenes compartidas entre splits

#### Resultados Finales

| Split | Imágenes | Porcentaje |
|-------|----------|------------|
| **Train** | 715 | 70.0% |
| **Val** | 102 | 10.0% |
| **Test** | 205 | 20.0% |
| **Total** | **1,022** | 100% |

#### Distribución por Split

| Categoría | Train | Val | Test |
|-----------|-------|-----|------|
| NORMAL | 210 | 30 | 60 |
| DEFORMACIONES | 94 | 13 | 26 |
| ROTURA_FRACTURA | 118 | 17 | 34 |
| RAYONES_ARANAZOS | 104 | 15 | 30 |
| PERFORACIONES | 106 | 15 | 31 |
| CONTAMINACION | 83 | 12 | 24 |

#### Validación Estadística

```json
{
  "chi2_test_results": {
    "train": {"chi2": 0.006, "p_value": 0.999, "similar_to_global": true},
    "val": {"chi2": 0.005, "p_value": 0.999, "similar_to_global": true},
    "test": {"chi2": 0.013, "p_value": 0.999, "similar_to_global": true}
  },
  "leakage_check": "passed"
}
```

### 5.6 Estructura del Dataset Final

```
curated_dataset_v4_splitted_20251101/
├── train/
│   ├── images/                    # 715 imágenes PNG
│   ├── train.json                 # Anotaciones COCO
│   └── train_files.txt            # Lista de archivos
├── val/
│   ├── images/                    # 102 imágenes PNG
│   ├── val.json                   # Anotaciones COCO
│   └── val_files.txt              # Lista de archivos
├── test/
│   ├── images/                    # 205 imágenes PNG
│   ├── test.json                  # Anotaciones COCO
│   └── test_files.txt             # Lista de archivos
└── metadata/
    └── phase5_splits_log.json     # Metadata del proceso
```

### 5.7 Resumen de la Etapa 4

| Subetapa | Operación | Input → Output |
|----------|-----------|----------------|
| 4.1 | Re-curación | 1,907 → 1,393 imgs |
| 4.2 | Unificación | 15 → 6 categorías |
| 4.3 | Balanceo | Ratio 24.5:1 → 2.08:1 |
| 4.4 | Splits | 70/10/20 estratificados |

**Dataset Final:** 1,022 imágenes listas para entrenamiento ViT

---

## 6. Etapa 5: Análisis Exhaustivo del Dataset Final

**Carpeta:** `Final_analysis_curated_dataset_1st_version/`

### 6.1 Objetivo

Esta etapa representa el análisis científico final y validación del dataset curado para garantizar su calidad antes del entrenamiento de modelos. Se realizan análisis exhaustivos de:

- Distribución de tamaños de imágenes
- Estadísticas de bounding boxes
- Balance entre categorías y splits
- Ratio de imágenes augmentadas vs originales
- Detección de problemas potenciales para el entrenamiento

### 6.2 Scripts de Análisis

| Script | Propósito |
|--------|-----------|
| `analyze_images_sizes.py` | Análisis de resoluciones y recomendación para entrenamiento |
| `dataset_balance_analysis.py` | Análisis exhaustivo de balance por categoría y split |
| `dataset_inspect.py` | Inspección detallada de cada split en formato COCO |

### 6.3 Resumen Ejecutivo del Dataset Final

```yaml
RESUMEN EJECUTIVO - DATASET CURADO FINAL
=========================================

Total imágenes: 1,022
Total anotaciones: 1,354

Splits:
  Train: 715 imágenes (944 anotaciones)
  Val:   102 imágenes (145 anotaciones)
  Test:  205 imágenes (265 anotaciones)
```

#### Distribución de Categorías (por anotaciones)

| Categoría | Train | Val | Test | **Total** |
|-----------|-------|-----|------|-----------|
| PERFORACIONES | 242 | 26 | 60 | **328** (24.2%) |
| NORMAL | 210 | 30 | 60 | **300** (22.2%) |
| ROTURA_FRACTURA | 138 | 33 | 40 | **211** (15.6%) |
| DEFORMACIONES | 136 | 21 | 38 | **195** (14.4%) |
| RAYONES_ARANAZOS | 111 | 17 | 34 | **162** (12.0%) |
| CONTAMINACION | 107 | 18 | 33 | **158** (11.7%) |

### 6.4 Estadísticas de Bounding Boxes

| Split | N BBoxes | Width (mediana) | Height (mediana) | Área (mediana) | Aspect Ratio |
|-------|----------|-----------------|------------------|----------------|--------------|
| Train | 944 | 222.4 px | 132.5 px | 6,082 px² | 1.00 |
| Val | 145 | 224.0 px | 112.0 px | 7,435 px² | 1.00 |
| Test | 265 | 314.0 px | 175.0 px | 7,914 px² | 1.00 |

**Estadísticas agregadas:**
- Width media: 450.5 px (σ=443.9)
- Height media: 412.8 px (σ=422.1)
- Área media: 338,576 px²
- Aspect Ratio medio: 1.49

### 6.5 Tamaños de Imágenes

| Split | N Imgs | Width (media) | Height (media) | Área (media) |
|-------|--------|---------------|----------------|--------------|
| Train | 715 | 1,647.5 px | 1,347.4 px | 3.09 MP |
| Val | 102 | 1,735.2 px | 1,456.7 px | 3.56 MP |
| Test | 205 | 1,703.7 px | 1,415.1 px | 3.37 MP |

**Rango de resoluciones:**
- Mínimo: 262×192 px
- Máximo: 3,840×3,620 px
- Mediana: ~1,024×1,024 px

### 6.6 Proporción de Augmentación

| Split | Original | Augmentado | % Augmentado |
|-------|----------|------------|--------------|
| Train | 458 (64.1%) | 257 (35.9%) | 35.9% |
| Val | 63 (61.8%) | 39 (38.2%) | 38.2% |
| Test | 133 (64.9%) | 72 (35.1%) | 35.1% |

**Total augmentado:** 368 imágenes (36.0%)

### 6.7 Problemas Detectados y Recomendaciones

#### 🚨 Problema Crítico: Bounding Boxes Pequeños

Se detectó que un porcentaje significativo de bboxes tienen dimensiones menores a 32px:

| Split | Width < 32px | Height < 32px |
|-------|--------------|---------------|
| Train | 23.4% | 22.2% |
| Val | 16.6% | 20.0% |
| Test | 17.0% | 16.2% |

**Implicación:** Faster R-CNN usa anchors mínimos de 32px por defecto.

**Solución recomendada:**
```python
# Configuración de AnchorGenerator con anchors más pequeños
from torchvision.models.detection.rpn import AnchorGenerator

anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128, 256),),  # Añadir 16px
    aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)  # Más variedad de ratios
)
```

#### ⚠️ Atención: Aspect Ratios Extremos

~11-15% de bboxes tienen aspect ratio extremo (<0.25 o >4.0).

**Recomendación:**
- Usar Feature Pyramid Networks (FPN) para detección multi-escala
- Considerar Vision Transformers (DINOv2/DINOv3) que manejan mejor estos casos
- Attention global captura mejor defectos pequeños y elongados

#### 📐 Recomendación de Resolución de Entrenamiento

Basado en el análisis de distribución de tamaños:

```yaml
Resolución recomendada: 1400×1400 px

Razones:
  1. Múltiplo de 14 (compatible con patch size de ViT/DINOv3)
  2. Cercano al percentil 75 del dataset
  3. Balance entre upscaling (35%) y downscaling (65%)
  4. Manejable en GPU RTX 4070 12GB con batch_size=1

Config YAML sugerida:
  transforms:
    - {type: Resize, size: [1400, 1400]}
  collate_fn:
    base_size: 1400
  eval_spatial_size: [1400, 1400]
```

### 6.8 Visualizaciones Generadas

#### Gráficas de Distribución

| Archivo | Descripción |
|---------|-------------|
| `image_size_distributions.png` | Histogramas de ancho, alto, lado corto y aspect ratio |
| `width_vs_height_scatter.png` | Scatter plot de dimensiones por split |
| `category_distribution.png` | Distribución de categorías por split |
| `category_proportions.png` | Pie charts de proporciones por split |
| `bbox_distribution.png` | Distribución de tamaños de bboxes |
| `image_sizes_distribution.png` | Distribución de tamaños de imágenes |
| `augmentation_distribution.png` | Comparativa original vs augmentado |
| `source_dataset_distribution.png` | Distribución por dataset de origen |

#### Histogramas por Split

Para cada split (train/val/test):
- `hist_bbox_area.png` - Distribución de áreas de bbox
- `hist_bbox_aspect_ratio.png` - Distribución de aspect ratios
- `hist_short_edge.png` - Distribución de lado más corto de imágenes

### 6.9 Outputs del Análisis

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

### 6.10 Conclusión del Análisis

| Aspecto | Estado | Observación |
|---------|--------|-------------|
| Balance de clases | ✅ Óptimo | Ratio 2.08:1 |
| Distribución entre splits | ✅ Validada | Chi² p-value > 0.99 |
| Integridad de imágenes | ✅ 100% | 0 archivos faltantes |
| Proporción augmentación | ✅ Adecuada | 36% augmentado |
| Bboxes pequeños | ⚠️ Atención | ~20% < 32px, ajustar anchors |
| Aspect ratios extremos | ⚠️ Atención | ~12% extremos, usar FPN |
| Trazabilidad | ✅ Completa | Origen y augmentación registrados |

**Veredicto:** Dataset APTO para entrenamiento con las configuraciones recomendadas.

---

## 7. Resumen de Métricas y Resultados Finales

### 7.1 Evolución del Dataset a través de Etapas

| Etapa | Imágenes | Categorías | Estado |
|-------|----------|------------|--------|
| Originales combinados | ~9,142 | 64 tipos | Sin curar |
| Post-Etapa 2 (filtrado inicial) | 1,907 | 15 | Curado básico |
| Post-Etapa 4.1 (re-curación) | 1,393 | 15 | Sin hazelnut/duplicados |
| Post-Etapa 4.2 (unificación) | 1,393 | 6 | Taxonomía normalizada |
| Post-Etapa 4.3 (balanceo) | 1,354 + 368 aug | 6 | Balanceado |
| **Post-Etapa 4.4 (splits)** | **1,022** | **6** | **✅ LISTO** |

### 7.2 Distribución Final del Dataset

```
DISTRIBUCIÓN DEL DATASET CURADO FINAL:
======================================

Categoría              | Train | Val  | Test | Total | %
-----------------------|-------|------|------|-------|------
NORMAL                 |  210  |  30  |  60  |  300  | 29.4%
DEFORMACIONES          |   94  |  13  |  26  |  133  | 13.0%
ROTURA_FRACTURA        |  118  |  17  |  34  |  169  | 16.5%
RAYONES_ARANAZOS       |  104  |  15  |  30  |  149  | 14.6%
PERFORACIONES          |  106  |  15  |  31  |  152  | 14.9%
CONTAMINACION          |   83  |  12  |  24  |  119  | 11.6%
-----------------------|-------|------|------|-------|------
TOTAL                  |  715  | 102  | 205  | 1,022 | 100%
```

### 7.3 Calidad del Dataset Final

| Aspecto | Estado | Detalle |
|---------|--------|---------|
| Integridad de imágenes | ✅ Excelente | 0 imágenes corruptas |
| Consistencia anotaciones | ✅ Excelente | Esquema COCO unificado |
| Duplicados | ✅ Resuelto | 0 duplicados en dataset final |
| Balance de clases | ✅ Óptimo | Ratio 2.08:1 (< 3:1) |
| Estratificación | ✅ Validada | Chi² p-value > 0.99 |
| Leakage | ✅ Ninguno | Splits completamente disjuntos |
| Formato | ✅ Estándar | COCO JSON + carpetas organizadas |
| Trazabilidad | ✅ Completa | Metadata en cada fase |

---

## 8. Estructura de Carpetas del Proyecto

```
VISION-mvtecad-mixedDataset/
│
├── Analisis-datasets-VISION-mvtecad/              # ETAPA 1
│   ├── Analisis_datasets/
│   │   ├── dataset_exploration.py                 # Explorador VISION
│   │   ├── analyze_mvtec_dataset.py               # Analizador MVTec
│   │   ├── mvtec_inspector.py                     # Inspector MVTec
│   │   ├── vision_dataset_report.json             # Reporte VISION
│   │   ├── mvtec_ad_report.json                   # Reporte MVTec
│   │   ├── datasets_technical_summary.txt         # Resumen técnico
│   │   └── detailed_categories_analysis.txt       # Análisis categorías
│   └── posibles_categorias-defectos.txt           # Taxonomía propuesta
│
├── curated_dataset_20250921_115859/               # ETAPA 2
│   ├── images/                                    # 1,907 imágenes PNG
│   ├── annotations.coco.json                      # Anotaciones COCO
│   ├── metadata/
│   │   └── dataset_info.json                      # Metadatos curación
│   ├── dataset_curator_20250921.py                # Script de curación
│   └── Curation-dataset-proposal.md               # Propuesta metodológica
│
├── curation-method_20251101/                      # ETAPAS 3 y 4
│   ├── analysis_curated_dataset_20251101/         # Resultados análisis (Etapa 3)
│   │   ├── analysis_report.json
│   │   ├── category_distribution.csv
│   │   ├── duplicates.csv
│   │   └── ...
│   ├── analysis_previous_curated_dataset_20250921.py  # Script Etapa 3
│   │
│   └── recuration_dataset_curated_20250921/       # ETAPA 4 (Recuración Final)
│       ├── scripts/                               # Scripts del pipeline
│       │   ├── recurate_dataset.py                # Subetapa 4.1
│       │   ├── unify_taxonomy.py                  # Subetapa 4.2
│       │   ├── balance_dataset.py                 # Subetapa 4.3
│       │   └── create_splits.py                   # Subetapa 4.4
│       ├── curated_dataset_v1_20251101/           # Output Subetapa 4.1
│       │   └── metadata/phase2_recuration_log.json
│       ├── curated_dataset_v2_20251101/           # Output Subetapa 4.2
│       │   └── metadata/phase3_unification_log.json
│       ├── curated_dataset_v3_balanced_20251101/  # Output Subetapa 4.3
│       │   └── metadata/phase4_balancing_log.json
│       ├── curated_dataset_v4_splitted_20251101/  # ✅ DATASET FINAL
│       │   ├── train/images/ + train.json         # 715 imágenes
│       │   ├── val/images/ + val.json             # 102 imágenes
│       │   ├── test/images/ + test.json           # 205 imágenes
│       │   └── metadata/phase5_splits_log.json
│       ├── Dataset_curation_technical_report.md   # Reporte técnico
│       └── Implementation_Guide.md                # Guía implementación
│
├── Final_analysis_curated_dataset_1st_version/    # ETAPA 5 - Análisis Final
│   ├── re-analyze-1st-dataset-version-scripts/    # Scripts de análisis
│   │   ├── analyze_images_sizes.py
│   │   ├── dataset_balance_analysis.py
│   │   └── dataset_inspect.py
│   └── outputs/                                   # Resultados del análisis
│       ├── analysis_imagessizes_plots/
│       ├── balanced-dataset-analysis-20251114/
│       └── dataset_info/
│           ├── inspect_train/
│           ├── inspect_val/
│           └── inspect_test/
│
├── flujo_curacion_dataset/                        # Scripts organizados
│   ├── etapa1_exploracion/
│   ├── etapa2_curacion_inicial/
│   ├── etapa3_analisis_exhaustivo/
│   ├── etapa4_recuracion_final/                   # Scripts Etapa 4
│   │   ├── 01_recurate_dataset.py
│   │   ├── 02_unify_taxonomy.py
│   │   ├── 03_balance_dataset.py
│   │   ├── 04_create_splits.py
│   │   └── README.md
│   └── etapa5_analisis_final/                     # Scripts Etapa 5
│       ├── 01_analizar_tamanios_imagenes.py
│       ├── 02_analisis_balance_dataset.py
│       ├── 03_inspeccionar_dataset.py
│       └── README.md
│
└── DOCUMENTACION_CURACION_DATASET.md              # Este documento
```

---

## Anexo A: Comandos de Ejecución

### Etapa 1: Exploración

```bash
# Analizar VISION-Datasets
cd Analisis-datasets-VISION-mvtecad/Analisis_datasets/
python dataset_exploration.py

# Analizar MVTec AD
python analyze_mvtec_dataset.py
```

### Etapa 2: Curación Inicial

```bash
cd curated_dataset_20250921_115859/
python dataset_curator_20250921.py \
  --vision-path ../path/to/VISION-Datasets \
  --mvtec-path ../path/to/mvtec-ad \
  --output-path ./
```

### Etapa 3: Análisis Exhaustivo

```bash
cd curation-method_20251101/
python analysis_previous_curated_dataset_20250921.py \
  --dataset ../curated_dataset_20250921_115859 \
  --output ./analysis_curated_dataset_20251101
```

### Etapa 4: Recuración Final (Pipeline Completo)

```bash
cd curation-method_20251101/recuration_dataset_curated_20250921/scripts/

# Subetapa 4.1: Re-curación
python recurate_dataset.py \
  --source ../../curated_dataset_20250921_115859 \
  --output ../curated_dataset_v1_20251101 \
  --duplicates ../analysis_curated_dataset_20251101/duplicates.csv

# Subetapa 4.2: Unificación de taxonomía
python unify_taxonomy.py \
  --source ../curated_dataset_v1_20251101 \
  --output ../curated_dataset_v2_20251101

# Subetapa 4.3: Balanceo
python balance_dataset.py \
  --source ../curated_dataset_v2_20251101 \
  --output ../curated_dataset_v3_balanced_20251101 \
  --seed 42

# Subetapa 4.4: Creación de splits
python create_splits.py \
  --source ../curated_dataset_v3_balanced_20251101 \
  --output ../curated_dataset_v4_splitted_20251101 \
  --seed 42
```

### Etapa 5: Análisis Final del Dataset

```bash
cd Final_analysis_curated_dataset_1st_version/re-analyze-1st-dataset-version-scripts/

# Análisis de tamaños de imágenes
python analyze_images_sizes.py

# Inspección detallada por split
python dataset_inspect.py \
  --coco-json ../../curated_dataset_splitted_20251101_provisional_1st_version/train/train.json \
  --images-dir ../../curated_dataset_splitted_20251101_provisional_1st_version/train/images \
  --out-dir ../outputs/dataset_info/inspect_train

python dataset_inspect.py \
  --coco-json ../../curated_dataset_splitted_20251101_provisional_1st_version/val/val.json \
  --images-dir ../../curated_dataset_splitted_20251101_provisional_1st_version/val/images \
  --out-dir ../outputs/dataset_info/inspect_val

python dataset_inspect.py \
  --coco-json ../../curated_dataset_splitted_20251101_provisional_1st_version/test/test.json \
  --images-dir ../../curated_dataset_splitted_20251101_provisional_1st_version/test/images \
  --out-dir ../outputs/dataset_info/inspect_test

# Análisis exhaustivo de balance
python dataset_balance_analysis.py \
  --dataset-root ../outputs/dataset_info \
  --output-dir ../outputs/balanced-dataset-analysis-20251114
```

---

## Anexo B: Dependencias Python

```txt
# requirements.txt para el pipeline de curación
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
pycocotools>=2.0.7
scikit-learn>=1.3.0
albumentations>=1.3.1
imagehash>=4.3.1
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
```

---

## Anexo C: Formato de Anotación COCO Híbrido

El dataset final utiliza un esquema híbrido COCO-compatible que combina información de ambos datasets:

```json
{
  "annotation": {
    "id": 1234,
    "image_id": 567,
    "category_id": 2,
    "unified_category_name": "ROTURA_FRACTURA",
    
    "bbox": [x, y, width, height],
    "segmentation": [[...]] | [],
    "area": 1234.5,
    
    "has_segmentation": true | false,
    "localization_type": "pixel_level" | "bbox_level" | "image_level",
    "confidence": 1.0,
    
    "source_dataset": "mvtec" | "vision",
    "original_category_id": 5,
    "original_label": "crack"
  }
}
```

---

**Documento generado:** Diciembre 2025  
**Versión:** 3.0 (incluye Etapas 1-5 completas)  
**Dataset Final:** `curated_dataset_splitted_20251101_provisional_1st_version/` (1,022 imágenes)  
**Verificado:** Análisis exhaustivo completado ✅

