# Etapa 3: Análisis Exhaustivo del Dataset Curado

## Objetivo

Realizar un análisis científico riguroso del dataset curado para:
- Validar calidad de imágenes
- Detectar duplicados
- Verificar consistencia de anotaciones
- Proyectar distribución tras unificación de taxonomía
- Identificar problemas a resolver en la recuración

---

## Script Incluido

### 01_analisis_profundo_dataset.py

**Clase principal:** `DatasetDeepAnalyzer`

**Argumentos CLI:**
```bash
python 01_analisis_profundo_dataset.py \
  --dataset /path/to/curated_dataset \
  --output /path/to/output_reports
```

---

## Análisis Implementados

### 1. Distribución de Categorías

```python
analyze_category_distribution()
```
- Cuenta anotaciones por categoría original
- Calcula porcentajes
- Genera DataFrame con estadísticas

### 2. Distribución por Origen

```python
analyze_source_distribution()
```
- Separa imágenes y anotaciones por dataset de origen
- Cuenta MVTec vs VISION

### 3. Detección de Duplicados

```python
detect_duplicates(hash_size=16, threshold=5)
```
- Usa **perceptual hashing** (pHash)
- Compara todas las imágenes entre sí
- Identifica pares con distancia < threshold
- Reporta archivo origen de cada duplicado

### 4. Análisis de Calidad de Imágenes

```python
analyze_image_quality()
```
- Verifica existencia de archivos
- Extrae resolución (width, height)
- Detecta formato y modo de color
- Identifica imágenes corruptas

### 5. Consistencia de Anotaciones

```python
analyze_annotation_consistency()
```

**Reglas verificadas:**
- Todas las imágenes tienen al menos una anotación
- Imágenes "good" tienen categoría "normal"
- Bounding boxes dentro de límites de imagen
- Áreas de bbox no sospechosamente pequeñas (< 100px²)

### 6. Mapeo a Taxonomía Unificada

```python
analyze_defect_mapping()
```
- Carga mapeo desde metadata
- Proyecta categorías originales a 6 unificadas
- Calcula distribución esperada post-unificación

---

## Outputs Generados

| Archivo | Descripción |
|---------|-------------|
| `analysis_report.json` | Reporte completo en JSON |
| `category_distribution.csv` | Distribución de categorías |
| `category_distribution.png` | Gráfico de barras |
| `source_distribution.csv` | Balance MVTec/VISION |
| `source_distribution.png` | Gráficos comparativos |
| `unified_distribution_preview.csv` | Proyección post-unificación |
| `duplicates.csv` | Lista de pares duplicados |
| `consistency_issues.csv` | Problemas detectados |
| `image_resolutions.csv` | Resoluciones de todas las imágenes |
| `resolution_scatter.png` | Scatter plot de resoluciones |

---

## Estructura del Reporte JSON

```json
{
  "timestamp": "2025-11-01T...",
  "dataset_path": "...",
  "summary": {
    "total_images": 1907,
    "total_annotations": 2125,
    "total_categories": 15,
    "duplicates_found": 49,
    "corrupted_images": 0,
    "consistency_issues": 3
  },
  "categories": {
    "count": {"normal": 1470, "break": 128, ...},
    "percentage": {"normal": 69.18, ...}
  },
  "source_distribution": {
    "images": {"mvtec": 1640, "vision": 267},
    "annotations": {"mvtec": 1640, "vision": 485}
  },
  "unified_preview": {
    "count": {"NORMAL": 1470, "PERFORACIONES": 187, ...},
    "percentage": {"NORMAL": 69.18, ...}
  },
  "quality_metrics": {
    "resolution_stats": {...},
    "format_distribution": {"PNG": 1907}
  }
}
```

---

## Resultados Típicos

### Distribución por Categoría Original

| Categoría | Count | % |
|-----------|-------|---|
| normal | 1,470 | 69.2% |
| break | 128 | 6.0% |
| missing_hole | 124 | 5.8% |
| Scratch | 60 | 2.8% |
| Dirty | 60 | 2.8% |
| ... | ... | ... |

### Proyección a Taxonomía Unificada

| Categoría | Count | % |
|-----------|-------|---|
| NORMAL | 1,470 | 69.2% |
| PERFORACIONES | 187 | 8.8% |
| ROTURA_FRACTURA | 169 | 8.0% |
| DEFORMACIONES | 133 | 6.3% |
| RAYONES_ARANAZOS | 106 | 5.0% |
| CONTAMINACION | 60 | 2.8% |

### Problemas Identificados

1. **Desbalance severo**: NORMAL 69% vs CONTAMINACION 2.8%
2. **Duplicados**: ~50 pares de imágenes similares
3. **Fragmentación**: 15 etiquetas para 6 categorías conceptuales

---

## Dependencias

```bash
pip install pandas imagehash Pillow matplotlib seaborn
```

---

## Interpretación de Resultados

### Duplicados
- **Distancia 0-2**: Prácticamente idénticos
- **Distancia 3-4**: Muy similares (posible augmentación)
- **Distancia 5+**: Diferentes

### Severidad de Issues
- **CRITICAL**: Bboxes fuera de límites
- **HIGH**: Inconsistencia etiqueta/anotación
- **MEDIUM**: Imágenes sin anotaciones
- **LOW**: Áreas sospechosamente pequeñas

---

## Acciones Recomendadas Post-Análisis

Basado en los resultados, la **Etapa 4 (Recuración)** debe:

1. ✅ Unificar taxonomía (15 → 6 categorías)
2. ✅ Eliminar/gestionar duplicados
3. ✅ Balancear clases:
   - Under-sampling de NORMAL (~300-500)
   - Over-sampling de CONTAMINACION y RAYONES
4. ✅ Crear splits train/val/test estratificados
5. ✅ Validar integridad final

