# Etapa 4: Recuración Final y Preparación del Dataset

## Objetivo

Pipeline completo de curación científica que transforma el dataset curado inicial en un dataset listo para entrenamiento de Vision Transformers:

1. **Subetapa 4.1** - Re-curación con eliminación de componentes irrelevantes
2. **Subetapa 4.2** - Unificación de taxonomía (15 → 6 categorías)
3. **Subetapa 4.3** - Balanceo de clases mediante under/over-sampling
4. **Subetapa 4.4** - Creación de splits estratificados train/val/test

---

## Scripts Incluidos

### 01_recurate_dataset.py

**Clase principal:** `DatasetRecurator`

**Objetivos:**
- Eliminar categoría "hazelnut" (fuera del scope electrónico)
- Remover duplicados detectados en análisis previo
- Filtrar solo componentes relevantes (electrónicos/industriales)

**Uso:**

```bash
python 01_recurate_dataset.py \
  --source /path/to/curated_dataset_20250921_115859 \
  --output /path/to/curated_dataset_v1 \
  --duplicates /path/to/duplicates.csv
```

**Resultados esperados:**
- 1,907 → ~1,393 imágenes
- Eliminación de 484 imágenes hazelnut
- Eliminación de ~30 duplicados

---

### 02_unify_taxonomy.py

**Clase principal:** `TaxonomyUnifier`

**Objetivos:**
- Mapear 15 categorías originales → 6 categorías unificadas
- Convertir a esquema híbrido COCO-compatible
- Añadir metadata de trazabilidad completa

**Taxonomía Unificada:**

```yaml
0: NORMAL          - Componente sin defectos (Criticidad: NINGUNA)
1: DEFORMACIONES   - Alteración geométrica (Criticidad: ALTA)
2: ROTURA_FRACTURA - Grietas, roturas (Criticidad: CRÍTICA)
3: RAYONES_ARANAZOS- Daño superficial (Criticidad: MEDIA)
4: PERFORACIONES   - Agujeros, cortes (Criticidad: CRÍTICA)
5: CONTAMINACION   - Material extraño (Criticidad: ALTA)
```

**Uso:**

```bash
python 02_unify_taxonomy.py \
  --source /path/to/curated_dataset_v1 \
  --output /path/to/curated_dataset_v2
```

---

### 03_balance_dataset.py

**Clase principal:** `DatasetBalancerFinalFix`

**Objetivos:**
- Under-sampling de clase NORMAL (1,039 → 300)
- Over-sampling de clases minoritarias mediante augmentación
- Lograr ratio máx/min < 3:1

**Estrategia de Balanceo:**

```yaml
Target Distribution:
  NORMAL: 300           # Under-sampled
  DEFORMACIONES: 133    # Original
  ROTURA_FRACTURA: 169  # Original
  RAYONES_ARANAZOS: 150 # Over-sampled
  PERFORACIONES: 187    # Original
  CONTAMINACION: 120    # Over-sampled
```

**Augmentación Conservadora:**

```python
# Transformaciones que preservan características de defectos
- HorizontalFlip(p=0.5)
- Rotate(limit=5, p=0.5)           # Solo ±5 grados
- RandomBrightnessContrast(±10%, p=0.5)
- GaussNoise(var_limit=(5.0, 15.0), p=0.3)
```

**Uso:**

```bash
python 03_balance_dataset.py \
  --source /path/to/curated_dataset_v2 \
  --output /path/to/curated_dataset_v3_balanced \
  --seed 42
```

---

### 04_create_splits.py

**Clase principal:** `StratifiedSplitterWithFolders`

**Objetivos:**
- Crear splits train/val/test con estratificación dual (categoría + origen)
- Organizar imágenes en carpetas separadas por split
- Validar no-leakage y distribución preservada

**Configuración:**

```yaml
Split Ratios:
  train: 70%
  val: 10%
  test: 20%

Estratificación: Dual (categoría + dataset origen)
Seed: 42 (reproducibilidad)
```

**Uso:**

```bash
python 04_create_splits.py \
  --source /path/to/curated_dataset_v3_balanced \
  --output /path/to/curated_dataset_v4_splitted \
  --seed 42
```

**Estructura de Salida:**

```
curated_dataset_v4_splitted/
├── train/
│   ├── images/       # 715 imágenes
│   └── train.json    # Anotaciones COCO
├── val/
│   ├── images/       # 102 imágenes
│   └── val.json
├── test/
│   ├── images/       # 205 imágenes
│   └── test.json
└── metadata/
    └── phase5_splits_log.json
```

---

## Ejecución del Pipeline Completo

### Opción 1: Manual (paso a paso)

```bash
# Desde el directorio de scripts
cd flujo_curacion_dataset/etapa4_recuracion_final/

# 1. Re-curación
python 01_recurate_dataset.py --source ... --output ... --duplicates ...

# 2. Unificación
python 02_unify_taxonomy.py --source ... --output ...

# 3. Balanceo
python 03_balance_dataset.py --source ... --output ... --seed 42

# 4. Splits
python 04_create_splits.py --source ... --output ... --seed 42
```

### Opción 2: Script maestro

Se puede crear un script bash que ejecute todo el pipeline secuencialmente.

---

## Validaciones Implementadas

### Chi-cuadrado (Distribución de Splits)

```python
# Verifica que cada split mantiene la distribución global
chi2_test_results:
  train: p_value > 0.99 ✅
  val: p_value > 0.99 ✅
  test: p_value > 0.99 ✅
```

### Leakage Check

```python
# Garantiza que no hay imágenes compartidas entre splits
train_ids ∩ val_ids = ∅
train_ids ∩ test_ids = ∅
val_ids ∩ test_ids = ∅
```

---

## Resultados del Pipeline

| Subetapa | Input | Output | Operación |
|----------|-------|--------|-----------|
| 4.1 | 1,907 imgs | 1,393 imgs | -484 hazelnut, -30 duplicados |
| 4.2 | 15 categorías | 6 categorías | Unificación semántica |
| 4.3 | Ratio 24.5:1 | Ratio 2.08:1 | Balanceo híbrido |
| 4.4 | 1,022 imgs | 715/102/205 | Splits estratificados |

**Dataset Final:** 1,022 imágenes listas para entrenamiento ViT

---

## Dependencias

```txt
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
pycocotools>=2.0.7
scikit-learn>=1.3.0
albumentations>=1.3.1
scipy>=1.11.0
```

---

## Outputs Generados

| Archivo | Descripción |
|---------|-------------|
| `phase2_recuration_log.json` | Log de re-curación |
| `phase2_summary.csv` | Resumen estadístico |
| `phase3_unification_log.json` | Log de unificación |
| `category_mapping.csv` | Mapeo de categorías |
| `phase4_balancing_log.json` | Log de balanceo |
| `phase5_splits_log.json` | Log de splits |

---

**Documento generado:** Diciembre 2025  
**Fase:** 4/4 del Pipeline de Curación  
**Resultado:** Dataset curado listo para entrenamiento ViT

