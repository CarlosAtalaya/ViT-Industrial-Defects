# Etapa 1: Exploración de Datasets Originales

## Objetivo

Analizar en profundidad los datasets originales **VISION-Datasets** y **MVTec AD** para:
- Identificar estructura y formato de datos
- Catalogar tipos de defectos disponibles
- Encontrar correspondencias semánticas entre datasets
- Definir taxonomía unificada para el proyecto

---

## Scripts Incluidos

### 01_explorar_vision_dataset.py

**Clase principal:** `VisionDatasetExplorer`

**Funcionalidades:**
- Extracción automática de archivos `.tar.gz`
- Análisis de estructura por componente (train/val/inference)
- Lectura de anotaciones COCO JSON
- Estadísticas de imágenes (resolución, canales)
- Distribución de defectos por componente
- Generación de reporte JSON
- Visualización de muestras

**Uso:**
```bash
# Por defecto usa <raíz del repo>/VISION-Datasets (o variable VISION_DATASETS_PATH)
python 01_explorar_vision_dataset.py
python 01_explorar_vision_dataset.py --vision-path /ruta/a/VISION-Datasets
```

**Outputs:**
- `vision_dataset_report.json`
- Imágenes de muestra por componente

---

### 02_analizar_mvtec_dataset.py

**Clase principal:** `MVTecAnalyzer`

**Funcionalidades:**
- Carga de `samples.json`
- Análisis estructural (categorías, defectos, splits)
- Detalle por categoría (normal vs anomalía)
- Patrones de defectos por categoría
- Visualizaciones estadísticas

**Uso:**
```bash
# Por defecto usa <raíz del repo>/mvtec-ad (o variable MVTEC_AD_PATH). Requiere samples.json.
python 02_analizar_mvtec_dataset.py
python 02_analizar_mvtec_dataset.py --mvtec-path /ruta/a/mvtec-ad
```

**Outputs:**
- `mvtec_ad_report.json`
- `mvtec_analysis_charts.png`

---

### 03_taxonomia_propuesta.txt

Archivo de configuración que define:

```python
defectos_tfg = [
    {
        "nombre": "ROTURA/FRACTURA",
        "vision_labels": ["break", "defect"],
        "mvtec_labels": ["broken", "broken_large", "crack"],
        "importancia": "CRÍTICA"
    },
    # ... más categorías
]

categorias_mvtec = ["transistor", "metal_nut", "cable", "capsule", "hazelnut"]
componentes_vision = ["PCB_1", "Console", "Electronics", "Cable", "Lens"]
```

---

## Resultados Clave de Esta Etapa

### VISION-Datasets
- **14 componentes** industriales
- **3,788 imágenes** (train: 1,760 | val: 2,028)
- **44 tipos de defectos** únicos
- Formato COCO JSON con bounding boxes

### MVTec AD
- **15 categorías** de objetos
- **5,354 muestras** (train: 3,629 | test: 1,725)
- **49 tipos de defectos**
- Máscaras pixel-level

### Mapeo Identificado

| Defecto Unificado | VISION | MVTec |
|-------------------|--------|-------|
| ROTURA/FRACTURA | break, defect | broken, crack |
| CONTAMINACIÓN | Dirty, impurities | contamination, metal_contamination |
| RAYONES/ARAÑAZOS | Scratch, s_scratch, t_scratch | scratch, scratch_head |
| PERFORACIONES | Hole, missing_hole | hole, cut |
| DEFORMACIONES | short, spur | bent, bent_lead, bent_wire |

---

## Siguiente Etapa

Con los resultados de esta exploración, proceder a **Etapa 2** para la curación inicial del dataset.

