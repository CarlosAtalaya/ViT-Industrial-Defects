# Etapa 2: Curación Inicial del Dataset

## Objetivo

Crear un dataset unificado combinando imágenes seleccionadas de VISION-Datasets y MVTec AD:
- Filtrar por categorías y defectos objetivo
- Convertir todas las imágenes a formato PNG
- Generar anotaciones COCO unificadas
- Mantener trazabilidad del origen

---

## Script Incluido

### 01_dataset_curator.py

**Clase principal:** `DatasetCurator`

**Argumentos CLI:**
```bash
python 01_dataset_curator.py \
  --vision-path /path/to/VISION-Datasets \
  --mvtec-path /path/to/mvtec-ad \
  --output-path ./output_directory
```

---

## Configuración

### Defectos Objetivo

El script filtra imágenes según estos tipos de defectos:

```python
defectos_objetivo = {
    "ROTURA_FRACTURA": {
        "vision_labels": ["break", "defect"],
        "mvtec_labels": ["broken", "broken_large", "crack"],
        "importancia": "CRITICA"
    },
    "CONTAMINACION": {
        "vision_labels": ["Dirty", "impurities"],
        "mvtec_labels": ["contamination", "metal_contamination"],
        "importancia": "ALTA"
    },
    "RAYONES_ARANAZOS": {
        "vision_labels": ["Scratch", "s_scratch", "t_scratch"],
        "mvtec_labels": ["scratch", "scratch_head"],
        "importancia": "MEDIA"
    },
    "PERFORACIONES": {
        "vision_labels": ["Hole", "missing_hole"],
        "mvtec_labels": ["hole", "cut"],
        "importancia": "CRITICA"
    },
    "DEFORMACIONES": {
        "vision_labels": ["short", "spur"],
        "mvtec_labels": ["bent", "bent_lead", "bent_wire"],
        "importancia": "ALTA"
    }
}
```

### Categorías/Componentes Incluidos

```python
# MVTec AD
categorias_mvtec = ["transistor", "metal_nut", "cable", "capsule", "hazelnut"]

# VISION-Datasets
componentes_vision = ["PCB_1", "PCB_2", "Console", "Electronics", "Cable", "Lens"]
```

---

## Flujo de Procesamiento

```
1. setup_output_structure()
   └── Crear carpetas images/ y metadata/

2. load_mvtec_annotations()
   └── Cargar samples.json de MVTec

3. filter_mvtec_samples()
   └── Filtrar por categorías y defectos objetivo

4. process_mvtec_samples()
   └── Convertir imágenes a PNG
   └── Crear entradas COCO

5. load_vision_annotations()
   └── Cargar _annotations.coco.json por componente

6. filter_vision_annotations()
   └── Filtrar por defectos objetivo

7. process_vision_samples()
   └── Convertir imágenes a PNG
   └── Preservar anotaciones con bboxes

8. save_coco_json()
   └── Guardar annotations.coco.json

9. save_metadata()
   └── Guardar dataset_info.json
```

---

## Estructura de Salida

```
output_directory/
├── images/                    # Todas las imágenes PNG
│   ├── 001-16.png            # Imágenes MVTec
│   ├── 000001.png            # Imágenes VISION
│   └── ...
├── annotations.coco.json      # Anotaciones COCO unificadas
└── metadata/
    └── dataset_info.json      # Metadatos del proceso
```

---

## Formato COCO Generado

### Estructura del JSON

```json
{
  "info": {
    "description": "TFG Dataset: VISION-Datasets + MVTec AD Curado",
    "version": "1.0",
    "date_created": "2025-09-21T..."
  },
  "images": [
    {
      "id": 1,
      "file_name": "001-16.png",
      "width": 1024,
      "height": 1024,
      "source_dataset": "mvtec",
      "original_category": "transistor",
      "defect_type": "bent_lead"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [0, 0, 1024, 1024],
      "area": 1048576,
      "source_dataset": "mvtec"
    }
  ],
  "categories": [
    {"id": 1, "name": "bent_lead", "supercategory": "mvtec_defect"},
    {"id": 2, "name": "Scratch", "supercategory": "vision_defect"}
  ]
}
```

---

## Resultados Esperados

| Métrica | Valor |
|---------|-------|
| Total imágenes | ~1,900 |
| Imágenes MVTec | ~1,640 (86%) |
| Imágenes VISION | ~267 (14%) |
| Categorías originales | 15 |
| Formato imágenes | PNG |

---

## Notas Importantes

1. **Imágenes MVTec "good"**: Se incluyen como categoría "normal"
2. **Bounding boxes MVTec**: Como no tienen localización específica, se usa el bbox de toda la imagen
3. **Manejo de duplicados de nombre**: Se añade sufijo numérico `_1`, `_2`, etc.
4. **Errores de conversión**: Se registran en `stats["conversion_errors"]`

---

## Siguiente Etapa

Ejecutar **Etapa 3** para analizar la calidad del dataset curado y detectar problemas.

