# Fase 0 — Preparación y curación del dataset unificado

Esta carpeta reproduce la **documentación y el pipeline** utilizados para construir el dataset curado en formato COCO (fuentes **VISION-Datasets** + **MVTec AD**, taxonomía unificada, balanceo y splits) descrito en la memoria técnica del TFG.

## Contenido

| Ruta | Descripción |
|------|-------------|
| [DESCARGA_DATASETS_ORIGEN.md](DESCARGA_DATASETS_ORIGEN.md) | **Cómo descargar** VISION y MVTec AD (enlaces oficiales, licencias, rutas recomendadas). |
| [DOCUMENTACION_CURACION_DATASET.md](DOCUMENTACION_CURACION_DATASET.md) | Documento largo: etapas 1–5, métricas y estructura del dataset resultante. |
| [flujo_curacion_dataset/README.md](flujo_curacion_dataset/README.md) | Guía operativa de scripts por etapa y dependencias. |
| [flujo_curacion_dataset/](flujo_curacion_dataset/) | Scripts Python por etapa (`etapa1_exploracion` … `etapa5_analisis_final`). |
| [outputs/](outputs/) | Salidas de ejemplo del análisis (CSV, reportes) **versionadas** en el repositorio; los datasets masivos crudos no están en Git. |

## Inicio rápido

1. Lee **[DESCARGA_DATASETS_ORIGEN.md](DESCARGA_DATASETS_ORIGEN.md)** y coloca **VISION-Datasets** y **mvtec-ad** fuera del control de versiones (p. ej. en la raíz del clon; ver `.gitignore`).
2. Crea un entorno e instala dependencias:

```bash
cd dataset_preparation
python -m venv venv_curacion
source venv_curacion/bin/activate
pip install -r requirements.txt
```

3. Sigue el orden de etapas en [flujo_curacion_dataset/README.md](flujo_curacion_dataset/README.md).

## Dataset final usado en entrenamiento

El nombre interno del split curado referenciado en el resto del repo es:

`curated_dataset_splitted_20251101_provisional_1st_version/`

Ese directorio **no** se sube a Git; se obtiene ejecutando el pipeline hasta la etapa 4 (y opcionalmente 5) o copiando el artefacto desde el entorno donde se generó.

---

*TFG 2025-26 — preparación de datos para detección de defectos industriales.*
