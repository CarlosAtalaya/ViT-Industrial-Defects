# ViT Industrial Defects — Experimentación y comparativa de arquitecturas

Repositorio de código asociado al **Trabajo de Fin de Grado (curso 2025-26)** y a la línea de investigación en curso sobre **detección de defectos industriales** en componentes electrónicos, con comparación sistemática entre **CNNs** (ResNet-18, EfficientNet-B0 + Faster R-CNN) y **Vision Transformers** (DEIMv2 con backbone preentrenado tipo DINOv3).

La memoria técnica del TFG (contexto industrial, objetivos, metodología y resultados) está recogida en el proyecto LaTeX independiente; este repositorio concentra **código, configuraciones y datos empaquetados** necesarios para reproducir la experimentación y explorar resultados de forma interactiva.

---

## Qué incluye este repositorio

| Área | Descripción |
|------|-------------|
| **Entrenamiento y evaluación** | Scripts por arquitectura en `scripts/resnet18/`, `scripts/efficientnet/`, `scripts/deimv2_multimodal/` (pipelines, evaluación COCO, visualización de predicciones). |
| **Dashboard comparativo** | Aplicación Streamlit autocontenida en `herramienta_comparativa/` con histórico de fases, métricas y visualización dinámica Ground Truth vs predicciones. |
| **Experimentación multimodal (exploratoria)** | Código adicional bajo `demo-Multimodal/` (extensiones no necesarias para la línea principal ResNet / EfficientNet / DEIMv2). |
| **Utilidades de dataset** | Scripts en `scripts/re-analyze-1st-dataset-version/` para análisis del conjunto de datos. |

---

## Requisitos generales

- **Python 3.8+** (3.10 recomendado para PyTorch reciente).
- **GPU NVIDIA** con CUDA compatible si se desea entrenar o evaluar en tiempos razonables (el trabajo original usó una RTX 4070 12 GB).
- Dependencias concretas **no están unificadas en un único `requirements.txt` en la raíz**: cada subsistema fija sus paquetes (PyTorch, torchvision, dependencias DEIMv2, etc.). Para el dashboard, usa `herramienta_comparativa/requirements.txt`.

---

## Instalación rápida (dashboard)

La forma más directa de explorar la comparativa sin reentrenar modelos:

```bash
cd herramienta_comparativa
python -m venv venv
source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
streamlit run dashboard.py
```

Documentación detallada: [herramienta_comparativa/README.md](herramienta_comparativa/README.md).

---

## Reproducir la experimentación (entrenamiento / evaluación)

1. **Dataset**  
   El directorio del dataset curado y particionado (`curated_dataset_splitted_20251101_provisional_1st_version/`) está **excluido del control de versiones** (tamaño y política de datos). Debes colocarlo en la **raíz del repositorio** con la estructura esperada por los scripts (`train/`, `test/`, anotaciones COCO, etc.) o ajustar las variables `DATASET_PATH` en los `run_pipeline.sh` y rutas equivalentes en Python.

2. **Por arquitectura**  
   - ResNet-18: [scripts/resnet18/README.md](scripts/resnet18/README.md) y `run_pipeline.sh`.  
   - EfficientNet: [scripts/efficientnet/README.md](scripts/efficientnet/README.md) y `run_pipeline.sh`.  
   - DEIMv2: [scripts/deimv2_multimodal/README.md](scripts/deimv2_multimodal/README.md), configuración YAML y scripts de entrenamiento/evaluación.

3. **Visualización de predicciones y alineación con el dashboard**  
   Instrucciones unificadas: [INSTRUCCIONES_VISUALIZACION.md](INSTRUCCIONES_VISUALIZACION.md).

4. **Metadatos de experimentos**  
   Los resultados agregados que alimentan el dashboard viven bajo `herramienta_comparativa/data/` (`experiments_metadata.json`, carpetas `fase1_baseline/`, `fase3_comparacion_justa/`, etc.).

---

## Datos empaquetados para visualización en el dashboard

Para la sección **Visualizaciones** del Streamlit se usa el árbol bajo **`herramienta_comparativa/data/images_selected_for_visualize/`**:

- `raw/` — imágenes de ejemplo (`.png` / `.jpg`).
- `predictions/{resnet18,efficientnet,deimv2}/predictions_all.json` — predicciones por modelo.
- Opcional: `test.json` — anotaciones COCO del subconjunto (si no está, puede usarse `herramienta_comparativa/data/test.json`).

El dashboard **no** depende de rutas externas tipo `curated_dataset_.../test/images_selected_for_visualize`; todo debe quedar bajo esa carpeta de la herramienta (o en los formatos export alternativos descritos en su README).

---

## Documentación adicional en el repositorio

| Documento | Contenido |
|-----------|-----------|
| [INSTRUCCIONES_VISUALIZACION.md](INSTRUCCIONES_VISUALIZACION.md) | Flujo de visualización por arquitectura e integración con el dashboard. |
| [herramienta_comparativa/FASE_EXPERIMENTACION.md](herramienta_comparativa/FASE_EXPERIMENTACION.md) | Narrativa de fases de experimentación y resultados resumidos. |

---

## Solución de problemas (troubleshooting)

### El dashboard no muestra imágenes ni predicciones

- Comprueba que exista `herramienta_comparativa/data/images_selected_for_visualize/raw/` con ficheros de imagen y que los JSON estén en `.../predictions/<arquitectura>/predictions_all.json`.
- Verifica que lanzas Streamlit desde `herramienta_comparativa/` o que la ruta de trabajo permite resolver `data/` (ver README de la herramienta).
- Activa depuración: `STREAMLIT_DEBUG=1 streamlit run dashboard.py` para mensajes extra en consola.

### Ground truth vacío o clases genéricas

- Asegura un `test.json` COCO válido en `herramienta_comparativa/data/test.json` o en `data/images_selected_for_visualize/test.json`.

### Entrenamiento falla por CUDA / memoria

- Reduce `batch size`, resolución de entrada o número de workers en los scripts YAML / Python correspondientes.
- Confirma versión de PyTorch compatible con tu driver NVIDIA (`nvidia-smi`).

### Rutas absolutas en ficheros de configuración antiguos

- Algunos `config.json` bajo `herramienta_comparativa/data/` pueden contener rutas absolutas de máquina de desarrollo; sustitúyelas por la ruta base de tu clon o por rutas relativas al repositorio.

### Dependencias DEIMv2 / entorno multimodal

- La carpeta `demo-Multimodal/` puede tener requisitos adicionales; revisa cada `README` local antes de ejecutar.

---

## Referencia académica

Si citas este trabajo, enlaza el repositorio y la memoria del TFG correspondiente. Los resultados clave reportados en la memoria incluyen, entre otros, **mAP@0.5 ≈ 0.785** para la mejor configuración DEIMv2 frente a baselines CNN bajo la metodología descrita en el documento.

---

## Autoría

Proyecto académico (TFG 2025-26) — detección de defectos industriales con Vision Transformers y CNNs. Para dudas sobre reproducibilidad, abre un issue en el repositorio o contacta al mantenedor del proyecto.
