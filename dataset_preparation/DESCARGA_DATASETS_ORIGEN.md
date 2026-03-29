# Descarga y preparación de los datasets originales (VISION y MVTec AD)

Este documento recoge **enlaces oficiales**, condiciones de uso y cómo colocar los datos respecto a este repositorio para ejecutar el flujo en `flujo_curacion_dataset/`.

> **Licencias:** ambos conjuntos imponen condiciones (uso no comercial, citación, registro, etc.). Léelas en las páginas enlazadas antes de descargar.

---

## 1. VISION-Datasets (componentes industriales, anotaciones COCO)

### Referencias oficiales

| Recurso | Enlace |
|--------|--------|
| **Dataset en Hugging Face** | [https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets) |
| **Paper (benchmark)** | [VISION Datasets: A Benchmark for Vision-based InduStrial InspectiON](https://arxiv.org/abs/2306.07890) (arXiv:2306.07890) |

### Cómo obtener los ficheros

1. Crea una cuenta en [Hugging Face](https://huggingface.co/) si no la tienes.
2. Abre el dataset [VISION-Workshop/VISION-Datasets](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets).
3. Si el dataset está **restringido (gated)**, acepta las condiciones y solicita acceso según indique la página.
4. Descarga los archivos publicados (habitualmente **un `.tar.gz` por componente**, p. ej. `PCB_1.tar.gz`, `Electronics.tar.gz`, etc.).

### Estructura esperada por los scripts de exploración

El script `etapa1_exploracion/01_explorar_vision_dataset.py` espera un directorio que contenga:

- Ficheros `*.tar.gz` por componente **o** carpetas ya extraídas con el mismo nombre que el `.tar.gz`.
- Dentro de cada componente, carpetas tipo `train/`, `val/`, `inference/` con imágenes `.jpg` y JSON COCO de anotaciones.

Coloca todo en una carpeta, por ejemplo en la **raíz del repositorio**:

```text
ViT-Industrial-Defects/
  VISION-Datasets/          # <-- tu descarga / extracción aquí
    PCB_1/
    Electronics/
    ...
```

Ejemplo de ejecución (desde `etapa1_exploracion/`):

```bash
python 01_explorar_vision_dataset.py --vision-path ../../../VISION-Datasets
```

(Ajusta `--vision-path` a la ruta real en tu máquina.)

---

## 2. MVTec AD (anomalías industriales, máscaras pixel-level)

### Referencias oficiales

| Recurso | Enlace |
|--------|--------|
| **Página del dataset (MVTec)** | [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **Paper** | [MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf) |
| **Licencia** | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (entre otras condiciones: **no comercial**; revisa el texto completo en la web de MVTec) |

### Cómo obtener los ficheros

1. En la página [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), usa el apartado **Download Dataset** (el flujo exacto puede pedir registro o formulario según la política actual de MVTec).
2. Descarga el archivo que distribuyen (estructura habitual: **una carpeta por categoría** de objeto, con `train` / `test`, imágenes y máscaras).

### Formato esperado por **este** repositorio (`samples.json`)

Los scripts de la Fase 0 (`02_analizar_mvtec_dataset.py`, y la **Etapa 2** `01_dataset_curator.py`) están preparados para un directorio `mvtec-ad/` que incluya un fichero agregado:

```text
mvtec-ad/
  samples.json    # listado de muestras con metadatos (categoría, defecto, split, rutas, etc.)
  ...
```

El paquete estándar que publica MVTec **suele venir como árbol de carpetas**, no como un único `samples.json`. En el trabajo del TFG se utilizó una **vista consolidada** en JSON compatible con el curador. Si tu copia solo tiene el formato oficial por carpetas, necesitarás:

- **generar** un `samples.json` que siga el esquema que consumen los scripts, o  
- **adaptar** las rutas y la carga en `01_dataset_curator.py` / `02_analizar_mvtec_dataset.py` para leer directamente el layout MVTec.

Revisa los campos que lee `MVTecAnalyzer` y `DatasetCurator` en esos ficheros antes de reproducir la Etapa 2.

Sugerencia de ubicación en el clon:

```text
ViT-Industrial-Defects/
  mvtec-ad/
    samples.json
    ...
```

Ejemplo:

```bash
python 02_analizar_mvtec_dataset.py --mvtec-path ../../../mvtec-ad
```

---

## 3. Resumen de rutas recomendadas (fuera de Git)

Los datos crudos **no** deben versionarse en Git (tamaño y licencia). En `.gitignore` de la raíz del repo se ignoran por defecto entradas como `VISION-Datasets/` y `mvtec-ad/` cuando las coloques junto al clon.

| Dataset | Dónde colocarlo (ejemplo) | Variable / argumento |
|--------|---------------------------|----------------------|
| VISION | `<repo>/VISION-Datasets/` | `--vision-path` en `01_explorar_vision_dataset.py` |
| MVTec AD | `<repo>/mvtec-ad/` | `--mvtec-path` en `02_analizar_mvtec_dataset.py` |

---

## 4. Cita bibliográfica (memoria / artículo)

Incluye al menos las referencias oficiales del benchmark VISION y del dataset MVTec AD cuando cites el origen de los datos; los enlaces del apartado 1 y 2 cubren paper y páginas de proyecto.

---

**Última revisión:** documentación alineada con el flujo en `dataset_preparation/flujo_curacion_dataset/` y la memoria técnica del TFG.
