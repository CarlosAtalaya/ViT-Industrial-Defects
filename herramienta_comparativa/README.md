# 🔬 Dashboard de Comparación de Arquitecturas

Herramienta interactiva e **independiente** para visualizar y comparar los resultados de experimentación del TFG sobre **Detección de Defectos Industriales con Vision Transformers vs CNNs**.

Diseñada para poder exportarse como un paquete completo y funcionar en un repositorio separado sin dependencias del proyecto principal.

---

## 🚀 Uso Rápido

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar el dashboard
streamlit run dashboard.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

**Alternativa:** Usa los scripts de configuración automática en la carpeta `scripts/` (ver más abajo).

---

## 📦 Dependencias (Librerías)

| Paquete     | Versión mínima | Uso                              |
|-------------|----------------|----------------------------------|
| streamlit   | 1.28.0         | Interfaz web del dashboard       |
| pandas      | 2.0.0          | Manipulación de tablas y datos   |
| plotly      | 5.18.0         | Gráficos interactivos            |
| Pillow      | 10.0.0         | Carga y procesamiento de imágenes|
| matplotlib  | 3.7.0          | Dibujo de bounding boxes         |
| numpy       | 1.24.0         | Cálculos numéricos               |

---

## 📁 Estructura de la Herramienta

```
herramienta_comparativa/
├── dashboard.py              # Aplicación principal (Streamlit)
├── requirements.txt         # Dependencias Python
├── README.md                 # Este archivo
├── FASE_EXPERIMENTACION.md   # Documentación de fases
│
├── data/                     # Datos necesarios (todo autocontenido)
│   ├── experiments_metadata.json    # Metadatos de experimentos
│   ├── test.json                    # Anotaciones COCO (ground truth)
│   │
│   ├── images_selected_for_visualize/  # Imágenes para visualización comparativa
│   │   ├── raw/                      # Imágenes .jpg/.png seleccionadas
│   │   └── predictions/              # Predicciones por arquitectura
│   │       ├── resnet18/predictions_all.json
│   │       ├── efficientnet/predictions_all.json
│   │       └── deimv2/predictions_all.json
│   │
│   ├── fase1_baseline/        # Resultados CNNs nativas
│   ├── fase2_vit/             # Resultados Vision Transformers
│   └── fase3_comparacion_justa/ # Resultados CNNs @ 1024px
│
└── scripts/                  # Scripts de configuración y arranque
    ├── setup.sh              # Configuración automática (Linux/macOS)
    └── setup.bat             # Configuración automática (Windows)
```

### Prioridad de Rutas de Datos

El dashboard busca los datos en este orden (sin depender de rutas externas al paquete de la herramienta):

1. **Predicciones:** `data/predictions/{arquitectura}_predictions.json` (export) o `data/images_selected_for_visualize/predictions/{resnet18,efficientnet,deimv2}/predictions_all.json`
2. **Ground truth COCO:** `data/ground_truth.json` (export), luego `data/images_selected_for_visualize/test.json`, luego `data/test.json`
3. **Imágenes raw:** `data/images_selected/` (export) o `data/images_selected_for_visualize/raw/`

---

## 📋 Contenido del Dashboard

| Sección | Descripción |
|---------|-------------|
| **🏠 Inicio** | Contexto del proyecto, metodología, descripción de arquitecturas (ResNet-18, EfficientNet-B0, DEIMv2) |
| **📜 Línea Temporal** | Evolución cronológica de las 4 fases de experimentación |
| **🔬 Explorador** | Análisis detallado por experimento: configuración, métricas (AP, Precision, Recall), curvas de entrenamiento |
| **📊 Comparativa** | Comparación directa entre arquitecturas con filtros y análisis de thresholds |
| **🖼️ Visualizaciones** | Comparación visual interactiva: Ground Truth vs predicciones con threshold dinámico |
| **📝 Conclusiones** | Tabla resumen, hallazgos principales y recomendaciones |

---

## 📊 Resultados Principales

| Arquitectura | Mejor Configuración | mAP@0.5 |
|--------------|---------------------|---------|
| ResNet-18 | 1024x1024 | 0.080 |
| EfficientNet-B0 | Nativa | 0.162 |
| **DEIMv2 (ViT)** | **1024x1024, 300ep** | **0.785** ⭐ |

**Conclusión:** La arquitectura Vision Transformer (DEIMv2) supera significativamente a las CNNs tradicionales para la detección de defectos industriales.

---

## 🛠️ Scripts de Configuración Automática

La carpeta `scripts/` incluye scripts para configurar el entorno y lanzar la herramienta con un solo comando.

### Linux / macOS (`scripts/setup.sh`)

```bash
cd herramienta_comparativa
chmod +x scripts/setup.sh
./scripts/setup.sh
```

El script:
1. Crea un entorno virtual Python si no existe
2. Activa el entorno e instala dependencias
3. Lanza el dashboard automáticamente

### Windows (`scripts/setup.bat`)

```cmd
cd herramienta_comparativa
scripts\setup.bat
```

Misma funcionalidad para Windows.

---

## 📦 Exportar como Herramienta Independiente

Para distribuir la herramienta en otro repositorio:

1. **Copia toda la carpeta** `herramienta_comparativa/`
2. Asegúrate de que `data/images_selected_for_visualize/` contiene:
   - `raw/` con imágenes
   - `predictions/` con los JSON de cada arquitectura
3. Incluye `data/test.json` (o `data/images_selected_for_visualize/test.json`) para las anotaciones ground truth
4. El dashboard solo necesita la carpeta `herramienta_comparativa/`; no requiere el dataset curado completo en la raíz del repo

---

## 🛠️ Requisitos

- **Python:** 3.8 o superior
- **Sistema:** Cualquier SO con Python (Linux, macOS, Windows)

---
*TFG 2025-26 - Detección de Defectos Industriales con Vision Transformers*
