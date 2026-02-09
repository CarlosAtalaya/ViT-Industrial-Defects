# 📸 Instrucciones para Módulos de Visualización y Dashboard

Este documento proporciona instrucciones paso a paso para:
1. Estandarizar módulos de visualización para cada arquitectura
2. Integrar visualizaciones en el dashboard comparativa

---

## Parte 1: Estandarizar Módulos de Visualización

### Objetivo
Crear scripts de visualización estandarizados que acepten checkpoint por argumento y generen visualizaciones consistentes para todas las arquitecturas.

### Estructura Objetivo

```
scripts/
├── deimv2_multimodal/
│   └── visualize_predictions.py (ya existe, necesita estandarización)
├── efficientnet/
│   └── visualize_predictions.py (ya existe, necesita estandarización)
└── resnet18/
    └── visualize_predictions.py (ya existe, necesita estandarización)
```

### Paso 1: Estandarizar Script de DEIMv2

**Archivo:** `scripts/deimv2_multimodal/visualize_predictions.py`

**Cambios necesarios:**

1. **Asegurar que acepta checkpoint por argumento:**
   ```python
   parser.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al checkpoint del modelo')
   ```

2. **Asegurar que acepta carpeta de imágenes:**
   ```python
   parser.add_argument('--img-folder', type=str, required=True,
                       help='Carpeta con imágenes de test a visualizar')
   ```

3. **Asegurar formato de salida consistente:**
   - Guardar en: `{checkpoint_dir}/visualizations_test/`
   - Nombre de archivo: `{image_name}_prediction.png`
   - Formato: Imagen lado a lado (Ground Truth | Predictions)

4. **Parámetros estándar:**
   ```python
   parser.add_argument('--score-threshold', type=float, default=0.15,
                       help='Score threshold para filtrar detecciones')
   parser.add_argument('--num-images', type=int, default=20,
                       help='Número de imágenes a visualizar')
   parser.add_argument('--random', action='store_true',
                       help='Seleccionar imágenes aleatoriamente')
   ```

### Paso 2: Estandarizar Script de EfficientNet

**Archivo:** `scripts/efficientnet/visualize_predictions.py`

**Cambios necesarios:**

1. **Modificar para aceptar checkpoint directamente:**
   ```python
   parser.add_argument('--checkpoint', type=str, required=True,
                       help='Ruta al checkpoint del modelo (.pth)')
   ```

2. **Asegurar que carga el modelo correctamente:**
   ```python
   # Cargar checkpoint
   checkpoint = torch.load(args.checkpoint, map_location=device)
   model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
   model.eval()
   ```

3. **Asegurar formato de salida consistente:**
   - Guardar en: `{checkpoint_dir}/visualizations_test/`
   - Mismo formato que DEIMv2

### Paso 3: Estandarizar Script de ResNet-18

**Archivo:** `scripts/resnet18/visualize_predictions.py`

**Cambios necesarios:**

1. **Mismos cambios que EfficientNet**
2. **Asegurar compatibilidad con diferentes formatos de checkpoint**

### Paso 4: Crear Script Helper Unificado

**Archivo:** `scripts/visualize_any_model.py` (nuevo)

Este script unificado detecta automáticamente la arquitectura y llama al script correspondiente:

```python
#!/usr/bin/env python3
"""
Script unificado para visualizar predicciones de cualquier modelo.
Detecta automáticamente la arquitectura y llama al script correspondiente.
"""

import argparse
import subprocess
from pathlib import Path

def detect_architecture(checkpoint_path):
    """Detecta la arquitectura basándose en la ruta del checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    # Buscar en la ruta
    if 'deimv2' in str(checkpoint_path).lower():
        return 'deimv2'
    elif 'efficientnet' in str(checkpoint_path).lower():
        return 'efficientnet'
    elif 'resnet' in str(checkpoint_path).lower():
        return 'resnet18'
    else:
        # Intentar detectar por contenido del checkpoint
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Lógica de detección...
        return 'unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img-folder', required=True)
    parser.add_argument('--score-threshold', type=float, default=0.15)
    parser.add_argument('--num-images', type=int, default=20)
    parser.add_argument('--random', action='store_true')
    
    args = parser.parse_args()
    
    arch = detect_architecture(args.checkpoint)
    
    if arch == 'deimv2':
        script = 'scripts/deimv2_multimodal/visualize_deimv2_predictions.py'
    elif arch == 'efficientnet':
        script = 'scripts/efficientnet/visualize_predictions.py'
    elif arch == 'resnet18':
        script = 'scripts/resnet18/visualize_predictions.py'
    else:
        raise ValueError(f"Arquitectura no reconocida: {arch}")
    
    # Ejecutar script correspondiente
    subprocess.run([
        'python3', script,
        '--checkpoint', args.checkpoint,
        '--img-folder', args.img-folder,
        '--score-threshold', str(args.score_threshold),
        '--num-images', str(args.num_images),
        '--random' if args.random else ''
    ])

if __name__ == '__main__':
    main()
```

---

## Parte 2: Integrar Visualizaciones en el Dashboard

### Objetivo
Añadir una nueva sección en el dashboard que permita visualizar predicciones de diferentes modelos de forma comparativa.

### Paso 1: Estructura de Datos para Visualizaciones

**Archivo:** `herramienta_comparativa/data/experiments_metadata.json`

Añadir campo `visualizations_path` a cada experimento:

```json
{
  "deimv2_1024_300ep": {
    ...
    "visualizations_path": "fase2_vit/deimv2_1024_300ep/visualizations_test",
    "visualizations_available": true
  }
}
```

### Paso 2: Función para Cargar Visualizaciones

**Archivo:** `herramienta_comparativa/dashboard.py`

Añadir función:

```python
def get_visualization_images(exp_path, num_images=10):
    """Obtiene las imágenes de visualización de un experimento."""
    vis_path = DATA_PATH / exp_path / "visualizations_test"
    if not vis_path.exists():
        return []
    
    # Obtener todas las imágenes PNG
    images = sorted(vis_path.glob("*.png"))
    return images[:num_images]
```

### Paso 3: Nueva Vista de Visualizaciones

**Archivo:** `herramienta_comparativa/dashboard.py`

Añadir nueva función `render_visualizations()`:

```python
def render_visualizations(metadata):
    """Vista 6: Visualizaciones Comparativas"""
    st.title("🖼️ Visualizaciones Comparativas")
    st.markdown("Comparación visual de predicciones entre diferentes modelos")
    
    st.markdown("---")
    
    # Selector de experimentos a comparar
    all_experiments = {}
    for phase_id, experiments in metadata["experiments"].items():
        for exp_id, exp_info in experiments.items():
            if exp_info.get("visualizations_available", False):
                all_experiments[exp_info["name"]] = (exp_id, exp_info)
    
    if not all_experiments:
        st.warning("No hay visualizaciones disponibles. Ejecuta los scripts de visualización primero.")
        return
    
    # Selector múltiple de experimentos
    selected_experiments = st.multiselect(
        "Selecciona modelos a comparar (máximo 3):",
        options=list(all_experiments.keys()),
        default=list(all_experiments.keys())[:min(3, len(all_experiments))],
        max_selections=3
    )
    
    if not selected_experiments:
        st.info("Selecciona al menos un modelo para visualizar")
        return
    
    # Selector de imágenes
    st.markdown("### 📸 Selección de Imágenes")
    
    # Obtener todas las imágenes disponibles del primer experimento
    first_exp_id, first_exp_info = all_experiments[selected_experiments[0]]
    all_images = get_visualization_images(first_exp_info["path"])
    
    if not all_images:
        st.warning(f"No se encontraron visualizaciones para {selected_experiments[0]}")
        return
    
    # Crear lista de nombres de imágenes (sin extensión y sin _prediction)
    image_names = [img.stem.replace('_prediction', '') for img in all_images]
    
    selected_image_name = st.selectbox(
        "Selecciona una imagen para comparar:",
        options=image_names,
        index=0
    )
    
    st.markdown("---")
    
    # Mostrar visualizaciones lado a lado
    st.markdown(f"### Comparación: {selected_image_name}")
    
    cols = st.columns(len(selected_experiments))
    
    for idx, exp_name in enumerate(selected_experiments):
        exp_id, exp_info = all_experiments[exp_name]
        
        with cols[idx]:
            st.markdown(f"**{exp_name}**")
            
            # Buscar imagen correspondiente
            vis_path = DATA_PATH / exp_info["path"] / "visualizations_test"
            image_file = vis_path / f"{selected_image_name}_prediction.png"
            
            if image_file.exists():
                st.image(Image.open(image_file), use_container_width=True)
                
                # Mostrar métricas del modelo si están disponibles
                results = load_experiment_results(exp_info["path"])
                if results:
                    st.caption(f"mAP: {results.get('mAP', 0):.3f}")
            else:
                st.warning(f"Imagen no encontrada para {exp_name}")
    
    # Galería de todas las imágenes
    st.markdown("---")
    st.markdown("### 🖼️ Galería de Visualizaciones")
    
    # Selector de modelo para la galería
    gallery_model = st.selectbox(
        "Selecciona modelo para la galería:",
        options=selected_experiments,
        index=0
    )
    
    gallery_exp_id, gallery_exp_info = all_experiments[gallery_model]
    gallery_images = get_visualization_images(gallery_exp_info["path"], num_images=30)
    
    if gallery_images:
        # Mostrar en grid
        num_cols = 3
        num_rows = (len(gallery_images) + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col in range(num_cols):
                idx = row * num_cols + col
                if idx < len(gallery_images):
                    with cols[col]:
                        img = Image.open(gallery_images[idx])
                        st.image(img, use_container_width=True)
                        st.caption(gallery_images[idx].stem.replace('_prediction', ''))
```

### Paso 4: Añadir a Navegación

**Archivo:** `herramienta_comparativa/dashboard.py`

En la función `main()`, añadir a `pages`:

```python
pages = {
    "🏠 Inicio": render_home,
    "📜 Línea Temporal": render_timeline,
    "🔬 Explorador": render_explorer,
    "📊 Comparativa": render_comparison,
    "🖼️ Visualizaciones": render_visualizations,  # NUEVO
    "📝 Conclusiones": render_conclusions
}
```

---

## Parte 3: Scripts de Generación de Visualizaciones

### Script para Generar Visualizaciones de Todos los Modelos

**Archivo:** `scripts/generate_all_visualizations.sh` (nuevo)

```bash
#!/bin/bash

# Script para generar visualizaciones de todos los modelos entrenados

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="${PROJECT_ROOT}/curated_dataset_splitted_20251101_provisional_1st_version"
TEST_IMG_FOLDER="${DATASET_PATH}/test/images"
TEST_ANN_FILE="${DATASET_PATH}/test/test.json"

echo "=========================================="
echo "Generando Visualizaciones de Todos los Modelos"
echo "=========================================="
echo ""

# DEIMv2 - Mejor modelo
echo "📸 Generando visualizaciones DEIMv2..."
python3 scripts/deimv2_multimodal/visualize_deimv2_predictions.py \
    --checkpoint scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth \
    --config scripts/deimv2_multimodal/configs/deimv2_industrial_defects.yml \
    --img-folder "$TEST_IMG_FOLDER" \
    --ann-file "$TEST_ANN_FILE" \
    --num-images 30 \
    --random \
    --score-threshold 0.15

# ResNet-18 - Mejor modelo
echo ""
echo "📸 Generando visualizaciones ResNet-18..."
python3 scripts/resnet18/visualize_predictions.py \
    --checkpoint scripts/resnet18/results/training/resnet18_fasterrcnn_*/checkpoints/best_checkpoint.pth \
    --dataset-path "$DATASET_PATH" \
    --split test \
    --num-images 30 \
    --random \
    --score-threshold 0.5

# EfficientNet - Mejor modelo
echo ""
echo "📸 Generando visualizaciones EfficientNet..."
python3 scripts/efficientnet/visualize_predictions.py \
    --checkpoint scripts/efficientnet/results/training/efficientnet_b0_fasterrcnn_*/checkpoints/best_checkpoint.pth \
    --dataset-path "$DATASET_PATH" \
    --split test \
    --num-images 30 \
    --random \
    --score-threshold 0.5

echo ""
echo "✅ Visualizaciones generadas para todos los modelos"
```

---

## Parte 4: Checklist de Implementación

### ✅ Para cada arquitectura:

- [ ] Verificar que `visualize_predictions.py` acepta `--checkpoint` como argumento
- [ ] Verificar que acepta `--img-folder` o equivalente
- [ ] Verificar que guarda en formato consistente: `{checkpoint_dir}/visualizations_test/`
- [ ] Verificar que genera imágenes con formato: `{image_name}_prediction.png`
- [ ] Verificar que muestra Ground Truth y Predictions lado a lado
- [ ] Probar con checkpoint real y generar al menos 20 visualizaciones

### ✅ Para el Dashboard:

- [ ] Añadir campo `visualizations_path` a `experiments_metadata.json`
- [ ] Añadir función `get_visualization_images()` en `dashboard.py`
- [ ] Añadir función `render_visualizations()` en `dashboard.py`
- [ ] Añadir "Visualizaciones" a la navegación
- [ ] Probar que carga y muestra imágenes correctamente
- [ ] Probar comparación lado a lado de múltiples modelos

### ✅ Testing:

- [ ] Generar visualizaciones para al menos 2 modelos diferentes
- [ ] Verificar que aparecen en el dashboard
- [ ] Verificar que la comparación lado a lado funciona
- [ ] Verificar que la galería muestra imágenes correctamente

---

## Parte 5: Consejos y Mejores Prácticas

### 1. Formato de Imágenes

- **Tamaño recomendado:** 1024x1024 o mantener aspect ratio original
- **Formato:** PNG para preservar calidad
- **Layout:** Ground Truth a la izquierda, Predictions a la derecha
- **Colores:** Usar colores consistentes para cada clase en todos los modelos

### 2. Nombres de Archivos

- **Formato estándar:** `{image_id}_prediction.png`
- **Ejemplo:** `000012_prediction.png`
- Esto facilita la comparación entre modelos

### 3. Selección de Imágenes

- **Diversidad:** Seleccionar imágenes que representen diferentes clases
- **Casos interesantes:** Incluir imágenes con múltiples defectos
- **Casos difíciles:** Incluir imágenes donde los modelos difieren

### 4. Performance

- **Caché:** Usar `@st.cache_data` para cargar imágenes
- **Lazy loading:** Cargar imágenes solo cuando se necesiten
- **Thumbnails:** Considerar generar thumbnails para la galería

### 5. Interactividad

- **Filtros:** Permitir filtrar por clase de defecto
- **Búsqueda:** Permitir buscar por nombre de imagen
- **Zoom:** Considerar añadir zoom a las imágenes

---

## Parte 6: Ejemplo de Uso

### Generar visualizaciones para un modelo específico:

```bash
# DEIMv2
python3 scripts/deimv2_multimodal/visualize_deimv2_predictions.py \
    --checkpoint scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth \
    --config scripts/deimv2_multimodal/configs/deimv2_industrial_defects.yml \
    --img-folder curated_dataset_splitted_20251101_provisional_1st_version/test/images \
    --ann-file curated_dataset_splitted_20251101_provisional_1st_version/test/test.json \
    --num-images 30 \
    --random \
    --score-threshold 0.15

# ResNet-18
python3 scripts/resnet18/visualize_predictions.py \
    --checkpoint scripts/resnet18/results/training/resnet18_fasterrcnn_20251208_004809/checkpoints/best_checkpoint.pth \
    --dataset-path curated_dataset_splitted_20251101_provisional_1st_version \
    --split test \
    --num-images 30 \
    --random \
    --score-threshold 0.5

# EfficientNet
python3 scripts/efficientnet/visualize_predictions.py \
    --checkpoint scripts/efficientnet/results/training/efficientnet_b0_fasterrcnn_20251208_011406/checkpoints/best_checkpoint.pth \
    --dataset-path curated_dataset_splitted_20251101_provisional_1st_version \
    --split test \
    --num-images 30 \
    --random \
    --score-threshold 0.5
```

### Ver en el dashboard:

1. Ejecutar: `streamlit run herramienta_comparativa/dashboard.py`
2. Ir a la sección "🖼️ Visualizaciones"
3. Seleccionar modelos a comparar
4. Seleccionar imagen específica para comparación lado a lado
5. Explorar galería completa

---

## Notas Finales

- **Consistencia:** Asegurar que todos los scripts usan el mismo formato de salida
- **Documentación:** Documentar cualquier diferencia entre arquitecturas
- **Mantenimiento:** Actualizar visualizaciones cuando se re-entrene un modelo
- **Versionado:** Considerar versionar las visualizaciones con el checkpoint usado



