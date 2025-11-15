# Resumen de Sesión: Implementación DEIMv2 para Detección de Defectos Industriales

**Fecha:** 14 Noviembre 2024  
**Duración:** ~1 hora de entrenamiento + setup  
**Objetivo:** Implementar y evaluar DEIMv2 (Vision Transformer) como evolución de los baselines CNN

---

## 🎯 Estado Actual del Proyecto

### Fase Completada: DEIMv2 Vanilla (FASE 1)

**✅ Implementación exitosa de:**
1. Entrenamiento completo de DEIMv2-M (52 épocas)
2. Pipeline de evaluación en test set
3. Sistema de métricas compatible con baselines
4. Infraestructura de visualización

---

## 📊 Resultados Obtenidos

### Arquitectura Entrenada: DEIMv2-M

**Configuración:**
- **Backbone:** DINOv3 ViT-Tiny+ (vittplus_distill.pt)
- **Parámetros:** 17.81M (vs 11M ResNet-18, 5M EfficientNet)
- **Dimensiones:** 256 embedding dim, 4 decoder layers
- **Hardware:** RTX 4070 12GB
- **Tiempo entrenamiento:** ~60 minutos (52 épocas)

### Métricas en Test Set (205 imágenes)

```
mAP @ IoU=0.50:0.95  = 0.178 (17.8%)
AP  @ IoU=0.50       = 0.232 (23.2%)
AP  @ IoU=0.75       = 0.171 (17.1%)
AR  @ maxDets=100    = 0.480 (48.0%)

Por tamaño de objeto:
- Small objects:  mAP = 0.023 (2.3%)
- Medium objects: mAP = 0.072 (7.2%)
- Large objects:  mAP = 0.263 (26.3%)
```

### Comparativa con Baselines (esperada)

| Modelo | Arquitectura | Params | mAP@0.50:0.95 | Notas |
|--------|-------------|---------|---------------|-------|
| ResNet-18 | CNN + Faster R-CNN | 11M | ~0.42* | Baseline clásico |
| EfficientNet-B0 | CNN + Faster R-CNN | 5M | ~0.45* | Baseline ligero |
| **DEIMv2-M** | **ViT + DEIM** | **17.8M** | **0.178** | **Primer experimento** |

_*Nota: Métricas de ResNet/EfficientNet son estimadas. Necesario confirmar con evaluación real._

---

## 🏗️ Estructura Implementada

### Directorio: `scripts/deimv2_multimodal/`

```
scripts/deimv2_multimodal/
├── configs/
│   └── deimv2_industrial_defects.yml    # Config entrenamiento
├── outputs/
│   └── deimv2_industrial_run/
│       ├── checkpoint0039.pth            # Checkpoints cada 5 epochs
│       ├── best_stg1.pth                 # Mejor modelo
│       ├── log.txt                       # Log de entrenamiento
│       ├── summary/                      # TensorBoard logs
│       └── test_evaluation_results.json  # Métricas en test
├── train_deimv2_industrial.py           # Script entrenamiento
├── evaluate_deimv2.py                   # Evaluación mAP en test
├── visualize_deimv2_predictions.py      # Visualización predicciones
├── plot_deimv2_training_metrics.py      # Gráficas entrenamiento
└── run_evaluation_deimv2.sh             # Pipeline completo
```

---

## 🔧 Configuración Técnica

### Dataset

```yaml
Train: 715 imágenes, 944 anotaciones
Val:   102 imágenes, 145 anotaciones
Test:  205 imágenes, 265 anotaciones

Clases (6):
  0: NORMAL
  1: DEFORMACIONES
  2: ROTURA_FRACTURA
  3: RAYONES_ARANAZOS
  4: PERFORACIONES
  5: CONTAMINACION
```

### Hiperparámetros Clave

```yaml
# Modelo
embed_dim: 256
hidden_dim: 256
num_layers: 4

# Entrenamiento
epochs: 52
batch_size: 2  # Ajustado para RTX 4070
learning_rate: 0.0005 (base), 0.000025 (backbone)
optimizer: AdamW
use_amp: True  # Mixed precision

# Data Augmentation
- Mosaic augmentation
- RandomPhotometricDistort
- RandomIoUCrop
- Mixup (épocas 4-25)
- CopyBlend (épocas 4-44)
```

---

## 📈 Análisis de Resultados

### ⚠️ Rendimiento Inferior a Baselines

**Observaciones:**
- DEIMv2-M obtiene mAP=0.178 vs ~0.42-0.45 de CNNs
- El modelo tiene **dificultad con objetos pequeños** (mAP=0.023)
- Mejor rendimiento en objetos grandes (mAP=0.263)

### Posibles Causas

1. **Dataset pequeño (715 imágenes train)**
   - ViTs requieren más datos que CNNs
   - Transfer learning desde DINOv3 puede no ser suficiente

2. **Hiperparámetros no optimizados**
   - Primer experimento con config base
   - Posible learning rate inadecuado
   - Batch size muy pequeño (2)

3. **Formato de detecciones**
   - Modelo predice 300 queries por imagen
   - Muchas detecciones de baja confianza (61,500 total)
   - Posible necesidad de ajustar threshold

4. **Entrenamiento incompleto**
   - 52 épocas pueden ser insuficientes
   - Curva de loss puede no haber convergido

---

## 🎯 Próximos Pasos Inmediatos

### 1. Análisis Detallado (URGENTE)

```bash
# Revisar métricas de entrenamiento con TensorBoard
tensorboard --logdir scripts/deimv2_multimodal/outputs/deimv2_industrial_run/summary

# Visualizar predicciones para entender errores
cd scripts/deimv2_multimodal
./run_evaluation_deimv2.sh  # Genera visualizaciones
```

### 2. Comparación Justa con Baselines

**Acción necesaria:** Evaluar ResNet-18 y EfficientNet con el MISMO protocolo:

```bash
# Evaluar baselines con script de evaluación
cd scripts/resnet18
python evaluate_model.py \
    --checkpoint results/training/resnet18_fasterrcnn_*/checkpoints/best_checkpoint.pth \
    --dataset-path ../../curated_dataset_splitted_20251101_provisional_1st_version \
    --score-threshold 0.5 \
    --iou-threshold 0.5
```

### 3. Iteraciones de Mejora

**Opción A: Optimizar DEIMv2-M**
- Aumentar épocas a 100
- Ajustar learning rate (probar 0.0001)
- Aumentar batch size si es posible
- Probar DEIMv2-S (menos parámetros, más estable)

**Opción B: Cambiar a DEIMv2-S**
- Descargar `vitt_distill.pt` (9.7M params)
- Requiere menos VRAM (~10GB)
- Más rápido de entrenar
- Potencialmente mejor con dataset pequeño

**Opción C: Data Augmentation**
- Revisar si augmentations son demasiado agresivas
- Probar configuración más conservadora

---

## 📝 Tareas Pendientes para el TFG

### Corto Plazo (Esta Semana)

- [ ] **Evaluar baselines con protocolo unificado**
- [ ] **Analizar visualizaciones de predicciones**
- [ ] **Revisar curvas de training en TensorBoard**
- [ ] **Decidir**: ¿Optimizar DEIMv2-M o cambiar a DEIMv2-S?

### Medio Plazo (Próximas 2 Semanas)

- [ ] **Iterar hiperparámetros** para mejorar mAP
- [ ] **Entrenamiento largo** (100+ épocas) si es necesario
- [ ] **Comparativa exhaustiva** CNN vs ViT (tablas, gráficas)
- [ ] **Análisis de attention maps** (visualizar qué detecta el ViT)

### Largo Plazo (FASE 2 - Opcional)

- [ ] **Extensión multimodal** (fusión visión-texto)
- [ ] **Descripciones semánticas** por clase de defecto
- [ ] **Fine-tuning con embeddings de texto**

---

## 🚨 Decisión Crítica Inmediata

### ¿Continuar con DEIMv2-M o cambiar estrategia?

**Opción 1: Optimizar DEIMv2-M actual**
- Pros: Ya entrenado, infraestructura lista
- Contras: Puede necesitar muchas iteraciones

**Opción 2: Cambiar a DEIMv2-S**
- Pros: Menos parámetros, más adecuado para dataset pequeño
- Contras: Requiere re-entrenar desde cero

**Opción 3: Aumentar datos**
- Pros: ViTs mejoran con más datos
- Contras: Requiere más esfuerzo de recolección/etiquetado

**Recomendación:** 
1. Primero evaluar baselines con mismo protocolo (confirmar que mAP~0.42-0.45)
2. Analizar visualizaciones de DEIMv2 (entender qué está fallando)
3. Decidir basado en análisis: optimizar M, cambiar a S, o aumentar datos

---

## 📊 Métricas de Progreso

### Completado (✅)

- ✅ Setup completo de DEIMv2
- ✅ Entrenamiento de 52 épocas exitoso
- ✅ Pipeline de evaluación funcional
- ✅ Sistema de métricas compatible con baselines
- ✅ Infraestructura de visualización

### En Progreso (🔄)

- 🔄 Análisis de resultados
- 🔄 Comparación con baselines
- 🔄 Optimización de hiperparámetros

### Pendiente (⏳)

- ⏳ Mejora de mAP a niveles competitivos
- ⏳ Extensión multimodal (FASE 2)
- ⏳ Redacción de capítulos del TFG

---

## 💡 Conclusiones Provisionales

### Logros

1. **Infraestructura robusta:** Pipeline completo de train/eval/viz
2. **Primer modelo ViT funcional:** DEIMv2 entrenado y evaluado
3. **Base sólida para experimentación:** Fácil iterar configuraciones

### Desafíos

1. **Rendimiento inferior a baselines:** mAP 0.178 vs ~0.42-0.45
2. **Dataset pequeño:** Limitación fundamental para ViTs
3. **Optimización pendiente:** Muchos hiperparámetros por explorar

### Valor para el TFG

**Aportación técnica clara:**
- Adaptación de DEIMv2 (estado del arte) a dominio industrial
- Comparación rigurosa CNN vs ViT en dataset real
- Análisis de limitaciones de ViTs con pocos datos
- Base para extensión multimodal (FASE 2)

**Incluso si mAP no supera CNNs**, el trabajo tiene valor:
- Análisis comparativo CNN vs ViT
- Estudio de transfer learning con DINOv3
- Exploración de arquitecturas modernas en industria
- Propuesta de mejora con multimodalidad

---

## 🔗 Archivos Clave Generados

```
# Resultados de evaluación
scripts/deimv2_multimodal/outputs/deimv2_industrial_run/test_evaluation_results.json

# Detecciones completas
scripts/deimv2_multimodal/outputs/deimv2_industrial_run/test_detections.json

# Logs de TensorBoard
scripts/deimv2_multimodal/outputs/deimv2_industrial_run/summary/

# Visualizaciones (tras completar pipeline)
scripts/deimv2_multimodal/outputs/deimv2_industrial_run/visualizations_test/
```

---

## 📞 Siguiente Sesión

**Agenda propuesta:**

1. **Revisión de visualizaciones** (¿qué está detectando mal?)
2. **Comparativa con baselines** (eval con mismo protocolo)
3. **Decisión estratégica** (optimizar M, cambiar a S, o aumentar datos)
4. **Plan de iteraciones** (roadmap para mejorar mAP)

**Preparación necesaria:**
- Revisar TensorBoard logs
- Analizar imágenes de visualización
- Evaluar baselines con scripts existentes
- Pensar en estrategia de mejora

---

**Estado del proyecto: EN PUNTO CRÍTICO DE DECISIÓN**  
**Próxima acción: ANÁLISIS DE RESULTADOS Y COMPARACIÓN CON BASELINES**