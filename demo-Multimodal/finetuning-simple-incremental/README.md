# Opción 1: Fine-tuning Incremental Simple

## 📋 Resumen

**Estrategia:** Congelar detector DEIMv2 completo y entrenar solo el módulo de fusión multimodal.

- **Baseline:** mAP 0.785 (300 epochs, best_stg1.pth)
- **Target:** mAP 0.80-0.82 (+2-4%)
- **Tiempo estimado:** 3-4 horas (30-40 epochs)
- **Riesgo:** Mínimo (modelo base protegido)

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    Imagen Input                         │
│                      [B, 3, 1024, 1024]                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DINOv3 Backbone (ViT-L/14)                 │
│                    ❄️ CONGELADO                         │
│              Extrae features visuales                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Detector DEIMv2                            │
│                    ❄️ CONGELADO                         │
│     Propone regiones + features [B, N, 256]            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│ Visual Proj    │      │ Text Embeddings│
│ [B, N, 256]    │      │ [6, 512]       │
│                │      │ (CLIP)         │
└───────┬────────┘      └───────┬────────┘
        │                       │
        ▼                       ▼
┌─────────────────────────────────────────┐
│    🔥 Multimodal Fusion Module          │
│         (ÚNICO ENTRENABLE)              │
│                                         │
│  1. Proyección a espacio común (256D)  │
│  2. Atención cruzada visual→texto      │
│  3. Clasificación refinada             │
└────────────────┬────────────────────────┘
                 │
                 ▼
         [B, N, num_classes]
           Logits refinados
```

---

## 📂 Estructura de Archivos

```
demo-Multimodal/opcion1/
├── data/
│   ├── __init__.py
│   └── class_descriptions.py         ✅ Descripciones textuales (6 clases)
│
├── models/
│   ├── __init__.py
│   ├── text_encoder.py               ✅ Wrapper CLIP (512-dim)
│   ├── multimodal_fusion.py          ✅ Módulo de fusión (atención cruzada)
│   └── deimv2_multimodal.py          ✅ Wrapper completo
│
├── configs/
│   └── opcion1_config.yml            ✅ Configuración completa
│
├── train_opcion1.py                  ✅ Script entrenamiento
├── README.md                         ✅ Este archivo
│
└── outputs/                          📁 (se crea al entrenar)
    └── fase2_opcion1_simple_YYYYMMDD_HHMMSS/
        ├── checkpoints/
        │   ├── best.pth
        │   └── checkpoint_XXX.pth
        ├── logs/
        │   └── training.log
        ├── visualizations/
        └── predictions/
```

---

## ✅ Estado de Implementación

### Completado

- [x] `class_descriptions.py` - Descripciones optimizadas para 6 clases
- [x] `text_encoder.py` - Wrapper CLIP para embeddings de texto
- [x] `multimodal_fusion.py` - Módulo de fusión con atención cruzada
- [x] `deimv2_multimodal.py` - Wrapper que integra DEIMv2 + fusión
- [x] `opcion1_config.yml` - Configuración completa
- [x] `train_opcion1.py` - Estructura del script de entrenamiento

### Pendiente de Integración

- [ ] **Carga de DEIMv2:** Integrar `load_deimv2_checkpoint()` con código real de DEIMv2
- [ ] **Dataloaders:** Conectar con datasets de FASE 1
- [ ] **Training loop:** Integrar con `engine.py` de DEIMv2
- [ ] **Evaluación:** Adaptar `evaluate()` para fusión multimodal
- [ ] **Extracción de features:** Modificar forward de DEIMv2 para exponer features intermedias

---

## 🚀 Siguientes Pasos

### 1. Verificar Dependencias (5 min)

```bash
# Instalar transformers para CLIP
pip install transformers --break-system-packages

# Verificar instalación
python -c "from transformers import CLIPTokenizer, CLIPTextModel; print('✅ Transformers OK')"
```

### 2. Test de Componentes (15 min)

```bash
# Test descripciones
python demo-Multimodal/opcion1/data/class_descriptions.py

# Test text encoder
python demo-Multimodal/opcion1/models/text_encoder.py

# Test fusion module
python demo-Multimodal/opcion1/models/multimodal_fusion.py
```

### 3. Integración con DEIMv2 (CRÍTICO)

**Archivos a modificar en DEIMv2:**

#### a) `DEIMv2/models/deim.py`

Modificar forward del modelo para exponer features intermedias:

```python
def forward(self, images, targets=None):
    # ... código existente ...
    
    # Añadir salida de features
    if self.training and hasattr(self, 'return_features'):
        return {
            'loss': losses,
            'features': decoder_features  # [B, N, 256]
        }
    else:
        return outputs
```

#### b) Crear `demo-Multimodal/opcion1/utils/deimv2_loader.py`

```python
"""
Utilidad para cargar checkpoint de DEIMv2.
"""

import torch
import sys
sys.path.append('DEIMv2')

from models.deim import build_model

def load_deimv2_from_checkpoint(checkpoint_path, config, device='cuda'):
    """
    Carga modelo DEIMv2 desde checkpoint de FASE 1.
    
    Args:
        checkpoint_path: Ruta a best_stg1.pth
        config: Config dict de DEIMv2
        device: Dispositivo
    
    Returns:
        model: Modelo DEIMv2 cargado
    """
    # 1. Construir modelo
    model = build_model(config)
    model.to(device)
    
    # 2. Cargar weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # 3. Poner en eval mode
    model.eval()
    
    return model
```

#### c) Adaptar dataloaders

Reutilizar dataloaders de FASE 1:

```python
# En train_opcion1.py
from scripts.deimv2_multimodal.dataset_industrial import build_dataloader

train_loader = build_dataloader(
    config['dataset'],
    split='train',
    batch_size=config['training']['batch_size']
)
```

### 4. Entrenamiento (3-4 horas)

```bash
# Ejecutar entrenamiento
python demo-Multimodal/opcion1/train_opcion1.py \
    --config demo-Multimodal/opcion1/configs/opcion1_config.yml
```

**Monitoreo esperado:**

```
Epoch 0: mAP = 0.785 (baseline)
Epoch 5: mAP = 0.792 (+0.7%)
Epoch 10: mAP = 0.798 (+1.3%)
Epoch 15: mAP = 0.803 (+1.8%)
Epoch 20: mAP = 0.807 (+2.2%)  ← Target alcanzado
...
Epoch 30-40: Convergencia
```

### 5. Evaluación Final

```bash
# Evaluar mejor checkpoint
python evaluate_opcion1.py \
    --checkpoint outputs/fase2_opcion1_simple_*/checkpoints/best.pth \
    --split test
```

**Métricas esperadas:**

| Clase | AP Baseline | AP Target | AP Obtenido |
|-------|-------------|-----------|-------------|
| NORMAL | 0.980 | - | ? |
| PERFORACIONES | 0.924 | - | ? |
| RAYONES | 0.806 | 0.83 | ? |
| DEFORMACIONES | 0.779 | 0.80 | ? |
| CONTAMINACION | 0.645 | 0.68 | ? |
| ROTURA | 0.576 | 0.62 | ? |
| **mAP** | **0.785** | **0.80** | **?** |

---

## 🎯 Criterios de Éxito

### ✅ Éxito Completo (proceder a Opción 3)
- mAP ≥ 0.80
- ROTURA AP ≥ 0.62
- Sin degradación en otras clases

### ⚠️ Éxito Parcial (revisar antes de Opción 3)
- mAP 0.79-0.80
- Mejora en ROTURA pero caída en otras clases

### ❌ Fallo (revisar arquitectura)
- mAP < 0.79
- Degradación general del modelo

---

## 📊 Comparación con Otras Opciones

| Aspecto | Opción 1 | Opción 2 | Opción 3 |
|---------|----------|----------|----------|
| **Tiempo** | 3-4h ⚡ | 16-20h | 6-7h |
| **Riesgo** | Bajo 🟢 | Alto 🔴 | Medio 🟡 |
| **Mejora esperada** | +2-4% | ? | +4-8% |
| **Complejidad** | Simple | Media | Alta |
| **Baseline protegido** | ✅ | ❌ | ✅ |

---

## 🔧 Troubleshooting

### Error: CUDA out of memory

```bash
# Reducir batch size
# En opcion1_config.yml
training:
  batch_size: 2  # en lugar de 4
```

### Error: mAP bajando durante entrenamiento

- Verificar que detector esté realmente congelado
- Reducir learning rate a 0.00005
- Aumentar warmup epochs a 10

### Error: No mejora después de 20 epochs

- Revisar que text embeddings sean diversos (similitud coseno < 0.85)
- Probar SimpleFusionModule en lugar de MultimodalFusionModule
- Verificar que augmentations no sean muy agresivas

---

## 📝 Notas Técnicas

### Dimensiones Clave

- Visual features: 256-dim (salida decoder DEIMv2)
- Text embeddings: 512-dim (CLIP ViT-B/16)
- Hidden space: 256-dim (espacio común proyección)
- Attention heads: 4 (atención cruzada)

### Memoria GPU Estimada

- Modelo base (congelado): ~8GB
- Fusion module: ~50MB
- Activations (batch_size=4): ~2GB
- **Total:** ~10-11GB ✅ (cabe en RTX 4070 12GB)

### Parámetros Entrenables

- Detector: ~50M parámetros (❄️ congelado)
- Fusion: ~500K parámetros (🔥 entrenable)
- **Ratio:** ~1% del modelo total entrena

---

## 📚 Referencias

- CLIP paper: "Learning Transferable Visual Models From Natural Language Supervision"
- DEIMv2: "Detection with Improved Multi-scale Vision Transformers"
- DINOv3: "DINOv3: A Self-Supervised Vision Transformer Model"

---

**Última actualización:** 23 Noviembre 2024