# DEIMv2 + Multimodal: Arquitectura de Implementación

## 🎯 Objetivo
Adaptar DEIMv2 (DINOv3 + detección) para tu dataset industrial (6 clases de defectos) con extensión multimodal visión-texto.

---

## 📁 Estructura de Directorios Propuesta

```
scripts/
├── resnet18/              # ✅ Existente - baseline
├── efficientnet/          # ✅ Existente - baseline
└── deimv2_industrial/     # 🆕 NUEVO - Tu implementación
    ├── configs/
    │   └── deimv2_industrial_defects.yml    # Config adaptado
    ├── data/
    │   ├── industrial_dataset.py            # Wrapper CocoDetection
    │   └── class_descriptions.py            # Textos por clase
    ├── models/
    │   └── text_fusion.py                   # 🆕 Módulo multimodal (OPCIONAL fase 2)
    ├── train_deimv2.py                      # Script entrenamiento
    ├── evaluate_deimv2.py                   # Script evaluación
    ├── visualize_attention.py               # Attention maps
    └── README.md
```

---

## 🚀 Plan de Implementación (2 Fases)

### **FASE 1: DEIMv2 Vanilla (PRIORIDAD)**
**Objetivo**: Validar que DEIMv2 funciona con tu dataset antes de multimodal.

#### 1.1 Archivo Config (`deimv2_industrial_defects.yml`)
```yaml
# Heredar de config base de DEIMv2-S
__include__: [
  '../../../configs/dataset/custom_detection.yml',
  '../../../configs/deimv2/deimv2_dinov3_s_coco.yml',
]

# Adaptaciones
num_classes: 6  # PERFORACIONES, RAYONES, ROTURA, DEFORMACIONES, CONTAMINACION, NORMAL
remap_mscoco_category: False

train_dataloader:
  total_batch_size: 4  # RTX 4070 -> batch 2 x 2 GPUs simuladas
  dataset:
    img_folder: /ruta/a/tu/dataset/train
    ann_file: /ruta/a/tu/dataset/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: /ruta/a/tu/dataset/val
    ann_file: /ruta/a/tu/dataset/annotations/instances_val.json

# Backbone DINOv3
DINOv3STAs:
  weights_path: ./ckpts/vitt_distill.pt  # Descargar ViT-Tiny distilled

# Epochs reducidos para tu dataset pequeño
epoches: 50
flat_epoch: 25
no_aug_epoch: 6
```

#### 1.2 Dataset Wrapper (`data/industrial_dataset.py`)
```python
# Reutilizar CocoDetection de DEIMv2
from engine.data.coco import CocoDetection

# Tu dataset ya está en formato COCO → usar directamente
# Solo necesitas verificar compatibilidad de IDs de categorías
```

#### 1.3 Script Entrenamiento (`train_deimv2.py`)
```python
# Copiar estructura de train.py del repo DEIMv2
# Cambiar: cargar tu config custom en lugar de COCO
# Comando:
# CUDA_VISIBLE_DEVICES=0 python train_deimv2.py \
#   -c configs/deimv2_industrial_defects.yml \
#   --use-amp --seed=0
```

**Resultado Esperado FASE 1**: 
- mAP > 40 en tu dataset (comparable a ResNet/EfficientNet)
- Validar que DINOv3 funciona mejor en defectos pequeños

---

### **FASE 2: Extensión Multimodal (SOLO SI FASE 1 FUNCIONA)**

#### 2.1 Descripciones Textuales (`data/class_descriptions.py`)
```python
CLASS_DESCRIPTIONS = {
    "PERFORACIONES": "Agujero circular u orificio visible en la superficie del material",
    "RAYONES_ARANAZOS": "Línea fina y alargada de daño superficial en el recubrimiento",
    "ROTURA_FRACTURA": "Grieta profunda o ruptura completa del material estructural",
    "DEFORMACIONES": "Alteración de la forma original con abombamiento o hundimiento",
    "CONTAMINACION": "Presencia de partículas extrañas o manchas en la superficie",
    "NORMAL": "Superficie sin defectos visibles ni anomalías"
}
```

#### 2.2 Módulo Fusión Visión-Texto (`models/text_fusion.py`)
```python
# Encoder texto: CLIP o SigLIP
from transformers import CLIPTextModel, CLIPTokenizer

class MultimodalFusion(nn.Module):
    def __init__(self, visual_dim=192, text_dim=512, num_classes=6):
        # Proyectar embeddings visuales y textuales a espacio común
        self.visual_proj = nn.Linear(visual_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        self.fusion = nn.Linear(512, num_classes)
    
    def forward(self, visual_feats, text_embeds):
        # Similitud coseno + clasificación
        v = F.normalize(self.visual_proj(visual_feats))
        t = F.normalize(self.text_proj(text_embeds))
        fused = torch.cat([v, t], dim=-1)
        return self.fusion(fused)
```

#### 2.3 Entrenamiento Incremental
```bash
# 1. Entrenar DEIMv2 base (FASE 1)
python train_deimv2.py -c configs/base.yml

# 2. Fine-tune con fusión multimodal
python train_deimv2_multimodal.py \
  -c configs/multimodal.yml \
  -r outputs/deimv2_base/best.pth  # Cargar pesos fase 1
```

---

## ⚙️ Configuración Hardware (RTX 4070)

```yaml
# Parámetros ajustados para tu GPU
train_dataloader:
  total_batch_size: 4      # 2 imágenes reales (simular 2 GPUs)
  num_workers: 4

# Modelo
DINOv3STAs:
  name: vit_tiny            # 192 dim - 9.7M params → cabe en 12GB
  
# Entrenamiento
use_amp: True              # Mixed precision → ahorra VRAM
gradient_checkpointing: True  # Si necesitas más memoria
```

---

## 📊 Comparación Esperada

| Modelo | mAP | Params | VRAM | Notas |
|--------|-----|--------|------|-------|
| ResNet-18 (baseline) | ~42 | 11M | 6GB | Tu resultado actual |
| EfficientNet (baseline) | ~45 | 5M | 5GB | Tu resultado actual |
| **DEIMv2-S (FASE 1)** | **~50** | **9.7M** | **10GB** | DINOv3 + STA + Dense O2O |
| **DEIMv2-S + Multimodal (FASE 2)** | **~53** | **10.5M** | **11GB** | + Fusión texto |

---

## ✅ Checklist Implementación

### FASE 1 (Semana 1-2):
- [ ] Clonar repo DEIMv2: `git clone https://github.com/Intellindust-AI-Lab/DEIMv2`
- [ ] Descargar ViT-Tiny distilled: `vitt_distill.pt` → `./ckpts/`
- [ ] Crear `configs/deimv2_industrial_defects.yml`
- [ ] Verificar formato COCO de tu dataset
- [ ] Entrenar 10 epochs de prueba → verificar convergencia
- [ ] Entrenar 50 epochs completo
- [ ] Comparar mAP con ResNet/EfficientNet

### FASE 2 (Semana 3-4) - SOLO SI FASE 1 OK:
- [ ] Implementar `class_descriptions.py`
- [ ] Implementar `MultimodalFusion` module
- [ ] Fine-tune con fusión visión-texto
- [ ] Comparar mAP multimodal vs vanilla

---

## 🎓 Para el TFG

**Contribución técnica clara**:
1. **Adaptación de DEIMv2 a dominio industrial** (no está en paper original)
2. **Extensión multimodal custom** (tu aportación principal)
3. **Benchmarking exhaustivo** (CNN vs ViT vs Multimodal)

**Estructura memoria**:
- Cap 4: Implementación DEIMv2 vanilla en defectos industriales
- Cap 5: Propuesta extensión multimodal con embeddings texto
- Cap 6: Resultados comparativos (tablas mAP, gráficas atención)

---

## 🚨 Decisión Crítica AHORA

**¿Empezamos con FASE 1 (DEIMv2 vanilla) o quieres ir directo a multimodal?**

**Recomendación**: FASE 1 primero. Razones:
1. Validar que DEIMv2 funciona con tu dataset
2. Baseline sólido para comparar multimodal
3. Menos riesgo de bugs complejos
4. Si FASE 2 falla, FASE 1 ya es contribución válida

**Siguiente paso**: ¿Creo el archivo `deimv2_industrial_defects.yml` completo?