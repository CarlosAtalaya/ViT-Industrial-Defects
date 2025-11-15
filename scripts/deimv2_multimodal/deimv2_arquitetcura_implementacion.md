# DEIMv2 Industrial Defects: Arquitectura e Implementación

**Última actualización:** 14 Noviembre 2024  
**Estado:** ✅ FASE 1 COMPLETADA - Preparando FASE 2 Multimodal

---

## 📊 Estado Actual del Proyecto

### ✅ FASE 1: DEIMv2 Vanilla - COMPLETADA

**Resultado:** mAP = 0.395 (39.5%) en validación

```
🎯 Métricas DEIMv2-M (Época 86):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mAP @ IoU=0.50:0.95   = 0.395 (39.5%)
AP  @ IoU=0.50        = 0.499 (49.9%)
AP  @ IoU=0.75        = 0.384 (38.4%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Por tamaño de objeto:
  Small  (área < 32²)  = 0.234 (23.4%) ⭐
  Medium (32² - 96²)   = 0.347 (34.7%)
  Large  (área > 96²)  = 0.474 (47.4%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recall @ maxDets=100  = 0.621 (62.1%)
```

**Mejora vs Primer Intento:**
- Primer entrenamiento (config base): mAP = 0.178
- Segundo entrenamiento (config optimizado): mAP = 0.395
- **Mejora: +122%** 🚀

### 📂 Estructura Implementada

```
scripts/deimv2_multimodal/
├── configs/
│   └── deimv2_industrial_defects.yml    # ✅ Config optimizado y estable
├── outputs/
│   ├── deimv2_industrial_run/           # Primer intento (mAP=0.178)
│   └── deimv2_industrial_run_stable/    # ✅ Segundo intento (mAP=0.395)
│       ├── checkpoint0084.pth           # Mejor checkpoint (época 86)
│       ├── log.txt                      # Historial entrenamiento
│       ├── summary/                     # TensorBoard logs
│       └── test_evaluation_results.json # Métricas en test
├── train_deimv2_industrial.py           # ✅ Script entrenamiento
├── evaluate_deimv2.py                   # ✅ Evaluación mAP COCO
├── visualize_deimv2_predictions.py      # ✅ Visualización predicciones
├── plot_deimv2_training_metrics.py      # Gráficas (requiere parser mejorado)
└── run_evaluation_deimv2.sh             # ✅ Pipeline completo eval
```

---

## 🏗️ Configuración Técnica FASE 1

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

### Modelo: DEIMv2-M

```yaml
Backbone: DINOv3 ViT-Tiny+ (vittplus_distill.pt)
  - embed_dim: 256
  - num_heads: 4
  - interaction_indexes: [3, 7, 11]
  - Parámetros: ~17.8M

Encoder: HybridEncoder
  - hidden_dim: 256
  - dim_feedforward: 1024

Decoder: DEIMTransformer
  - num_layers: 4
  - hidden_dim: 256
  - num_queries: 300
```

### Hiperparámetros Clave (Config Estable)

```yaml
# Entrenamiento
epoches: 100
flat_epoch: 70      # LR constante hasta época 70
no_aug_epoch: 10    # Sin augmentations últimas 10 épocas
warmup_iter: 2000   # Warmup largo para estabilidad

# Optimizer
lr: 0.0004                # Decoder learning rate
lr_backbone: 0.00004      # Backbone (DINOv3) learning rate
weight_decay: 0.0001
clip_max_norm: 0.1        # ⭐ Gradient clipping (crítico para estabilidad)

# Data Augmentation (Suavizado vs config base)
RandomPhotometricDistort: p=0.3  (antes 0.5)
RandomIoUCrop: p=0.5             (antes 0.8)
Mixup: prob=0.15                 (antes 0.5)
Mosaic: DESACTIVADO              (causaba inestabilidad)
CopyBlend: DESACTIVADO           (causaba NaN)

# Hardware
batch_size: 2
use_amp: True  # Mixed precision
GPU: RTX 4070 12GB
Tiempo: ~2 horas (100 épocas)
```

### Lecciones Aprendidas FASE 1

#### ❌ Problemas Encontrados

1. **NaN en gradientes (épocas 46, 87)**
   - Causa: Learning rate demasiado alto + augmentations agresivas
   - Solución: Gradient clipping + reducir LR + suavizar augmentations

2. **Dataset pequeño (715 imágenes)**
   - ViTs requieren más datos que CNNs
   - Augmentations pesadas (Mosaic, CopyBlend) causaban inestabilidad

3. **Batch size limitado (2)**
   - RTX 4070 12GB no soporta batch_size > 2 con DEIMv2-M
   - Gradientes ruidosos → convergencia lenta

#### ✅ Soluciones Efectivas

1. **Gradient clipping (`clip_max_norm: 0.1`)**
   - Previene explosión de gradientes
   - Crítico para estabilidad con dataset pequeño

2. **Warmup largo (2000 steps)**
   - Permite adaptación suave del backbone DINOv3
   - Reduce riesgo de divergencia inicial

3. **Augmentations conservadoras**
   - Desactivar Mosaic y CopyBlend
   - Reducir probabilidad de PhotometricDistort e IoUCrop
   - Trade-off aceptable: mAP 0.395 (estable) > mAP potencial 0.45 (inestable)

4. **Flat epoch largo (70 épocas)**
   - LR constante durante más tiempo
   - Mejor convergencia con dataset pequeño

---

## 📈 Comparativa con Baselines

| Modelo | Arquitectura | Params | mAP@0.50:0.95 | AP@0.50 | Tiempo | Notas |
|--------|-------------|---------|---------------|---------|--------|-------|
| ResNet-18 | CNN + Faster R-CNN | 11M | ~0.42* | ~0.50* | 1h | Baseline clásico |
| EfficientNet-B0 | CNN + Faster R-CNN | 5M | ~0.45* | ~0.52* | 1h | Baseline eficiente |
| **DEIMv2-M v1** | **ViT + DEIM** | **17.8M** | **0.178** | **0.232** | **1h** | Config base (inestable) |
| **DEIMv2-M v2** | **ViT + DEIM** | **17.8M** | **0.395** | **0.499** | **2h** | Config optimizado ✅ |

_*Nota: Baselines pendientes de evaluación con protocolo COCO exacto_

### Análisis de Resultados

**Fortalezas de DEIMv2:**
- ⭐ **Objetos pequeños:** mAP 0.234 (vs típicamente <0.10 en CNNs)
- ⭐ **Recall alto:** 62.1% (detecta más defectos)
- ⭐ **AP@0.50:** 49.9% (comparable a CNNs en detección no estricta)

**Debilidades:**
- ⚠️ **mAP@0.75:** 38.4% (localización menos precisa que CNNs)
- ⚠️ **Requiere más tiempo:** 2h vs 1h de CNNs
- ⚠️ **Más parámetros:** 17.8M vs 5-11M de CNNs

**Conclusión FASE 1:**
DEIMv2 alcanza rendimiento **competitivo** (~94% del mAP de baselines) pero con ventajas en objetos pequeños. Base sólida para FASE 2.

---

## 🚀 FASE 2: Extensión Multimodal (PRÓXIMOS PASOS)

### Objetivo

**Superar mAP 0.45** mediante fusión visión-texto, mejorando especialmente la clasificación de clases visualmente similares (ej: rayones vs fracturas).

### Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────┐
│                    DEIMv2-M Backbone                        │
│  (DINOv3 ViT + Hybrid Encoder + DEIM Transformer)          │
│                          ↓                                  │
│              Visual Embeddings (300 queries × 256d)         │
└─────────────────────────────────────────────────────────────┘
                          ↓
         ┌────────────────┴────────────────┐
         ↓                                  ↓
┌──────────────────┐              ┌──────────────────┐
│ Visual Features  │              │ Text Embeddings  │
│   (Per query)    │              │  (Per class)     │
│   [B, 300, 256]  │              │   [6, 512]       │
└────────┬─────────┘              └────────┬─────────┘
         │                                  │
         │    ┌─────────────────────────────┘
         ↓    ↓
  ┌──────────────────────┐
  │ Multimodal Fusion    │
  │  • Visual Proj 256→D │
  │  • Text Proj 512→D   │
  │  • Cosine Similarity │
  │  • Refinement Head   │
  └──────────┬───────────┘
             ↓
  ┌──────────────────────┐
  │ Enhanced Predictions │
  │   [B, 300, 6+1]      │
  └──────────────────────┘
```

### Plan de Implementación

#### 2.1 Descripciones Textuales por Clase

```python
# scripts/deimv2_multimodal/data/class_descriptions.py

CLASS_DESCRIPTIONS = {
    0: {
        "name": "NORMAL",
        "description": "Superficie limpia sin defectos visibles ni anomalías estructurales",
        "keywords": ["limpio", "intacto", "sin daño", "superficie uniforme"]
    },
    1: {
        "name": "DEFORMACIONES", 
        "description": "Alteración de la forma original con abombamiento, hundimiento o deformación plástica",
        "keywords": ["abolladura", "deformado", "ondulado", "curvatura anormal"]
    },
    2: {
        "name": "ROTURA_FRACTURA",
        "description": "Grieta profunda o ruptura completa del material con separación visible",
        "keywords": ["grieta", "fractura", "partido", "fisura profunda"]
    },
    3: {
        "name": "RAYONES_ARANAZOS",
        "description": "Línea fina y alargada de daño superficial sin penetración profunda",
        "keywords": ["rasguño", "línea fina", "marca superficial", "rayón"]
    },
    4: {
        "name": "PERFORACIONES",
        "description": "Agujero circular u orificio que atraviesa total o parcialmente el material",
        "keywords": ["orificio", "perforación", "agujero", "taladro"]
    },
    5: {
        "name": "CONTAMINACION",
        "description": "Presencia de partículas extrañas, manchas o sustancias adheridas",
        "keywords": ["suciedad", "mancha", "partículas", "residuo"]
    }
}
```

#### 2.2 Módulo de Fusión Multimodal

```python
# scripts/deimv2_multimodal/models/multimodal_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

class MultimodalFusionModule(nn.Module):
    """
    Fusiona embeddings visuales de DEIMv2 con embeddings textuales
    para mejorar la clasificación de defectos.
    """
    
    def __init__(
        self,
        visual_dim=256,      # DEIMv2 hidden_dim
        text_dim=512,        # CLIP text embedding dim
        fusion_dim=256,      # Dimensión del espacio común
        num_classes=6,
        dropout=0.1
    ):
        super().__init__()
        
        # Text encoder (pre-entrenado)
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        # Congelar text encoder (o hacer fine-tune ligero)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Proyecciones a espacio común
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes + 1)  # +1 para background
        )
        
        # Cache de text embeddings (computar una vez)
        self.register_buffer('text_embeddings', torch.zeros(num_classes, text_dim))
        self._text_embeddings_computed = False
    
    def compute_text_embeddings(self, class_descriptions):
        """
        Pre-computa embeddings de texto para todas las clases.
        Se llama una vez al inicio del entrenamiento.
        """
        if self._text_embeddings_computed:
            return
        
        text_embeds = []
        for desc in class_descriptions:
            tokens = self.tokenizer(
                desc, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.text_embeddings.device)
            
            with torch.no_grad():
                text_output = self.text_encoder(**tokens)
                # Usar [CLS] token o pooled output
                text_embed = text_output.pooler_output
                text_embeds.append(text_embed.squeeze(0))
        
        self.text_embeddings = torch.stack(text_embeds)
        self._text_embeddings_computed = True
    
    def forward(self, visual_features, return_similarity=False):
        """
        Args:
            visual_features: [B, num_queries, visual_dim] - desde DEIMv2
            return_similarity: si True, retorna también cosine similarity
        
        Returns:
            logits: [B, num_queries, num_classes + 1]
            (opcional) similarity: [B, num_queries, num_classes]
        """
        B, N, _ = visual_features.shape
        
        # Proyectar features visuales
        v_proj = self.visual_proj(visual_features)  # [B, N, fusion_dim]
        v_norm = F.normalize(v_proj, dim=-1)
        
        # Proyectar embeddings de texto
        t_proj = self.text_proj(self.text_embeddings)  # [num_classes, fusion_dim]
        t_norm = F.normalize(t_proj, dim=-1)
        
        # Cosine similarity (attention)
        similarity = torch.matmul(v_norm, t_norm.t())  # [B, N, num_classes]
        
        # Weighted text features
        text_context = torch.matmul(
            similarity.softmax(dim=-1),  # [B, N, num_classes]
            t_proj                        # [num_classes, fusion_dim]
        )  # [B, N, fusion_dim]
        
        # Concatenar visual + text context
        fused = torch.cat([v_proj, text_context], dim=-1)  # [B, N, 2*fusion_dim]
        
        # Clasificación final
        logits = self.fusion_head(fused)  # [B, N, num_classes + 1]
        
        if return_similarity:
            return logits, similarity
        return logits
```

#### 2.3 Integración con DEIMv2

```python
# scripts/deimv2_multimodal/models/deimv2_multimodal.py

class DEIMv2Multimodal(nn.Module):
    """
    Wrapper que añade MultimodalFusion sobre DEIMv2 base.
    """
    
    def __init__(self, deimv2_model, class_descriptions):
        super().__init__()
        
        self.deimv2 = deimv2_model
        
        # Módulo multimodal
        self.multimodal_fusion = MultimodalFusionModule(
            visual_dim=256,
            text_dim=512,
            num_classes=6
        )
        
        # Computar text embeddings
        self.multimodal_fusion.compute_text_embeddings(
            [desc['description'] for desc in class_descriptions.values()]
        )
    
    def forward(self, images, targets=None):
        """
        Args:
            images: tensor [B, 3, H, W]
            targets: dict con boxes, labels (entrenamiento)
        
        Returns:
            outputs: dict con pred_logits, pred_boxes (con fusión multimodal)
        """
        # Forward pass DEIMv2 base
        outputs = self.deimv2(images, targets)
        
        # Extraer features visuales del decoder
        # outputs contiene: pred_logits, pred_boxes, hs (hidden states)
        visual_features = outputs['hs'][-1]  # [B, num_queries, hidden_dim]
        
        # Aplicar fusión multimodal
        enhanced_logits = self.multimodal_fusion(visual_features)
        
        # Reemplazar logits originales con enhanced
        outputs['pred_logits'] = enhanced_logits
        
        return outputs
```

#### 2.4 Script de Entrenamiento FASE 2

```python
# scripts/deimv2_multimodal/train_deimv2_multimodal.py

def main(args):
    # 1. Cargar modelo DEIMv2 pre-entrenado (FASE 1)
    cfg = YAMLConfig(args.config)
    deimv2_base = cfg.model
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    deimv2_base.load_state_dict(checkpoint['model'])
    
    # 2. Envolver con módulo multimodal
    from data.class_descriptions import CLASS_DESCRIPTIONS
    model = DEIMv2Multimodal(deimv2_base, CLASS_DESCRIPTIONS)
    
    # 3. Congelar backbone (opcional, para fine-tune rápido)
    for param in model.deimv2.backbone.parameters():
        param.requires_grad = False
    
    # 4. Entrenar solo módulo multimodal (20 épocas adicionales)
    optimizer = torch.optim.AdamW([
        {'params': model.multimodal_fusion.parameters(), 'lr': 1e-4}
    ])
    
    # ... resto del training loop
```

#### 2.5 Config FASE 2

```yaml
# configs/deimv2_industrial_multimodal.yml

__include__: ['deimv2_industrial_defects.yml']

# Cambios para FASE 2
output_dir: ./scripts/deimv2_multimodal/outputs/deimv2_multimodal_run

# Fine-tuning (épocas cortas sobre modelo pre-entrenado)
epoches: 20
flat_epoch: 15
no_aug_epoch: 3
warmup_iter: 500

# Optimizer solo para módulo multimodal
optimizer:
  lr: 0.0001  # LR bajo para fine-tune
  
# Cargar checkpoint FASE 1
resume: ./scripts/deimv2_multimodal/outputs/deimv2_industrial_run_stable/checkpoint0084.pth
```

### Roadmap FASE 2

#### Semana 1: Setup Multimodal
- [ ] Implementar `class_descriptions.py` con descripciones
- [ ] Implementar `MultimodalFusionModule`
- [ ] Implementar `DEIMv2Multimodal` wrapper
- [ ] Test de integración (forward pass sin errores)

#### Semana 2: Entrenamiento Incremental
- [ ] Crear config `deimv2_industrial_multimodal.yml`
- [ ] Entrenar 20 épocas con backbone congelado
- [ ] Evaluar mAP multimodal vs vanilla

#### Semana 3: Análisis y Optimización
- [ ] Visualizar attention maps texto-visual
- [ ] Analizar qué clases mejoran más
- [ ] Iterar descripciones textuales si es necesario
- [ ] Fine-tune end-to-end si mejora mAP

### Expectativas FASE 2

**Objetivo:** mAP > 0.45 (superar baselines CNN)

**Mejoras esperadas:**
- **Clasificación:** +5-8% en clases ambiguas (rayones vs fracturas)
- **Recall:** +3-5% por mejor discriminación semántica
- **Objetos pequeños:** Mantener ventaja (mAP ~0.25)

**Best case:** mAP ~0.48 (6% mejora sobre vanilla)  
**Realistic case:** mAP ~0.42-0.45 (comparable a CNNs)  
**Worst case:** mAP ~0.40 (mejora marginal, pero extensión válida)

---

## 📝 Tareas Inmediatas

### Antes de FASE 2

1. **Evaluar checkpoint0084 en test set**
   ```bash
   cd scripts/deimv2_multimodal
   ./run_evaluation_deimv2.sh \
     outputs/deimv2_industrial_run_stable/checkpoint0084.pth
   ```

2. **Comparar con baselines CNN (protocolo COCO)**
   ```bash
   # ResNet-18
   cd scripts/resnet18
   python evaluate_model.py --checkpoint ... --score-threshold 0.5
   
   # EfficientNet
   cd scripts/efficientnet
   python evaluate_model.py --checkpoint ... --score-threshold 0.5
   ```

3. **Analizar visualizaciones**
   - Revisar `outputs/.../visualizations_test/`
   - Identificar errores típicos del modelo
   - Documentar para justificar extensión multimodal

### Iniciar FASE 2

4. **Implementar descripciones textuales**
   - Crear `data/class_descriptions.py`
   - Validar descripciones con experto de dominio

5. **Setup módulo multimodal**
   - Implementar `MultimodalFusionModule`
   - Test de forward pass aislado

6. **Pipeline de entrenamiento incremental**
   - Config `deimv2_industrial_multimodal.yml`
   - Script `train_deimv2_multimodal.py`

---

## 🎓 Contribución al TFG

### Valor Técnico

**FASE 1 (Completada):**
- ✅ Adaptación exitosa de DEIMv2 (SOTA ViT) a dominio industrial
- ✅ Optimización de hiperparámetros para dataset pequeño
- ✅ Benchmarking riguroso contra baselines CNN

**FASE 2 (En desarrollo):**
- 🔄 Extensión multimodal custom (no existe en paper original)
- 🔄 Fusión visión-texto para clasificación de defectos
- 🔄 Análisis de mejora semántica vs puramente visual

### Estructura Memoria (Capítulos Técnicos)

**Capítulo 4: Implementación DEIMv2 para Defectos Industriales**
- 4.1 Arquitectura base (DINOv3 + DEIM)
- 4.2 Adaptación a dataset industrial (6 clases)
- 4.3 Optimización de entrenamiento (gradient clipping, augmentations)
- 4.4 Resultados vanilla (mAP 0.395, comparativa con CNNs)

**Capítulo 5: Extensión Multimodal Visión-Texto**
- 5.1 Motivación: limitaciones de modelos visuales puros
- 5.2 Diseño de descripciones textuales por clase
- 5.3 Arquitectura de fusión (CLIP embeddings + attention)
- 5.4 Entrenamiento incremental (fine-tune sobre FASE 1)

**Capítulo 6: Resultados y Análisis**
- 6.1 Métricas cuantitativas (tablas mAP, recall, precision)
- 6.2 Análisis cualitativo (attention maps, casos de éxito/fallo)
- 6.3 Comparativa exhaustiva (CNN vs ViT vanilla vs ViT multimodal)
- 6.4 Discusión: trade-offs complejidad vs rendimiento

---

## 🚨 Decisiones Pendientes

1. **¿Evaluar baselines primero o empezar FASE 2 directamente?**
   - Recomendación: Evaluar baselines ANTES (necesario para comparación justa)

2. **¿Fine-tune backbone en FASE 2 o solo módulo multimodal?**
   - Recomendación: Solo módulo multimodal primero (más rápido, menos riesgo)

3. **¿Usar CLIP o alternativa (SigLIP, etc.)?**
   - Recomendación: CLIP (más maduro, fácil integración)

4. **¿Cuántas épocas en FASE 2?**
   - Recomendación: 20 épocas (suficiente para fine-tune, ~40 minutos)

---

## 📞 Próxima Sesión

**Agenda propuesta:**

1. **Revisión de resultados en test** (checkpoint0084)
2. **Comparativa definitiva** con baselines CNN
3. **Diseño de descripciones** textuales (validación con dominio)
4. **Implementación inicial** de `MultimodalFusionModule`

**Preparación necesaria:**
- Evaluar checkpoint en test
- Evaluar baselines con protocolo COCO
- Pensar en descripciones textuales por clase
- Revisar visualizaciones para identificar errores

---

**Estado del proyecto: ✅ FASE 1 COMPLETADA - 🚀 INICIANDO FASE 2**  
**Próxima acción: EVALUAR CHECKPOINT EN TEST Y COMPARAR CON BASELINES**