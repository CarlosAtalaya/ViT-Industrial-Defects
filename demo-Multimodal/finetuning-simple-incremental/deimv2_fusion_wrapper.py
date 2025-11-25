"""
Wrapper DEIMv2 + Multimodal Fusion (VERSION FINAL: DEIM NAMING FIX)
===================================================================
- Intercepta features en 'dec_score_head' (específico de DEIMv2).
- Mantiene compatibilidad con 'class_embed' por seguridad.
- Genera salidas auxiliares para DEIMCriterion.
"""

import torch
import torch.nn as nn

class DEIMv2WithFusion(nn.Module):
    
    def __init__(self, deimv2_model, fusion_module, text_embeddings):
        super().__init__()
        
        self.deimv2 = deimv2_model
        self.fusion = fusion_module
        self.register_buffer('text_embeddings', text_embeddings)
        
        # Almacenamiento temporal
        self._captured_features = []
        self.hooks = []
        
        # --- ESTRATEGIA DE HOOKS MULTI-NOMBRE ---
        # DEIMv2 usa 'dec_score_head', DETR usa 'class_embed'
        target_names = ['dec_score_head', 'class_embed', 'score_head']
        hook_count = 0
        
        print("\n🔍 Buscando capas de clasificación en DEIMv2...")
        
        for name, module in self.deimv2.named_modules():
            # Verificamos si el nombre coincide con alguno de los objetivos
            # Y aseguramos que sea Linear (la capa que hace la proyección)
            is_target = any(t in name for t in target_names)
            
            if is_target and isinstance(module, nn.Linear):
                # Evitamos capas de codificación posicional o auxiliares que no sean de score
                if 'enc' not in name and 'query' not in name:
                    h = module.register_forward_pre_hook(self._hook_fn)
                    self.hooks.append(h)
                    hook_count += 1
                    print(f"   ⚓ Hook registrado en: {name}")
        
        if hook_count > 0:
            print(f"✅ Total hooks registrados: {hook_count}")
        else:
            # Diagnóstico extendido si falla
            print("\n❌ ERROR: No se encontraron capas. Módulos disponibles (primeros 20):")
            for i, (n, _) in enumerate(self.deimv2.named_modules()):
                if i < 20: print(f"   - {n}")
            raise RuntimeError("❌ ERROR CRÍTICO: Imposible encontrar capas de clasificación (dec_score_head).")

    def _hook_fn(self, module, args):
        """Captura features [Batch, Queries, 256] antes de la clasificación."""
        if isinstance(args[0], torch.Tensor):
            self._captured_features.append(args[0])
    
    def forward(self, images, targets=None):
        # 1. Limpiar capturas previas
        self._captured_features = []
        
        # 2. Forward del detector base (congelado)
        with torch.set_grad_enabled(not self.training or any(p.requires_grad for p in self.deimv2.parameters())):
            outputs = self.deimv2(images, targets)
        
        # 3. Fusión Multimodal
        fusion_applied = False
        
        if isinstance(outputs, dict) and 'pred_logits' in outputs:
            # Buscamos el tensor de features correcto
            valid_features = None
            
            if self._captured_features:
                # DEIMv2 tiene múltiples capas de decoder.
                # La última captura válida suele corresponder a la salida final.
                # Filtramos por dimensionalidad para asegurar que sea [B, N, 256]
                candidate = self._captured_features[-1]
                
                if candidate.dim() == 3 and candidate.shape[1] == outputs['pred_logits'].shape[1]:
                    valid_features = candidate
            
            if valid_features is not None:
                B, N, D = valid_features.shape
                
                # Fusión: [B*N, D] + [Classes, TextDim] -> [B*N, Classes]
                fused_logits, _ = self.fusion(valid_features.view(B*N, D), self.text_embeddings)
                fused_logits = fused_logits.view(B, N, -1)
                
                # Reemplazar logits originales
                outputs['pred_logits'] = fused_logits
                fusion_applied = True
            
            # Diagnóstico silencioso para no saturar logs, activado solo si falla el Safety Check

        # 4. SAFETY CHECK DE GRADIENTE
        if self.training and not fusion_applied:
            print(f"\n⚠️ DIAGNÓSTICO DE FALLO DE FUSIÓN:")
            print(f"   Capturas totales: {len(self._captured_features)}")
            if self._captured_features:
                print(f"   Shape última captura: {self._captured_features[-1].shape}")
            raise RuntimeError("⛔ FUSIÓN NO APLICADA: Fallo en captura de features.")

        # 5. FIX PARA DEIMCriterion (Blindaje)
        if isinstance(outputs, dict):
            # A) Simular aux_outputs
            if 'aux_outputs' not in outputs and 'pred_logits' in outputs:
                fake_aux = {
                    'pred_logits': outputs['pred_logits'],
                    'pred_boxes': outputs.get('pred_boxes', None)
                }
                outputs['aux_outputs'] = [fake_aux for _ in range(5)]
            
            # B) Simular enc_aux_outputs y meta
            if 'enc_aux_outputs' not in outputs and 'pred_logits' in outputs:
                fake_enc = {
                    'pred_logits': outputs['pred_logits'],
                    'pred_boxes': outputs.get('pred_boxes', None)
                }
                outputs['enc_aux_outputs'] = [fake_enc]
                if 'enc_meta' not in outputs:
                    outputs['enc_meta'] = {'class_agnostic': False}

        return outputs
    
    def freeze_detector(self):
        for param in self.deimv2.parameters():
            param.requires_grad = False
            
    def unfreeze_fusion(self):
        for param in self.fusion.parameters():
            param.requires_grad = True

def build_deimv2_with_fusion(checkpoint_path, config_path, text_embeddings, device='cuda'):
    import sys
    from pathlib import Path
    
    deimv2_path = Path(__file__).parent.parent.parent / "DEIMv2"
    if str(deimv2_path) not in sys.path:
        sys.path.insert(0, str(deimv2_path))
        
    from engine.core import YAMLConfig
    from models_utils import MultimodalFusionModule
    
    print(f"📦 Cargando configuración DEIMv2...")
    cfg = YAMLConfig(config_path)
    deimv2_model = cfg.model.to(device)
    
    print(f"📦 Cargando pesos desde {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    deimv2_model.load_state_dict(state_dict)
    deimv2_model.eval()
    
    deim_cfg = cfg.yaml_cfg.get('DEIMTransformer', {})
    hidden_dim = deim_cfg.get('hidden_dim', 256)
    
    print(f"🔧 Inicializando Fusión Multimodal (dim={hidden_dim})...")
    fusion = MultimodalFusionModule(
        visual_dim=hidden_dim,
        text_dim=512,
        hidden_dim=hidden_dim,
        num_classes=cfg.yaml_cfg.get('num_classes', 6),
        dropout=0.1
    ).to(device)
    
    model = DEIMv2WithFusion(deimv2_model, fusion, text_embeddings)
    model.freeze_detector()
    model.unfreeze_fusion()
    
    return model