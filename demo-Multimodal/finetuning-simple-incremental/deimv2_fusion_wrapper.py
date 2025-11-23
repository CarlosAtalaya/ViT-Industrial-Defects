"""
Wrapper DEIMv2 + Multimodal Fusion
===================================

Añade fusion multimodal ENCIMA de DEIMv2 sin modificar código fuente.
"""

import torch
import torch.nn as nn


class DEIMv2WithFusion(nn.Module):
    """
    Wrapper que añade fusion multimodal a DEIMv2.
    
    Forward:
        1. DEIMv2: img → (bboxes, logits, ...)
        2. Fusion: logits + text_embeddings → logits_mejorados
        3. Return: (bboxes, logits_mejorados, ...)
    """
    
    def __init__(self, deimv2_model, fusion_module, text_embeddings):
        super().__init__()
        
        self.deimv2 = deimv2_model
        self.fusion = fusion_module
        self.register_buffer('text_embeddings', text_embeddings)
    
    def forward(self, images, targets=None):
        """
        Args:
            images: [B, 3, H, W]
            targets: training targets
        
        Returns:
            Same as DEIMv2 but with fused logits
        """
        # 1. Forward DEIMv2 (frozen)
        with torch.set_grad_enabled(not self.training or any(p.requires_grad for p in self.deimv2.parameters())):
            outputs = self.deimv2(images, targets)
        
        # DEIMv2 retorna dict con 'pred_logits' y 'pred_boxes'
        # O tuple (bboxes, logits, corners, refs, pre_bboxes, pre_scores)
        
        # Detectar formato
        if isinstance(outputs, dict):
            logits = outputs['pred_logits']  # [B, N, C]
            
            # Aplicar fusion
            B, N, C = logits.shape
            # Usar logits como "visual features proxy"
            fused_logits, _ = self.fusion(logits.view(B*N, C), self.text_embeddings)
            fused_logits = fused_logits.view(B, N, C)
            
            # Reemplazar
            outputs['pred_logits'] = fused_logits
            
        else:
            # Formato tuple
            dec_out_bboxes, dec_out_logits, dec_out_pred_corners, \
            dec_out_refs, pre_bboxes, pre_scores = outputs
            
            # dec_out_logits: [num_layers, B, N, C]
            fused_layers = []
            for layer_logits in dec_out_logits:
                B, N, C = layer_logits.shape
                fused, _ = self.fusion(layer_logits.view(B*N, C), self.text_embeddings)
                fused_layers.append(fused.view(B, N, C))
            
            dec_out_logits = torch.stack(fused_layers)
            outputs = (dec_out_bboxes, dec_out_logits, dec_out_pred_corners,
                      dec_out_refs, pre_bboxes, pre_scores)
        
        return outputs
    
    def freeze_detector(self):
        """Congela DEIMv2 completo."""
        for param in self.deimv2.parameters():
            param.requires_grad = False
    
    def unfreeze_fusion(self):
        """Descongela fusion module."""
        for param in self.fusion.parameters():
            param.requires_grad = True


def build_deimv2_with_fusion(checkpoint_path, config_path, 
                              text_embeddings, device='cuda'):
    """
    Construye modelo completo: DEIMv2 + Fusion.
    
    Args:
        checkpoint_path: Path a best_stg1.pth
        config_path: Path a deimv2_industrial_defects.yml
        text_embeddings: [num_classes, 512] CLIP embeddings
        device: cuda/cpu
    
    Returns:
        model: DEIMv2WithFusion
    """
    import sys
    from pathlib import Path
    
    # Importar DEIMv2
    deimv2_path = Path(__file__).parent.parent.parent / "DEIMv2"
    sys.path.insert(0, str(deimv2_path))
    from engine.core import YAMLConfig
    
    # Importar fusion
    sys.path.insert(0, str(Path(__file__).parent))
    from models_utils import MultimodalFusionModule
    
    # 1. Cargar DEIMv2
    print(f"📦 Cargando DEIMv2...")
    cfg = YAMLConfig(config_path)
    deimv2_model = cfg.model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    deimv2_model.load_state_dict(checkpoint['model'])
    deimv2_model.eval()
    
    print(f"✅ DEIMv2 cargado")
    
    # 2. Crear fusion
    print(f"🔧 Creando fusion module...")
    
    # Acceder a config correctamente
    deim_cfg = cfg.yaml_cfg.get('DEIMTransformer', {})
    hidden_dim = deim_cfg.get('hidden_dim', 256)
    num_classes = cfg.yaml_cfg.get('num_classes', 6)
    
    fusion = MultimodalFusionModule(
        visual_dim=hidden_dim,
        text_dim=512,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.1
    ).to(device)
    
    print(f"✅ Fusion creado: {sum(p.numel() for p in fusion.parameters()):,} params")
    
    # 3. Wrapper
    model = DEIMv2WithFusion(deimv2_model, fusion, text_embeddings)
    model.freeze_detector()
    model.unfreeze_fusion()
    
    print(f"✅ Modelo completo construido")
    print(f"   Detector: FROZEN")
    print(f"   Fusion: TRAINABLE")
    
    return model