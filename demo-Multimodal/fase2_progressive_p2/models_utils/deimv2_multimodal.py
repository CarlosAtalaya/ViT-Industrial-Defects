import torch
import torch.nn as nn

class DEIMv2Multimodal(nn.Module):
    def __init__(self, deimv2_model, fusion_module, text_embeddings):
        super().__init__()
        self.deimv2 = deimv2_model
        self.fusion = fusion_module
        self.register_buffer('text_embeddings', text_embeddings)
        
        self._features = []
        self._hook_handle = None
        self._register_extraction_hook()

    def _register_extraction_hook(self):
        # (Tu código de hook existente, igual que antes)
        target_found = False
        for name, module in self.deimv2.named_modules():
            if ('score_head' in name or 'class_embed' in name) and isinstance(module, nn.Linear):
                if 'enc' not in name: 
                    self._hook_handle = module.register_forward_pre_hook(self._hook_fn)
                    print(f"⚓ Hook instalado en: {name}")
                    target_found = True
                    break
        if not target_found:
            raise RuntimeError("❌ Error Crítico: No se encontró score_head en DEIMv2.")

    def _hook_fn(self, module, args):
        if isinstance(args[0], torch.Tensor):
            self._features.append(args[0])

    def forward(self, images, targets=None):
        self._features = [] 
        
        # 1. Ejecutar DEIMv2 (Obtenemos los logits originales BUENOS)
        outputs = self.deimv2(images, targets)
        
        # 2. Fusión Multimodal (Corrección Residual)
        if self._features and 'pred_logits' in outputs:
            visual_feats = self._features[-1] 
            
            if visual_feats.dim() == 3:
                # Calculamos SOLO la corrección basada en texto
                text_correction = self.fusion(visual_feats, self.text_embeddings)
                
                # --- CAMBIO CLAVE ---
                # Sumamos la corrección a los logits originales
                # Original (0.78) + Corrección (~0 al inicio) = Resultado Robusto
                outputs['pred_logits'] = outputs['pred_logits'] + text_correction
            
        # 3. Blindaje para Eval (Igual que antes)
        if 'aux_outputs' not in outputs and 'pred_logits' in outputs:
            fake_aux = {
                'pred_logits': outputs['pred_logits'],
                'pred_boxes': outputs.get('pred_boxes', None)
            }
            outputs['aux_outputs'] = [fake_aux for _ in range(5)]
            if 'enc_aux_outputs' not in outputs: outputs['enc_aux_outputs'] = [fake_aux]
            if 'enc_meta' not in outputs: outputs['enc_meta'] = {'class_agnostic': False}

        return outputs