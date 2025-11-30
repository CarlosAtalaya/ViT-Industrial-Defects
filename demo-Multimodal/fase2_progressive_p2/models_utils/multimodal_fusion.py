import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionModule(nn.Module):
    def __init__(self, visual_dim=256, text_dim=512, num_classes=6, hidden_dim=256):
        super().__init__()
        
        # Proyección Visual: [256] -> [512]
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        
        # --- CORRECCIÓN CRÍTICA 1: Inicialización de pesos casi nula ---
        # Inicializamos los pesos de la proyección muy cerca de cero para que
        # al principio, la proyección visual sea muy suave.
        nn.init.normal_(self.visual_proj.weight, std=0.01)
        nn.init.constant_(self.visual_proj.bias, 0)
        
        # Temperatura aprendible
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # --- CORRECCIÓN CRÍTICA 2: Alpha inicializado en 0.0 ---
        # Esto es vital. Al inicio, la influencia del texto será EXACTAMENTE 0.
        # El modelo empezará con el rendimiento del baseline (0.785) y solo
        # aumentará alpha si el gradiente indica que el texto ayuda a reducir el loss.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, visual_features, text_embeddings):
        # 1. Alinear espacio visual al textual
        v_proj = self.visual_proj(visual_features) # [B, N, 512]
        
        # 2. Normalizar (L2) - Importante para estabilidad
        v_norm = F.normalize(v_proj, p=2, dim=-1)
        t_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
        # 3. Similitud Coseno
        # [B, N, 512] @ [512, Classes] -> [B, N, Classes]
        similarity_logits = torch.matmul(v_norm, t_norm.t()) / self.temperature
        
        # Devolvemos la similitud escalada por alpha (que empieza siendo 0)
        return similarity_logits * self.alpha