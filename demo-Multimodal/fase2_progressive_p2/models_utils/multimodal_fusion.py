import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionModule(nn.Module):
    def __init__(self, visual_dim=256, text_dim=512, num_classes=6, hidden_dim=256):
        super().__init__()
        
        # Proyección Visual: [256] -> [512] para igualar a CLIP
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        
        # Temperatura aprendible (inicializada baja para no meter ruido al principio)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # Factor de escala aprendible (Alpha) para controlar cuánto caso hacemos al texto
        # Lo iniciamos pequeño para priorizar el visual al principio
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual_features, text_embeddings):
        # 1. Alinear espacio visual al textual
        v_proj = self.visual_proj(visual_features) # [B, N, 512]
        
        # 2. Normalizar (L2)
        v_norm = F.normalize(v_proj, p=2, dim=-1)
        t_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
        # 3. Similitud Coseno
        # [B, N, 512] @ [512, Classes] -> [B, N, Classes]
        similarity_logits = torch.matmul(v_norm, t_norm.t()) / self.temperature
        
        # Devolvemos la similitud escalada por alpha
        return similarity_logits * self.alpha