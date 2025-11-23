"""
Módulo de Fusión Multimodal Visión-Texto
=========================================

Combina features visuales del detector DEIMv2 con embeddings de texto CLIP
para mejorar clasificación de defectos industriales.

Arquitectura:
    - Input visual: [B, 256] features del detector
    - Input texto: [num_classes, 512] embeddings CLIP
    - Output: [B, num_classes] logits mejorados
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionModule(nn.Module):
    """
    Módulo de fusión que combina features visuales y texto mediante:
    1. Proyección de visual features a espacio común
    2. Atención cruzada visual-texto
    3. Clasificación refinada
    """
    
    def __init__(
        self,
        visual_dim=256,      # Dim features visuales del detector
        text_dim=512,        # Dim embeddings CLIP
        hidden_dim=256,      # Dim espacio común
        num_classes=6,
        dropout=0.1
    ):
        """
        Args:
            visual_dim: Dimensión de features visuales del detector
            text_dim: Dimensión de embeddings de texto (CLIP)
            hidden_dim: Dimensión del espacio común de proyección
            num_classes: Número de clases (6 para defectos industriales)
            dropout: Probabilidad de dropout para regularización
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Proyección visual: 256 -> hidden_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Proyección texto: 512 -> hidden_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Atención cruzada visual -> texto
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Cabeza de clasificación final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat visual + attended
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"✅ MultimodalFusionModule inicializado:")
        print(f"   Visual: {visual_dim} -> {hidden_dim}")
        print(f"   Text: {text_dim} -> {hidden_dim}")
        print(f"   Output: {num_classes} classes")
    
    def forward(self, visual_features, text_embeddings):
        """
        Forward pass de fusión multimodal.
        
        Args:
            visual_features: [B, visual_dim] features del detector por región
            text_embeddings: [num_classes, text_dim] embeddings de texto
        
        Returns:
            torch.Tensor: [B, num_classes] logits de clasificación
        """
        batch_size = visual_features.shape[0]
        
        # 1. Proyectar a espacio común
        visual_proj = self.visual_proj(visual_features)  # [B, hidden_dim]
        text_proj = self.text_proj(text_embeddings)      # [num_classes, hidden_dim]
        
        # 2. Expandir texto para batch
        # [num_classes, hidden_dim] -> [B, num_classes, hidden_dim]
        text_proj_expanded = text_proj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Atención cruzada: visual (query) atiende a texto (key, value)
        visual_query = visual_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        attended_features, attention_weights = self.cross_attention(
            query=visual_query,           # [B, 1, hidden_dim]
            key=text_proj_expanded,       # [B, num_classes, hidden_dim]
            value=text_proj_expanded      # [B, num_classes, hidden_dim]
        )
        
        attended_features = attended_features.squeeze(1)  # [B, hidden_dim]
        
        # 4. Concatenar visual original + attended
        fused_features = torch.cat([visual_proj, attended_features], dim=-1)  # [B, hidden_dim*2]
        
        # 5. Clasificación final
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        return logits, attention_weights
    
    def forward_similarity(self, visual_features, text_embeddings):
        """
        Forward alternativo basado solo en similitud coseno (más simple).
        Útil para comparación o debugging.
        
        Args:
            visual_features: [B, visual_dim]
            text_embeddings: [num_classes, text_dim]
        
        Returns:
            torch.Tensor: [B, num_classes] similitudes
        """
        # Proyectar
        visual_proj = self.visual_proj(visual_features)  # [B, hidden_dim]
        text_proj = self.text_proj(text_embeddings)      # [num_classes, hidden_dim]
        
        # Normalizar L2
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)
        text_norm = F.normalize(text_proj, p=2, dim=-1)
        
        # Similitud coseno
        similarity = visual_norm @ text_norm.T  # [B, num_classes]
        
        # Escalar (temperatura)
        temperature = 0.07  # Valor típico CLIP
        logits = similarity / temperature
        
        return logits
    
    def get_attention_weights(self, visual_features, text_embeddings):
        """
        Extrae pesos de atención para visualización.
        
        Returns:
            torch.Tensor: [B, num_classes] pesos de atención
        """
        with torch.no_grad():
            _, attention_weights = self.forward(visual_features, text_embeddings)
            # attention_weights: [B, 1, num_classes]
            return attention_weights.squeeze(1)  # [B, num_classes]


class SimpleFusionModule(nn.Module):
    """
    Versión simplificada: solo proyección + similitud coseno.
    Útil como baseline o si la versión completa es inestable.
    """
    
    def __init__(self, visual_dim=256, text_dim=512, hidden_dim=256, num_classes=6):
        super().__init__()
        
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)  # Learnable temperature
        
        print(f"✅ SimpleFusionModule inicializado (baseline)")
    
    def forward(self, visual_features, text_embeddings):
        """Forward con similitud coseno simple."""
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_embeddings)
        
        # Normalizar
        visual_norm = F.normalize(visual_proj, p=2, dim=-1)
        text_norm = F.normalize(text_proj, p=2, dim=-1)
        
        # Similitud
        logits = (visual_norm @ text_norm.T) / self.temperature
        
        return logits, None  # None para compatibilidad con API


def test_fusion_module():
    """Test de funcionamiento del módulo."""
    print("\n" + "="*70)
    print("TEST: Multimodal Fusion Module")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear módulo
    fusion = MultimodalFusionModule(
        visual_dim=256,
        text_dim=512,
        hidden_dim=256,
        num_classes=6,
        dropout=0.1
    ).to(device)
    
    # Datos sintéticos
    batch_size = 8
    visual_features = torch.randn(batch_size, 256).to(device)
    text_embeddings = torch.randn(6, 512).to(device)
    
    print(f"Input shapes:")
    print(f"   Visual: {visual_features.shape}")
    print(f"   Text: {text_embeddings.shape}")
    
    # Forward pass
    logits, attention = fusion(visual_features, text_embeddings)
    
    print(f"\nOutput shapes:")
    print(f"   Logits: {logits.shape}")
    print(f"   Attention: {attention.shape}")
    
    # Verificar gradientes
    loss = logits.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in fusion.parameters())
    print(f"\n✅ Gradientes computados: {has_grad}")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in fusion.parameters())
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    
    print(f"\nParámetros:")
    print(f"   Total: {total_params:,}")
    print(f"   Entrenables: {trainable_params:,}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_fusion_module()