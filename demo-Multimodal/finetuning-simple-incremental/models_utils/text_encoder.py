"""
Text Encoder para Fusión Multimodal
====================================

Wrapper de CLIP (ViT-B/16) para generar embeddings de texto.
Compatible con descripciones de defectos industriales.

Modelo: openai/clip-vit-base-patch16 (512-dim embeddings)
"""

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoder(nn.Module):
    """
    Encoder de texto basado en CLIP para descripciones de defectos.
    
    Genera embeddings de 512 dimensiones normalizados.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze=True):
        """
        Args:
            model_name: Nombre del modelo CLIP de HuggingFace
            freeze: Si True, congela pesos del encoder (recomendado)
        """
        super().__init__()
        
        print(f"📝 Cargando CLIP Text Encoder: {model_name}")
        
        # Cargar tokenizer y modelo
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        
        # Congelar si se especifica
        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            print("   ❄️  Text encoder congelado (freeze=True)")
        
        self.embedding_dim = self.text_model.config.hidden_size  # 512 para ViT-B/16
        print(f"   ✓ Embedding dimension: {self.embedding_dim}")
    
    def encode_texts(self, text_prompts, device='cuda'):
        """
        Codifica lista de prompts de texto a embeddings.
        
        Args:
            text_prompts: Lista de strings (e.g., descripciones de clases)
            device: Dispositivo para computación
        
        Returns:
            torch.Tensor: [num_texts, 512] embeddings normalizados
        """
        # Tokenizar
        inputs = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=77,  # Longitud máxima CLIP
            return_tensors="pt"
        ).to(device)
        
        # Generar embeddings
        with torch.no_grad():
            text_features = self.text_model(**inputs).pooler_output  # [B, 512]
        
        # Normalizar L2 (estándar CLIP)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward(self, text_prompts, device='cuda'):
        """
        Forward pass para compatibilidad con training loops.
        
        Args:
            text_prompts: Lista de strings
            device: Dispositivo
        
        Returns:
            torch.Tensor: Embeddings normalizados [B, 512]
        """
        return self.encode_texts(text_prompts, device)
    
    def get_class_embeddings(self, class_descriptions_func, device='cuda'):
        """
        Genera embeddings para todas las clases del dataset.
        
        Args:
            class_descriptions_func: Función que retorna lista de prompts
                                      (e.g., get_text_prompts() de class_descriptions.py)
            device: Dispositivo
        
        Returns:
            torch.Tensor: [num_classes, 512] embeddings
        """
        text_prompts = class_descriptions_func()
        embeddings = self.encode_texts(text_prompts, device)
        
        print(f"\n✅ Embeddings de texto generados:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Norm range: [{embeddings.norm(dim=-1).min():.3f}, {embeddings.norm(dim=-1).max():.3f}]")
        
        return embeddings
    
    @property
    def device(self):
        """Retorna dispositivo del modelo."""
        return next(self.text_model.parameters()).device


def test_text_encoder():
    """Test de funcionamiento básico."""
    print("\n" + "="*70)
    print("TEST: Text Encoder")
    print("="*70 + "\n")
    
    # Importar descripciones
    import sys
    sys.path.append('/home/claude/demo-Multimodal/opcion1')
    from data.class_descriptions import get_text_prompts
    
    # Crear encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = TextEncoder(freeze=True)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Generar embeddings para las 6 clases
    text_prompts = get_text_prompts()
    print(f"\nPrompts a codificar ({len(text_prompts)}):")
    for i, prompt in enumerate(text_prompts):
        print(f"  [{i}] {prompt}")
    
    # Codificar
    print("\n🔄 Codificando prompts...")
    embeddings = encoder.get_class_embeddings(get_text_prompts, device)
    
    # Verificar ortogonalidad (diversidad semántica)
    print("\n📊 Análisis de Similitud Coseno:")
    similarity_matrix = embeddings @ embeddings.T
    
    # Extraer similitudes (excluyendo diagonal)
    mask = ~torch.eye(6, dtype=bool, device=device)
    similarities = similarity_matrix[mask]
    
    print(f"   Media: {similarities.mean():.3f}")
    print(f"   Min: {similarities.min():.3f}")
    print(f"   Max: {similarities.max():.3f}")
    print(f"   Std: {similarities.std():.3f}")
    
    # Pares más similares (potencial confusión)
    print("\n⚠️  Pares más similares (>0.85 = alta confusión):")
    class_names = ["NORMAL", "PERFORACIONES", "RAYONES", "DEFORMACIONES", "CONTAMINACION", "ROTURA"]
    for i in range(6):
        for j in range(i+1, 6):
            sim = similarity_matrix[i, j].item()
            if sim > 0.85:
                print(f"   {class_names[i]} ↔ {class_names[j]}: {sim:.3f}")
    
    print("\n✅ Test completado exitosamente")
    print("="*70 + "\n")
    
    return encoder, embeddings


if __name__ == "__main__":
    test_text_encoder()