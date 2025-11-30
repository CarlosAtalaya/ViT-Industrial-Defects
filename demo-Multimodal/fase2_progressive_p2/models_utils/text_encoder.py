import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        print(f"📦 Inicializando TextEncoder con {model_name}...")
        self.device = device
        
        # Cargar modelo y procesador de Hugging Face
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # CONGELACIÓN TOTAL: No queremos entrenar CLIP, solo usarlo.
        self.model.eval() 
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, text_list):
        """
        Genera embeddings normalizados para una lista de textos.
        Retorna: Tensor [Num_Clases, 512]
        """
        inputs = self.processor(text=text_list, return_tensors="pt", 
                               padding=True, truncation=True).to(self.device)
        
        outputs = self.model.get_text_features(**inputs)
        
        # Normalización L2 (Crucial para similitud coseno)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        return outputs.detach() # Desconectamos del grafo para ahorrar memoria