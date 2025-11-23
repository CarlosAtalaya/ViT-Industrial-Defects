"""
DEIMv2 Multimodal Wrapper
==========================

Integra detector DEIMv2 + DINOv3 con módulo de fusión multimodal.

Arquitectura:
    1. DEIMv2 + DINOv3: Detección base (congelado en Opción 1)
    2. Fusión multimodal: Mejora clasificación con embeddings texto
    3. Output: Detecciones con scores refinados
"""

import torch
import torch.nn as nn
import sys


class DEIMv2Multimodal(nn.Module):
    """
    Wrapper multimodal para DEIMv2.
    
    Mantiene arquitectura del detector y añade fusión texto-visual
    para refinar clasificación de defectos.
    """
    
    def __init__(
        self,
        deimv2_model,           # Modelo DEIMv2 preentrenado
        fusion_module,          # Módulo MultimodalFusionModule
        text_embeddings,        # [num_classes, 512] embeddings CLIP
        use_fusion=True         # Si False, usa solo DEIMv2 (para comparación)
    ):
        """
        Args:
            deimv2_model: Instancia de DEIMv2 cargada desde checkpoint
            fusion_module: Instancia de MultimodalFusionModule
            text_embeddings: Tensor con embeddings de texto [6, 512]
            use_fusion: Si True, usa fusión multimodal; si False, solo DEIMv2
        """
        super().__init__()
        
        self.deimv2 = deimv2_model
        self.fusion = fusion_module
        self.use_fusion = use_fusion
        
        # Registrar embeddings de texto como buffer (no trainable)
        self.register_buffer('text_embeddings', text_embeddings)
        
        print(f"\n{'='*70}")
        print("DEIMv2 Multimodal Wrapper Inicializado")
        print(f"{'='*70}")
        print(f"   Fusión activa: {use_fusion}")
        print(f"   Text embeddings: {text_embeddings.shape}")
        print(f"   Num classes: {text_embeddings.shape[0]}")
        print(f"{'='*70}\n")
    
    def forward(self, images, targets=None):
        """
        Forward pass completo.
        
        Args:
            images: [B, 3, H, W] imágenes de entrada
            targets: Lista de diccionarios con 'boxes' y 'labels' (training)
        
        Returns:
            Durante entrenamiento (targets != None):
                dict: {'loss_cls', 'loss_box', 'loss_total'}
            
            Durante inferencia (targets == None):
                list: Lista de predicciones por imagen [
                    {
                        'boxes': [N, 4],
                        'labels': [N],
                        'scores': [N]
                    }
                ]
        """
        
        # 1. Forward pass DEIMv2 (detección base)
        if targets is not None:
            # Training mode
            deimv2_outputs = self.deimv2(images, targets)
            
            if not self.use_fusion:
                # Sin fusión: retornar outputs originales
                return deimv2_outputs
            
            # Con fusión: necesitamos extraer features para refinar
            # Nota: Esto requiere modificar el forward del detector para
            # retornar features intermedias
            # Por ahora, entrenamos con loss original + loss de fusión
            
            # TODO: Implementar extracción de features y refinamiento
            # de clasificación durante training
            
            return deimv2_outputs
        
        else:
            # Inference mode
            deimv2_outputs = self.deimv2(images)
            
            if not self.use_fusion:
                return deimv2_outputs
            
            # Refinar clasificación con fusión multimodal
            refined_outputs = self._refine_predictions(deimv2_outputs)
            
            return refined_outputs
    
    def _refine_predictions(self, deimv2_outputs):
        """
        Refina scores de clasificación usando fusión multimodal.
        
        Args:
            deimv2_outputs: Lista de predicciones del detector
        
        Returns:
            list: Predicciones refinadas con scores multimodales
        """
        refined_predictions = []
        
        for pred in deimv2_outputs:
            if len(pred['boxes']) == 0:
                # Sin detecciones
                refined_predictions.append(pred)
                continue
            
            # Extraer features visuales de las cajas detectadas
            # Nota: Esto requiere acceso a features del decoder
            # Por simplicidad, usamos el embedding de la clase predicha
            
            # TODO: Implementar extracción real de features visuales
            # Por ahora, mantenemos scores originales
            
            refined_predictions.append(pred)
        
        return refined_predictions
    
    def freeze_detector(self):
        """Congela todos los parámetros del detector DEIMv2."""
        for param in self.deimv2.parameters():
            param.requires_grad = False
        print("❄️  Detector DEIMv2 congelado completamente")
    
    def freeze_backbone(self):
        """Congela solo el backbone (DINOv3)."""
        for param in self.deimv2.backbone.parameters():
            param.requires_grad = False
        print("❄️  Backbone DINOv3 congelado")
    
    def unfreeze_head(self):
        """Descongela cabeza de clasificación del detector."""
        # Esto depende de la estructura interna de DEIMv2
        # Típicamente es el decoder o la cabeza de clasificación
        if hasattr(self.deimv2, 'class_embed'):
            for param in self.deimv2.class_embed.parameters():
                param.requires_grad = True
            print("🔥 Cabeza de clasificación descongelada")
    
    def get_trainable_params(self):
        """Retorna estadísticas de parámetros entrenables."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        detector_params = sum(p.numel() for p in self.deimv2.parameters())
        detector_trainable = sum(p.numel() for p in self.deimv2.parameters() if p.requires_grad)
        
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        fusion_trainable = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'detector_total': detector_params,
            'detector_trainable': detector_trainable,
            'fusion_total': fusion_params,
            'fusion_trainable': fusion_trainable,
            'frozen_ratio': 1.0 - (trainable_params / total_params)
        }
    
    def print_params_summary(self):
        """Imprime resumen de parámetros."""
        stats = self.get_trainable_params()
        
        print(f"\n{'='*70}")
        print("RESUMEN DE PARÁMETROS")
        print(f"{'='*70}")
        print(f"Total:            {stats['total']:>15,} parámetros")
        print(f"Entrenables:      {stats['trainable']:>15,} ({stats['trainable']/stats['total']*100:.1f}%)")
        print(f"Congelados:       {stats['total']-stats['trainable']:>15,} ({stats['frozen_ratio']*100:.1f}%)")
        print(f"\nDetector DEIMv2:  {stats['detector_trainable']:>15,} / {stats['detector_total']:,}")
        print(f"Fusion Module:    {stats['fusion_trainable']:>15,} / {stats['fusion_total']:,}")
        print(f"{'='*70}\n")


def load_deimv2_checkpoint(checkpoint_path, device='cuda'):
    """
    Carga modelo DEIMv2 desde checkpoint.
    
    Args:
        checkpoint_path: Ruta al checkpoint .pth
        device: Dispositivo de cómputo
    
    Returns:
        model: Modelo DEIMv2 cargado
    """
    print(f"\n📂 Cargando checkpoint DEIMv2: {checkpoint_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Aquí necesitas inicializar el modelo DEIMv2 con la config correcta
    # y cargar los pesos del checkpoint
    
    # Esto requiere acceso al código de DEIMv2 y su config
    # Por ahora, retornamos un placeholder
    
    print("⚠️  TODO: Implementar carga real del modelo DEIMv2")
    print("    Requiere: config + build_model() de DEIMv2")
    
    return None  # Placeholder


def build_multimodal_model(
    deimv2_checkpoint_path,
    text_embeddings,
    visual_dim=256,
    text_dim=512,
    hidden_dim=256,
    num_classes=6,
    freeze_detector=True,
    device='cuda'
):
    """
    Constructor completo del modelo multimodal.
    
    Args:
        deimv2_checkpoint_path: Ruta al checkpoint best_stg1.pth
        text_embeddings: Tensor [6, 512] con embeddings de texto
        visual_dim: Dim features visuales
        text_dim: Dim embeddings texto (512 para CLIP)
        hidden_dim: Dim espacio común
        num_classes: Número de clases (6)
        freeze_detector: Si True, congela DEIMv2 completo
        device: Dispositivo
    
    Returns:
        DEIMv2Multimodal: Modelo completo listo para entrenar
    """
    from .multimodal_fusion import MultimodalFusionModule
    
    # 1. Cargar DEIMv2
    deimv2_model = load_deimv2_checkpoint(deimv2_checkpoint_path, device)
    
    # 2. Crear módulo de fusión
    fusion_module = MultimodalFusionModule(
        visual_dim=visual_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.1
    ).to(device)
    
    # 3. Crear wrapper multimodal
    model = DEIMv2Multimodal(
        deimv2_model=deimv2_model,
        fusion_module=fusion_module,
        text_embeddings=text_embeddings.to(device),
        use_fusion=True
    )
    
    # 4. Congelar detector si se especifica
    if freeze_detector:
        model.freeze_detector()
    
    # 5. Imprimir resumen
    model.print_params_summary()
    
    return model


if __name__ == "__main__":
    print("✅ DEIMv2Multimodal wrapper definido")
    print("⚠️  Requiere integración completa con DEIMv2 para funcionamiento real")