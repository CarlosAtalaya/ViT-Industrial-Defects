"""
Utilidad para Cargar Checkpoint de DEIMv2
==========================================

Funciones helper para cargar modelo DEIMv2 desde checkpoint de FASE 1.
"""

import torch
import sys
from pathlib import Path

# Añadir path de DEIMv2
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / 'DEIMv2'


def load_deimv2_checkpoint(checkpoint_path, config_path=None, device='cuda', return_config=False):
    """
    Carga modelo DEIMv2 desde checkpoint de FASE 1.
    
    Args:
        checkpoint_path: Ruta a best_stg1.pth
        config_path: Ruta al archivo de config .yml (si es None, busca el default)
        device: Dispositivo ('cuda' o 'cpu')
        return_config: Si True, retorna también la config
    
    Returns:
        model: Modelo DEIMv2 cargado y en eval mode
        config (opcional): Configuración si return_config=True
    """
    print(f"\n{'='*70}")
    print("CARGANDO CHECKPOINT DEIMv2")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Verificar que existe checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    # Buscar config si no se proporciona
    if config_path is None:
        # Buscar en la ubicación estándar
        config_path = Path("scripts/deimv2_multimodal/configs/deimv2_industrial_defects.yml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config no encontrado en: {config_path}")
    
    print(f"Config: {config_path}")
    
    # Importar YAMLConfig de DEIMv2
    try:
        sys.path.insert(0, str(DEIMV2_PATH))
        from engine.core import YAMLConfig
    except ImportError as e:
        raise ImportError(f"No se puede importar DEIMv2. Asegúrate de que existe en: {DEIMV2_PATH}\nError: {e}")
    
    # Cargar configuración
    print("📋 Cargando configuración...")
    cfg = YAMLConfig(str(config_path))
    
    # Construir modelo desde config
    print("🏗️  Construyendo modelo...")
    model = cfg.model
    
    # Cargar checkpoint
    print("📂 Cargando pesos del checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"✅ Checkpoint cargado")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   Epoch: {checkpoint.get('last_epoch', 'N/A')}")
    
    # Cargar state dict
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✅ Modelo DEIMv2 cargado exitosamente")
    print(f"{'='*70}\n")
    
    if return_config:
        return model, cfg
    return model


def verify_checkpoint_compatibility(checkpoint_path):
    """
    Verifica que el checkpoint sea compatible y tenga la estructura esperada.
    
    Args:
        checkpoint_path: Ruta al checkpoint
    
    Returns:
        dict: Información del checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'keys': list(checkpoint.keys()),
        'has_model': 'model' in checkpoint,
        'has_config': 'config' in checkpoint or 'args' in checkpoint,
        'has_epoch': 'epoch' in checkpoint,
    }
    
    if info['has_model']:
        model_state = checkpoint['model']
        info['num_params'] = len(model_state.keys())
        info['param_shapes'] = {k: v.shape for k, v in list(model_state.items())[:5]}
    
    return info


def extract_features_from_deimv2(model, images):
    """
    Extrae features visuales intermedias del detector.
    
    Args:
        model: Modelo DEIMv2
        images: Tensor [B, 3, H, W]
    
    Returns:
        features: Tensor [B, N, 256] features por región
    """
    # TODO: Implementar extracción real
    # Requiere modificar forward de DEIMv2 para exponer decoder_features
    
    print("⚠️  extract_features_from_deimv2() no implementado")
    return None


def get_deimv2_config_for_industrial():
    """
    Retorna configuración de DEIMv2 para dataset industrial.
    
    Returns:
        dict: Configuración compatible con FASE 1
    """
    config = {
        # Modelo
        'backbone': 'dinov3_vitl14',
        'num_classes': 6,  # NORMAL, PERFORACIONES, RAYONES, DEFORMACIONES, CONTAMINACION, ROTURA
        'input_size': 1024,
        
        # Arquitectura
        'hidden_dim': 256,
        'num_queries': 300,
        'nheads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        
        # Loss weights
        'class_loss_coef': 2.0,
        'bbox_loss_coef': 5.0,
        'giou_loss_coef': 2.0,
        
        # Matching
        'set_cost_class': 2.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,
    }
    
    return config


def print_checkpoint_info(checkpoint_path):
    """Imprime información detallada del checkpoint."""
    print(f"\n{'='*70}")
    print("INFORMACIÓN DEL CHECKPOINT")
    print(f"{'='*70}")
    
    info = verify_checkpoint_compatibility(checkpoint_path)
    
    print(f"\nArchivo: {checkpoint_path}")
    print(f"Tamaño: {Path(checkpoint_path).stat().st_size / 1024**2:.1f} MB")
    
    print(f"\nContenido:")
    for key in info['keys']:
        print(f"  - {key}")
    
    print(f"\nVerificación:")
    print(f"  ✓ Tiene 'model': {info['has_model']}")
    print(f"  ✓ Tiene 'config': {info['has_config']}")
    print(f"  ✓ Tiene 'epoch': {info['has_epoch']}")
    
    if info['has_model']:
        print(f"\nModelo:")
        print(f"  Parámetros: {info['num_params']}")
        print(f"  Primeras shapes:")
        for k, v in info['param_shapes'].items():
            print(f"    {k}: {v}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Test básico
    checkpoint_path = "scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth"
    config_path = "scripts/deimv2_multimodal/configs/deimv2_industrial_defects.yml"
    
    if Path(checkpoint_path).exists():
        print_checkpoint_info(checkpoint_path)
        
        # Intentar cargar
        try:
            model, config = load_deimv2_checkpoint(
                checkpoint_path, 
                config_path=config_path,
                return_config=True
            )
            print("✅ Test de carga completado exitosamente")
            print(f"   Modelo: {type(model)}")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
    else:
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        print("   Ajustar ruta según ubicación real del checkpoint")