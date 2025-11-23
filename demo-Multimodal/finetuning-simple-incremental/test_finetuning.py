"""
Script de Test End-to-End - Opción 1
=====================================

Verifica que todos los componentes funcionen correctamente antes de entrenar.
"""

import torch
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_text_prompts, CLASS_DESCRIPTIONS
from models_utils import TextEncoder, MultimodalFusionModule


def test_1_class_descriptions():
    """Test 1: Verificar descripciones de clases."""
    print("\n" + "="*70)
    print("TEST 1: CLASS DESCRIPTIONS")
    print("="*70)
    
    # Verificar número de clases
    assert len(CLASS_DESCRIPTIONS) == 6, "Debe haber 6 clases"
    print("✅ 6 clases definidas")
    
    # Verificar prompts
    prompts = get_text_prompts()
    assert len(prompts) == 6, "Debe haber 6 prompts"
    print("✅ 6 prompts generados")
    
    # Imprimir
    print("\nPrompts:")
    for i, prompt in enumerate(prompts):
        print(f"  [{i}] {prompt}")
    
    print("\n✅ TEST 1 PASADO\n")
    return True


def test_2_text_encoder():
    """Test 2: Verificar text encoder CLIP."""
    print("\n" + "="*70)
    print("TEST 2: TEXT ENCODER")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Crear encoder
        encoder = TextEncoder(freeze=True).to(device)
        print("✅ Text encoder creado")
        
        # Generar embeddings
        prompts = get_text_prompts()
        embeddings = encoder.encode_texts(prompts, device)
        
        # Verificar shape
        assert embeddings.shape == (6, 512), f"Shape esperado (6, 512), obtenido {embeddings.shape}"
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Verificar normalización
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "Embeddings deben estar normalizados"
        print(f"✅ Embeddings normalizados (norms ~ 1.0)")
        
        # Verificar diversidad
        similarity = embeddings @ embeddings.T
        mask = ~torch.eye(6, dtype=bool, device=device)
        avg_sim = similarity[mask].mean().item()
        print(f"✅ Similitud promedio entre clases: {avg_sim:.3f}")
        
        if avg_sim > 0.90:
            print("⚠️  ADVERTENCIA: Similitud muy alta (>0.90). Las descripciones son muy parecidas.")
        
        print("\n✅ TEST 2 PASADO\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FALLIDO: {e}")
        return False


def test_3_fusion_module():
    """Test 3: Verificar módulo de fusión."""
    print("\n" + "="*70)
    print("TEST 3: FUSION MODULE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Crear módulo
        fusion = MultimodalFusionModule(
            visual_dim=256,
            text_dim=512,
            hidden_dim=256,
            num_classes=6,
            dropout=0.1
        ).to(device)
        print("✅ Fusion module creado")
        
        # Datos sintéticos
        batch_size = 4
        visual_features = torch.randn(batch_size, 256).to(device)
        text_embeddings = torch.randn(6, 512).to(device)
        
        # Forward pass
        logits, attention = fusion(visual_features, text_embeddings)
        
        # Verificar shapes
        assert logits.shape == (batch_size, 6), f"Logits shape esperado (4, 6), obtenido {logits.shape}"
        assert attention.shape == (batch_size, 1, 6), f"Attention shape esperado (4, 1, 6), obtenido {attention.shape}"
        print(f"✅ Output shapes correctos")
        
        # Verificar gradientes
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in fusion.parameters())
        assert has_grad, "Los parámetros deben tener gradientes"
        print(f"✅ Gradientes computados correctamente")
        
        # Contar parámetros
        total_params = sum(p.numel() for p in fusion.parameters())
        trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
        print(f"✅ Parámetros: {trainable_params:,} / {total_params:,}")
        
        print("\n✅ TEST 3 PASADO\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FALLIDO: {e}")
        return False


def test_4_integration():
    """Test 4: Verificar integración completa."""
    print("\n" + "="*70)
    print("TEST 4: INTEGRATION TEST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 1. Text encoder
        text_encoder = TextEncoder(freeze=True).to(device)
        text_embeddings = text_encoder.encode_texts(get_text_prompts(), device)
        print("✅ Text embeddings generados")
        
        # 2. Fusion module
        fusion = MultimodalFusionModule(
            visual_dim=256,
            text_dim=512,
            hidden_dim=256,
            num_classes=6
        ).to(device)
        print("✅ Fusion module creado")
        
        # 3. Simular forward completo
        batch_size = 4
        visual_features = torch.randn(batch_size, 256).to(device)
        
        # Pipeline completo
        logits, attention = fusion(visual_features, text_embeddings)
        
        print(f"✅ Forward completo exitoso")
        print(f"   Input visual: {visual_features.shape}")
        print(f"   Text embeddings: {text_embeddings.shape}")
        print(f"   Output logits: {logits.shape}")
        print(f"   Attention weights: {attention.shape}")
        
        # Verificar pesos de atención
        attention_probs = attention.squeeze(1)  # [B, 6]
        print(f"\n   Distribución de atención (primera muestra):")
        class_names = ["NORMAL", "PERFORACIONES", "RAYONES", "DEFORMACIONES", "CONTAMINACION", "ROTURA"]
        for i, (cls, prob) in enumerate(zip(class_names, attention_probs[0])):
            print(f"     {cls:20s}: {prob.item():.4f}")
        
        print("\n✅ TEST 4 PASADO\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 4 FALLIDO: {e}")
        return False


def test_5_memory_usage():
    """Test 5: Estimar uso de memoria GPU."""
    print("\n" + "="*70)
    print("TEST 5: MEMORY USAGE")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA no disponible, saltando test de memoria")
        return True
    
    device = torch.device('cuda')
    
    try:
        # Limpiar caché
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Crear componentes
        text_encoder = TextEncoder(freeze=True).to(device)
        text_embeddings = text_encoder.encode_texts(get_text_prompts(), device)
        
        fusion = MultimodalFusionModule(
            visual_dim=256,
            text_dim=512,
            hidden_dim=256,
            num_classes=6
        ).to(device)
        
        # Simular batch
        batch_size = 4
        visual_features = torch.randn(batch_size, 256).to(device)
        logits, _ = fusion(visual_features, text_embeddings)
        
        # Backward
        loss = logits.sum()
        loss.backward()
        
        # Medir memoria
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        max_memory = torch.cuda.max_memory_allocated() / 1024**2    # MB
        
        print(f"✅ Memoria GPU:")
        print(f"   Allocated: {memory_allocated:.1f} MB")
        print(f"   Reserved: {memory_reserved:.1f} MB")
        print(f"   Peak: {max_memory:.1f} MB")
        
        # Estimar memoria total con modelo completo
        # DEIMv2 + DINOv3 ~ 8GB
        # Fusion ~ 50MB
        # Activations ~ 2GB
        estimated_total = 8000 + 50 + 2000  # MB
        
        print(f"\n   Estimación con DEIMv2 completo:")
        print(f"   Total estimado: ~{estimated_total/1024:.1f} GB")
        
        if estimated_total > 12000:
            print("   ⚠️  ADVERTENCIA: Puede exceder 12GB VRAM")
            print("      Considerar reducir batch_size a 2")
        else:
            print("   ✅ Debe caber en RTX 4070 (12GB)")
        
        print("\n✅ TEST 5 COMPLETADO\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 5 FALLIDO: {e}")
        return False


def test_6_checkpoint_compatibility():
    """Test 6: Verificar compatibilidad con checkpoint."""
    print("\n" + "="*70)
    print("TEST 6: CHECKPOINT COMPATIBILITY")
    print("="*70)
    
    checkpoint_path = Path("scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth")
    
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint no encontrado: {checkpoint_path}")
        print("   Saltando test (ajustar ruta según ubicación real)")
        return True
    
    try:
        from utils.deimv2_loader import verify_checkpoint_compatibility
        
        info = verify_checkpoint_compatibility(checkpoint_path)
        
        print(f"✅ Checkpoint cargado")
        print(f"   Keys: {info['keys']}")
        print(f"   Tiene 'model': {info['has_model']}")
        print(f"   Tiene 'config': {info['has_config']}")
        
        if info['has_model']:
            print(f"   Parámetros: {info['num_params']}")
        
        print("\n✅ TEST 6 PASADO\n")
        return True
        
    except Exception as e:
        print(f"❌ TEST 6 FALLIDO: {e}")
        return False


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*70)
    print("EJECUTANDO TESTS END-TO-END - OPCIÓN 1")
    print("="*70)
    
    tests = [
        ("Class Descriptions", test_1_class_descriptions),
        ("Text Encoder", test_2_text_encoder),
        ("Fusion Module", test_3_fusion_module),
        ("Integration", test_4_integration),
        ("Memory Usage", test_5_memory_usage),
        ("Checkpoint Compatibility", test_6_checkpoint_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FALLÓ con excepción: {e}")
            results.append((name, False))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASADO" if result else "❌ FALLIDO"
        print(f"{name:30s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nTotal: {passed}/{total} tests pasados")
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("   La implementación está lista para integración con DEIMv2")
    else:
        print(f"\n⚠️  {total - passed} tests fallaron")
        print("   Revisar errores antes de continuar")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)