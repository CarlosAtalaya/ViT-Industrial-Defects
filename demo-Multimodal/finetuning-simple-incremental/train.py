#!/usr/bin/env python3
"""
FASE 2 Training - Final
=======================

Entrena fusion multimodal usando DEIMv2 como base.
"""

import os
import sys
from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"

sys.path.insert(0, str(DEIMV2_PATH))
sys.path.insert(0, str(Path(__file__).parent))

# Imports
from engine.core import YAMLConfig
from models_utils import TextEncoder
from data import get_text_prompts
from deimv2_fusion_wrapper import build_deimv2_with_fusion

# Configuración
checkpoint_path = "scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth"
deimv2_config_path = "scripts/deimv2_multimodal/configs/deimv2_industrial_defects.yml"
output_dir = "demo-Multimodal/finetuning-simple-incremental/outputs/fase2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*70)
print("FASE 2: Entrenamiento Multimodal")
print("="*70)

# 1. Generar text embeddings
print("\n📝 Generando text embeddings...")
text_encoder = TextEncoder(freeze=True).to(device)
text_prompts = get_text_prompts()
text_embeddings = text_encoder.encode_texts(text_prompts, device)
print(f"✅ Text embeddings: {text_embeddings.shape}")

# 2. Construir modelo con fusion
print("\n🏗️  Construyendo modelo...")
model = build_deimv2_with_fusion(
    checkpoint_path=checkpoint_path,
    config_path=deimv2_config_path,
    text_embeddings=text_embeddings,
    device=device
)

# 3. Optimizer solo para fusion
optimizer = torch.optim.AdamW(
    [p for p in model.fusion.parameters() if p.requires_grad],
    lr=0.0001,
    weight_decay=0.0001
)

print(f"\n📊 Setup:")
print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"   Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

# 4. Cargar config para dataloaders
cfg = YAMLConfig(deimv2_config_path)

# TEMPORAL: num_workers=0 para evitar error multiprocessing
cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
cfg.yaml_cfg['val_dataloader']['num_workers'] = 0

train_loader = cfg.train_dataloader
val_loader = cfg.val_dataloader

print(f"\n📦 Data:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# 5. Training loop
os.makedirs(output_dir, exist_ok=True)

epochs = 40
best_map = 0.785  # Baseline

print(f"\n🚀 Comenzando entrenamiento:")
print(f"   Epochs: {epochs}")
print(f"   Target mAP: 0.80")
print(f"   Output: {output_dir}\n")

# Criterion y postprocessor de DEIMv2
criterion = cfg.criterion.to(device)
postprocessor = cfg.postprocessor

print(f"✅ Criterion: {type(criterion).__name__}")
print(f"✅ Postprocessor: {type(postprocessor).__name__}")

# Scaler para AMP
scaler = torch.cuda.amp.GradScaler() if cfg.yaml_cfg.get('use_amp', False) else None

# Training loop
model.train()
model.deimv2.eval()  # Detector frozen en eval

for epoch in range(epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*70}")
    
    # Train
    model.train()
    model.deimv2.eval()
    
    epoch_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images, targets)
                losses = criterion(outputs, targets)
                loss = sum(losses.values())
        else:
            outputs = model(images, targets)
            losses = criterion(outputs, targets)
            loss = sum(losses.values())
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: loss={loss.item():.4f}")
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"\n📊 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    # Eval cada 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"\n🔍 Evaluando...")
        model.eval()
        
        # TODO: Implementar evaluación COCO
        # Por ahora solo guardar checkpoint
        
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss
        }
        
        torch.save(checkpoint, f"{output_dir}/checkpoint_epoch{epoch+1}.pth")
        print(f"💾 Checkpoint guardado: epoch {epoch+1}")

# Guardar final
torch.save({
    'epoch': epochs,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}, f"{output_dir}/final.pth")

print(f"\n✅ Entrenamiento completado")
print(f"   Checkpoints en: {output_dir}")
print("="*70 + "\n")