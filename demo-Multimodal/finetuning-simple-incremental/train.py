#!/usr/bin/env python3
"""
FASE 2 Training - Test Run (5 Epochs)
=====================================
Entrenamiento incremental del módulo de fusión multimodal.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from pycocotools.cocoeval import COCOeval

# --- PATHS SETUP ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"

if str(DEIMV2_PATH) not in sys.path:
    sys.path.insert(0, str(DEIMV2_PATH))
sys.path.insert(0, str(current_file.parent))

# Imports del proyecto
from engine.core import YAMLConfig
from models_utils import TextEncoder
from data import get_text_prompts
from deimv2_fusion_wrapper import build_deimv2_with_fusion

# --- FIX: IMPORTAR MÓDULOS PARA REGISTRO ---
# Esto es vital: al importarlos, se ejecutan los @register y el config puede encontrar 'Resize', 'ToTensor', etc.
import engine.data.transforms 
import engine.data.dataset
# -------------------------------------------

# --- CONFIGURACIÓN DE LA PRUEBA ---
EPOCHS = 5
CHECKPOINT_SOURCE = "scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth"
# Asegúrate de que este path es correcto según donde guardaste el config corregido
CONFIG_PATH = "demo-Multimodal/finetuning-simple-incremental/configs/config.yml"
OUTPUT_DIR = "demo-Multimodal/finetuning-simple-incremental/outputs/fase2_test_run-5epochs"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_coco_map(model, loader, device, postprocessor):
    """Ejecuta validación COCO estándar."""
    model.eval()
    coco_gt = loader.dataset.coco
    results = []
    
    print(f"   Running COCO evaluation...", end="", flush=True)
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
            results_batch = postprocessor(outputs, orig_target_sizes)
            
            for i, (res, tgt) in enumerate(zip(results_batch, targets)):
                image_id = tgt["image_id"].item()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"].cpu().numpy()
                boxes = res["boxes"].cpu().numpy()
                
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                
                for score, label, box in zip(scores, labels, boxes):
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x) for x in box],
                        "score": float(score)
                    })
    
    if not results:
        print(" ⚠️ No detections found!")
        return [0.0] * 12

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    print(f" Done! -> mAP: {coco_eval.stats[0]:.4f}")
    return coco_eval.stats.tolist()

def main():
    print("\n" + "="*70)
    print(f"FASE 2: Entrenamiento Multimodal (TEST RUN: {EPOCHS} Epochs)")
    print("="*70)

    # 1. SETUP DE DATOS Y TEXTO
    print("\n📝 Generando embeddings de texto...")
    text_encoder = TextEncoder(freeze=True).to(DEVICE)
    prompts = get_text_prompts()
    text_embeddings = text_encoder.encode_texts(prompts, DEVICE)
    print(f"✅ Embeddings listos: {text_embeddings.shape}")

    # 2. CONSTRUCCIÓN DEL MODELO
    print("\n🏗️  Construyendo DEIMv2 + Fusión...")
    model = build_deimv2_with_fusion(
        checkpoint_path=CHECKPOINT_SOURCE,
        config_path=CONFIG_PATH,
        text_embeddings=text_embeddings,
        device=DEVICE
    )
    
    # 3. SETUP TRAINING
    optimizer = torch.optim.AdamW(
        [p for p in model.fusion.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-4
    )

    # Cargar Dataloaders
    # Al haber hecho los imports arriba, ahora 'cfg.train_dataloader' funcionará
    cfg = YAMLConfig(CONFIG_PATH)
    
    # Forzar workers a 0 por seguridad
    if 'train_dataloader' in cfg.yaml_cfg:
        cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
    if 'val_dataloader' in cfg.yaml_cfg:
        cfg.yaml_cfg['val_dataloader']['num_workers'] = 0
    
    print("\n📦 Construyendo Dataloaders (esto puede tardar unos segundos)...")
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    print(f"✅ Dataloaders listos. Batches train: {len(train_loader)}")
    
    criterion = cfg.criterion.to(DEVICE)
    postprocessor = cfg.postprocessor
    
    # Filtros de seguridad para losses
    if hasattr(criterion, 'aux_loss'): criterion.aux_loss = False
    classification_losses = ['focal', 'vfl', 'mal', 'ce', 'class', 'labels'] 
    criterion.losses = [l for l in criterion.losses if l in classification_losses]
    
    print(f"🔧 Losses activas: {criterion.losses}")
    if not criterion.losses:
        raise RuntimeError("❌ ERROR CRÍTICO: Todas las losses filtradas.")

    # 4. LOOP
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "log.txt")
    with open(log_path, "w") as f: pass

    print(f"\n🚀 Iniciando entrenamiento -> Output: {OUTPUT_DIR}\n")
    
    for epoch in range(EPOCHS):
        model.train()
        model.deimv2.eval() 
        
        epoch_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            outputs = model(images, targets)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        
        # Validación
        coco_stats = [0.0]*12
        if (epoch + 1) == EPOCHS or (epoch + 1) % 5 == 0:
            print(f"🔍 Validando Epoch {epoch+1}...")
            coco_stats = evaluate_coco_map(model, val_loader, DEVICE, postprocessor)
            
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mAP': coco_stats[0]
            }, f"{OUTPUT_DIR}/checkpoint_epoch{epoch+1}.pth")

        print(f"📊 Summary Epoch {epoch+1}: Avg Loss={avg_loss:.4f} | mAP={coco_stats[0]:.4f}")

        log_entry = {
            'epoch': epoch + 1,
            'train_lr': optimizer.param_groups[0]['lr'],
            'train_loss': avg_loss,
            'train_loss_mal': avg_loss,
            'train_loss_bbox': 0.0,
            'train_loss_giou': 0.0,
            'train_loss_fgl': 0.0,
            'test_coco_eval_bbox': coco_stats
        }
        
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final.pth")
    print(f"\n✅ Prueba completada.")

if __name__ == "__main__":
    main()