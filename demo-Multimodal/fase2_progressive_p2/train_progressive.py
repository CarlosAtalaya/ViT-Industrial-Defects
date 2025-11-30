#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path

# Fix para warnings de HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"

if str(DEIMV2_PATH) not in sys.path:
    sys.path.insert(0, str(DEIMV2_PATH))

# --- IMPORTS DEIMv2 ---
import engine.data.transforms 
import engine.data.dataset
from engine.core import YAMLConfig

# --- IMPORTS PROPIOS ---
sys.path.insert(0, str(current_file.parent))
from models_utils.text_encoder import TextEncoder
from models_utils.multimodal_fusion import MultimodalFusionModule
from models_utils.deimv2_multimodal import DEIMv2Multimodal

TEXT_PROMPTS = [
    "Normal, clean surface without defects",
    "Surface deformation, dent or irregularity",
    "Fracture, crack, broken material or rupture",
    "Scratch, surface abrasion or line mark",
    "Perforation, hole or drilled spot",
    "Contamination, dirt, stain or foreign particle"
]

def setup_logger(output_dir):
    logger = logging.getLogger("DEIMv2_Multimodal")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    model.train()
    model.deimv2.eval() # Congelar detector base
    
    total_loss = 0
    num_batches = len(loader)
    
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        outputs = model(images, targets)
        
        loss_dict = criterion(outputs, targets)
        
        # Sumar todas las losses disponibles (focal, bbox, etc.)
        loss = sum(l for k, l in loss_dict.items() if 'loss' in k)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            logger.info(f"Epoch: {epoch} | Batch: {i}/{num_batches} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    return total_loss / num_batches

def main(args):
    cfg = YAMLConfig(args.config)
    output_dir = cfg.yaml_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info(f"🚀 Iniciando FASE {args.phase} - Opción 3")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Text Embeddings
    text_enc = TextEncoder(device=device)
    text_embeds = text_enc.encode_texts(TEXT_PROMPTS)
    
    # 2. Modelo Base
    deim_model = cfg.model.to(device)
    
    if args.phase == 1:
        ckpt_path = PROJECT_ROOT / "scripts/deimv2_multimodal/outputs/deimv2_1024_300epochs/best_stg1.pth"
    else:
        prev_phase = args.phase - 1
        ckpt_path = Path(f"./outputs/fase2_progressive_p{prev_phase}/final.pth")
    
    logger.info(f"📂 Cargando checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 3. Wrapper Multimodal
    fusion = MultimodalFusionModule(num_classes=6).to(device)
    model = DEIMv2Multimodal(deim_model, fusion, text_embeds).to(device)
    
    # Carga de pesos
    if args.phase == 1:
        model.deimv2.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)

    # 4. Congelación
    logger.info(f"❄️ FASE {args.phase}: Configurando capas...")
    for p in model.parameters(): p.requires_grad = False
    for p in model.fusion.parameters(): p.requires_grad = True
    
    if args.phase == 2:
        for name, p in model.deimv2.named_parameters():
            if 'score_head' in name or 'class_embed' in name:
                p.requires_grad = True
                
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   📊 Entrenables: {trainable:,}")

    # 5. Dataloader & Optimizer
    train_loader = cfg.train_dataloader
    criterion = cfg.criterion.to(device)
    
    # FIX OPTIMIZER FLAT CONFIG
    optim_key = cfg.yaml_cfg['optimizer']
    if isinstance(optim_key, str):
        optim_params = cfg.yaml_cfg[optim_key]
        lr_val = optim_params['lr']
        wd_val = optim_params.get('weight_decay', 0.0)
    else:
        lr_val = optim_key['lr']
        wd_val = optim_key.get('weight_decay', 0.0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_val,
        weight_decay=wd_val
    )
    
    # 6. Loop
    epochs = 10 
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        logger.info(f"🏁 Epoch {epoch} Final Loss: {avg_loss:.4f}")
        torch.save({'model': model.state_dict(), 'epoch': epoch}, f"{output_dir}/epoch_{epoch}.pth")
        
    torch.save({'model': model.state_dict(), 'phase': args.phase}, f"{output_dir}/final.pth")
    logger.info("✅ Fase completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)