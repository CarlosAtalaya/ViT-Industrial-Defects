#!/usr/bin/env python3
"""
Script Maestro de Automatización - Fase 2 (Fine-tuning Progresivo)
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T
from PIL import Image

# Configuración de Entorno
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"

if str(DEIMV2_PATH) not in sys.path:
    sys.path.insert(0, str(DEIMV2_PATH))

# --- IMPORTS DEIMv2 ---
from engine.core import YAMLConfig
import engine.data.transforms 
import engine.data.dataset

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

def setup_scientific_logger(output_dir):
    logger = logging.getLogger("AutoPhase2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(os.path.join(output_dir, "experiment_log.txt"), mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

@torch.no_grad()
def evaluate_model(model, coco_gt, img_folder, device, score_thresh=0.01):
    model.eval()
    results = []
    transform = T.Compose([
        T.Resize([1024, 1024]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_ids = sorted(coco_gt.getImgIds())
    
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(img_folder, info['file_name'])
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue

        h_orig, w_orig = info['height'], info['width']
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        outputs = model(input_tensor)
        if 'pred_logits' not in outputs: continue
            
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        scores = pred_logits.softmax(-1).max(-1)[0]
        labels = pred_logits.softmax(-1).argmax(-1)
        
        boxes = pred_boxes.cpu()
        boxes[:, 0] *= w_orig; boxes[:, 2] *= w_orig
        boxes[:, 1] *= h_orig; boxes[:, 3] *= h_orig
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        
        scores = scores.cpu()
        labels = labels.cpu()
        
        for box, score, label in zip(boxes, scores, labels):
            if label < 6 and score >= score_thresh:
                results.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': box.tolist(),
                    'score': float(score)
                })
                
    if not results: return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    model.train()
    model.deimv2.eval() 
    total_loss = 0
    steps = len(loader)
    
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        outputs = model(images, targets)
        loss_dict = criterion(outputs, targets)
        loss = sum(l for k, l in loss_dict.items() if 'loss' in k)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 20 == 0:
            logger.info(f"   [Epoch {epoch}][Batch {i}/{steps}] Loss: {loss.item():.4f}")
            
    return total_loss / steps

def main(args):
    cfg = YAMLConfig(args.config)
    auto_cfg = cfg.yaml_cfg.get('automation', {})
    baseline_map = auto_cfg.get('baseline_map', 0.785)
    patience_limit = auto_cfg.get('patience', 6)
    min_delta = auto_cfg.get('min_delta', 0.001)
    max_epochs = auto_cfg.get('max_epochs', 50)
    
    output_dir = cfg.yaml_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_scientific_logger(output_dir)
    logger.info("="*60)
    logger.info("🚀 INICIANDO PIPELINE AUTOMATIZADO - FASE 2")
    logger.info(f"🎯 Meta: Superar mAP {baseline_map}")
    logger.info("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("🔧 Construyendo arquitectura multimodal...")
    text_enc = TextEncoder(device=device)
    text_embeds = text_enc.encode_texts(TEXT_PROMPTS)
    
    deim_model = cfg.model.to(device)
    fusion = MultimodalFusionModule(num_classes=6).to(device)
    model = DEIMv2Multimodal(deim_model, fusion, text_embeds).to(device)

    # --- CARGA DE PESOS INTELIGENTE ---
    p1_path = PROJECT_ROOT / "outputs/fase2_progressive_p1/final.pth"
    
    if args.resume:
        load_path = args.resume
        logger.info(f"🔄 Reanudando desde: {load_path}")
    else:
        load_path = p1_path
        logger.info(f"📥 Cargando resultado de Fase 1: {load_path}")

    if not os.path.exists(load_path):
        logger.error(f"❌ No se encuentra el checkpoint: {load_path}")
        return

    ckpt = torch.load(load_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # --- CORRECCIÓN CLAVE: strict=False para tolerar el nuevo parámetro alpha ---
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("✅ Pesos cargados correctamente (strict=False para nuevos parámetros).")
    except Exception as e:
        logger.error(f"❌ Error cargando pesos: {e}")
        return

    logger.info("🔓 Descongelando cabeceras de clasificación y módulo de fusión...")
    for p in model.parameters(): p.requires_grad = False
    for p in model.fusion.parameters(): p.requires_grad = True
    for name, p in model.deimv2.named_parameters():
        if 'score_head' in name or 'class_embed' in name:
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Parámetros entrenables: {trainable_params:,}")

    train_loader = cfg.train_dataloader
    criterion = cfg.criterion.to(device)
    
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
        lr=lr_val, weight_decay=wd_val
    )

    logger.info("📚 Cargando Ground Truth...")
    coco_gt = COCO(args.test_ann)

    best_map = 0.0
    patience_counter = 0
    start_epoch = 0
    
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"\n🌀 EPOCH {epoch+1}/{max_epochs}")
        
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        logger.info(f"   📉 Training Loss: {avg_loss:.4f}")
        
        logger.info("   🔎 Evaluando en Test Set...")
        current_map = evaluate_model(model, coco_gt, args.test_imgs, device)
        logger.info(f"   📈 mAP Actual: {current_map:.4f} (Récord: {best_map:.4f})")
        
        improved = current_map > (best_map + min_delta)
        
        if improved:
            best_map = current_map
            patience_counter = 0 
            save_name = f"best_phase2_auto_mAP_{best_map:.4f}.pth"
            save_path = os.path.join(output_dir, save_name)
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'mAP': best_map}, save_path)
            logger.info(f"   🏆 ¡NUEVO RÉCORD! Guardado en: {save_name}")
            if best_map > baseline_map:
                logger.info(f"   🔥 SUPERADO EL BASELINE ORIGINAL ({baseline_map}) 🔥")
        else:
            patience_counter += 1
            logger.info(f"   ⏳ Sin mejora. Paciencia: {patience_counter}/{patience_limit}")
        
        if patience_counter >= patience_limit:
            logger.info("\n🛑 EARLY STOPPING ACTIVADO")
            break
            
    logger.info("✅ Proceso finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test-imgs', type=str, required=True)
    parser.add_argument('--test-ann', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args)