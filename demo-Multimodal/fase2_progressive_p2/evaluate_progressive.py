#!/usr/bin/env python3
"""
Script de evaluación para DEIMv2 Multimodal (Opción 3).
Genera metrics JSON idénticas a los baselines para comparación directa en TFG.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
DEIMV2_PATH = PROJECT_ROOT / "DEIMv2"

if str(DEIMV2_PATH) not in sys.path:
    sys.path.insert(0, str(DEIMV2_PATH))

from engine.core import YAMLConfig

# --- IMPORTS MODELO MULTIMODAL ---
sys.path.insert(0, str(current_file.parent))
from models_utils.text_encoder import TextEncoder
from models_utils.multimodal_fusion import MultimodalFusionModule
from models_utils.deimv2_multimodal import DEIMv2Multimodal

# --- PROMPTS (Mismos que training) ---
# Nuevos Prompts enfocados en características visuales distintivas
TEXT_PROMPTS = [
    # 0: NORMAL
    "Flawless industrial metal surface, uniform texture, no anomalies, clean background.",
    
    # 1: DEFORMACIONES (Enfocarse en cambios de luz/geometría)
    "Dented metal surface, uneven geometry, warped area with light reflection distortion.",
    
    # 2: ROTURA_FRACTURA (Enfocarse en bordes irregulares y separación)
    "Fractured material, jagged edges, deep structural crack, broken component part with separation.",
    
    # 3: RAYONES_ARANAZOS (Enfocarse en linealidad y superficie)
    "Linear surface scratch, thin scored line, metal abrasion mark, surface scar.",
    
    # 4: PERFORACIONES (Enfocarse en contraste y forma circular)
    "Circular hole, drilled puncture, dark void spot, penetrating opening in material.",
    
    # 5: CONTAMINACION (Enfocarse en color y superposición)
    "Surface stain, oil residue, dirt patch, foreign discoloration spot on metal."
]

def load_multimodal_model(checkpoint_path, config_path, device):
    print(f"\n🔧 Construyendo arquitectura Multimodal...")
    
    # 1. Config base y modelo base
    cfg = YAMLConfig(config_path)
    base_model = cfg.model.to(device)
    
    # 2. Embeddings de texto
    print("📝 Generando embeddings de texto (TextEncoder)...")
    text_enc = TextEncoder(device=device)
    text_embeds = text_enc.encode_texts(TEXT_PROMPTS)
    
    # 3. Módulo de Fusión
    fusion = MultimodalFusionModule(num_classes=6).to(device)
    
    # 4. Ensamblar Wrapper
    model = DEIMv2Multimodal(base_model, fusion, text_embeds)
    model.to(device)
    model.eval()
    
    # 5. Cargar Pesos
    print(f"📂 Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Carga flexible
    try:
        model.load_state_dict(state_dict)
        print("✅ Pesos cargados correctamente.")
    except Exception as e:
        print(f"⚠️ Advertencia en carga de pesos: {e}")
        print("Intentando carga parcial...")
        model.load_state_dict(state_dict, strict=False)

    return model

@torch.no_grad()
def evaluate(model, coco_gt, img_folder, device, score_thresh=0.15):
    print("\n🚀 Iniciando inferencia en Test Set...")
    results = []
    
    transform = T.Compose([
        T.Resize([1024, 1024]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_ids = sorted(coco_gt.getImgIds())
    
    for idx, img_id in enumerate(img_ids):
        if idx % 50 == 0: print(f"  Procesando {idx}/{len(img_ids)}...")
        
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(img_folder, info['file_name'])
        
        img = Image.open(path).convert('RGB')
        h_orig, w_orig = info['height'], info['width']
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward Multimodal
        outputs = model(input_tensor)
        
        # Procesar salidas
        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]
        
        scores = pred_logits.softmax(-1).max(-1)[0]
        labels = pred_logits.softmax(-1).argmax(-1)
        
        # Convertir boxes a COCO [x,y,w,h] absoluto
        boxes = pred_boxes.cpu()
        boxes[:, 0] *= w_orig # cx
        boxes[:, 1] *= h_orig # cy
        boxes[:, 2] *= w_orig # w
        boxes[:, 3] *= h_orig # h
        
        boxes[:, 0] -= boxes[:, 2] / 2 # x
        boxes[:, 1] -= boxes[:, 3] / 2 # y
        
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
                
    return results

def calculate_metrics_comparable(coco_gt, results, iou_thresh=0.5):
    if not results: return {}, {}
    
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.array([iou_thresh])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Métricas Globales
    metrics = {
        'mAP': float(coco_eval.stats[0]),
        'AP50': float(coco_eval.stats[1]),
        'AP75': float(coco_eval.stats[2])
    }
    
    # Métricas por Clase
    per_class_metrics = {}
    for cat_id, cat_info in enumerate(coco_gt.cats.values()):
        cat_name = cat_info['name']
        
        # Extraer stats internos de cocoeval
        # Dim: [iou, recall, class, area, maxDets]
        precision = coco_eval.eval['precision'][0, :, cat_id, 0, 2]
        recall_val = coco_eval.eval['recall'][0, cat_id, 0, 2]
        
        ap = np.mean(precision[precision > -1])
        prec = np.max(precision[precision > -1]) if len(precision[precision > -1]) > 0 else 0.0
        rec = float(recall_val) if recall_val > -1 else 0.0
        
        per_class_metrics[cat_name] = {
            'AP': float(ap), 
            'precision': float(prec), 
            'recall': float(rec)
        }
        
    return metrics, per_class_metrics

def save_comparable_json(metrics, per_class, output_dir, num_imgs, iou_t, score_t):
    data = {
        'mAP': metrics['mAP'],
        'iou_threshold': iou_t,
        'score_threshold': score_t,
        'num_test_images': num_imgs,
        'AP_per_class': {k: v['AP'] for k,v in per_class.items()},
        'precision_per_class': {k: v['precision'] for k,v in per_class.items()},
        'recall_per_class': {k: v['recall'] for k,v in per_class.items()}
    }
    
    out_path = os.path.join(output_dir, 'test_evaluation_results_comparable.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ JSON Comparable guardado en: {out_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Cargar modelo
    model = load_multimodal_model(args.checkpoint, args.config, device)
    
    # 2. Cargar GT
    coco_gt = COCO(args.test_ann)
    
    # 3. Evaluar
    results = evaluate(model, coco_gt, args.test_imgs, device, args.score_threshold)
    
    # 4. Calcular Métricas
    metrics, per_class = calculate_metrics_comparable(coco_gt, results, args.iou_threshold)
    
    # 5. Guardar
    output_dir = os.path.dirname(args.checkpoint)
    save_comparable_json(metrics, per_class, output_dir, len(coco_gt.getImgIds()), 
                         args.iou_threshold, args.score_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True) # Usar el config YAML de fase 1 o 2
    parser.add_argument('--test-imgs', required=True)
    parser.add_argument('--test-ann', required=True)
    parser.add_argument('--score-threshold', type=float, default=0.15)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    main(args)